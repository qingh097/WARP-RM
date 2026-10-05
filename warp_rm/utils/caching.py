"""
Disk-based feature pre-computation.

Extracts backbone features once and caches them as .npy files.
Shared across all experiments on the same machine.

Optimized for high GPU utilization via pipelined CPU decode + GPU inference.
"""

import hashlib
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..data.dataset import Episode
from ..data.preprocess import resize_frame
from ..data.video_reader import read_frames


def _dataset_name_from_video_path(video_path: Path) -> str:
    """Extract the dataset directory name from a video path.

    For .../<dataset>/videos/...: returns '<dataset>'.
    For .../<org>/<dataset>/videos/...: returns '<org>__<dataset>' (joined
    with '__' so the result is one shell-friendly path segment, and two
    different orgs hosting datasets with the same leaf name don't collide).
    Falls back to 'unknown' when no 'videos/' segment exists.
    """
    parts = video_path.parts
    if "videos" not in parts:
        return "unknown"
    idx = parts.index("videos")
    # Walk up from videos/ until we hit something that looks like a mount
    # prefix (data, lerobot, .cache, huggingface, /, home, etc).
    _MOUNT_TOKENS = {"", "/", "data", "lerobot", ".cache", "huggingface",
                     "home", "Users", "tmp", "root"}
    name_parts: list[str] = []
    for p in reversed(parts[:idx]):
        if p in _MOUNT_TOKENS or p.startswith("/"):
            break
        name_parts.append(p)
    if not name_parts:
        return "unknown"
    name_parts.reverse()
    return "__".join(name_parts)


def _ep_camera_video(ep: Episode, camera: str | None) -> tuple[Path, int]:
    """Resolve (video_path, frame_offset) for a given camera on an episode.

    ``camera=None`` returns the episode's primary (ep.video_path, ep.frame_offset)
    — byte-identical to the pre-multicam behavior so existing caches resolve.
    When ``camera`` is set, look it up in ``ep.camera_videos`` (each camera
    carries its OWN frame_offset, because on v3.0 concatenated videos different
    cameras can have different from_timestamps). Falls back to the primary video
    if the camera isn't present (single-cam episodes).
    """
    if camera is not None and ep.camera_videos and camera in ep.camera_videos:
        vp, off = ep.camera_videos[camera]
        return Path(vp), int(off)
    return Path(ep.video_path), int(ep.frame_offset)


def _ep_cache_key_stable(ep: Episode, backbone: str, feature_stride: int,
                         camera: str | None = None,
                         crop_mode: str = "squash", aug_tag: str | None = None) -> str:
    """Path-stable hash: same key regardless of mount prefix.

    Shared caches across machines (local ~/.cache vs cloud /data) resolve
    to the same .npy. We hash the path tail from one level above 'videos/'
    — sufficient to identify repo + episode without encoding mount prefix.

    For v3.0 concatenated videos (frame_offset > 0), the offset is included
    in the hash so episodes sharing a video file get distinct cache keys.

    ``camera`` selects which camera's video path/offset to hash. None ==
    legacy single-camera behavior (uses ep.video_path), so existing
    top-camera caches resolve to the same key. Different cameras live under
    ``videos/<camera>/...`` so their path tails — and thus keys — differ
    naturally; no extra camera token is mixed in.

    ``crop_mode`` is mixed in ONLY when it differs from the historical
    "squash" default, so every pre-existing cache keeps resolving to the same
    key. Without this token, center-cropped and squashed features — genuinely
    different pixels — would collide under one key and silently poison a run.
    """
    vpath, frame_offset = _ep_camera_video(ep, camera)
    parts = vpath.parts
    if "videos" in parts:
        idx = parts.index("videos")
        start = max(0, idx - 1)
        rel = "/".join(parts[start:])
    else:
        rel = str(vpath)
    if frame_offset:
        rel += f"@{frame_offset}"
    if crop_mode and crop_mode != "squash":
        rel += f"#{crop_mode}"
    if aug_tag:
        rel += f"#aug{aug_tag}"   # augmented variants get their own cache files
    return hashlib.md5(f"{rel}:{backbone}:{feature_stride}".encode()).hexdigest()[:12]


def _ep_cache_key_legacy(ep: Episode, backbone: str, feature_stride: int,
                         camera: str | None = None) -> str:
    """Old hash keyed by absolute video_path. Kept for backward compat."""
    vpath, _ = _ep_camera_video(ep, camera)
    return hashlib.md5(
        f"{vpath}:{backbone}:{feature_stride}".encode()
    ).hexdigest()[:12]


def _ep_cache_path(cache_dir: str, ep: Episode, backbone: str,
                   *_a, aug_tag: str | None = None, **_k):
    return _ep_cache_path_impl(cache_dir, ep, backbone, *_a, aug_tag=aug_tag, **_k)


def _ep_cache_path_impl(cache_dir: str, ep: Episode, backbone: str,
                   feature_stride: int, camera: str | None = None,
                   crop_mode: str = "squash", aug_tag: str | None = None) -> Path:
    """Return the cache path for an episode (optionally for a specific camera).

    Layout: <cache_dir>/<dataset_name>/<key>.npy. Dataset namespace prevents
    cross-dataset bleed and makes manual `aws s3 sync` of a single dataset's
    cache straightforward.

    ``camera`` makes the path camera-specific: both the dataset namespace and
    the stable key are derived from THAT camera's video path (which contains
    ``videos/<camera>/...``), so each camera caches separately. ``camera=None``
    is the legacy default and resolves identically to pre-multicam writes — so
    existing top-camera caches still hit.

    Lookup falls back through (in order):
      1. <cache_dir>/<dataset>/<stable_key>.npy   (canonical, new writes)
      2. <cache_dir>/<stable_key>.npy             (pre-namespaced flat caches)
      3. <cache_dir>/<legacy_abs_key>.npy         (very old absolute-path caches)
    Order matters — once a dataset has been re-precached into the namespaced
    layout, the older flat copies become stale and should be deleted.
    """
    cdir = Path(cache_dir)
    vpath, _ = _ep_camera_video(ep, camera)
    dataset = _dataset_name_from_video_path(vpath)
    stable_key = _ep_cache_key_stable(ep, backbone, feature_stride, camera, crop_mode, aug_tag)
    namespaced = cdir / dataset / f"{stable_key}.npy"
    if namespaced.exists():
        return namespaced
    flat_stable = cdir / f"{stable_key}.npy"
    if flat_stable.exists():
        return flat_stable
    # Legacy absolute-path keys predate crop modes, so they can only ever be
    # squash-mode caches — never resolve a non-default crop_mode to one.
    if crop_mode == "squash":
        legacy = cdir / f"{_ep_cache_key_legacy(ep, backbone, feature_stride, camera)}.npy"
        if legacy.exists():
            return legacy
    # Nothing on disk — write to the namespaced path
    return namespaced


def _make_episode_aug(ep: Episode, aug_tag: str):
    """Per-episode-CONSISTENT visual augmentation (same params on every frame so
    the motion signal is untouched): colour jitter, random crop/scale, optional
    blur / grayscale. Deterministic in (episode path, aug_tag)."""
    import hashlib, cv2
    seed = int(hashlib.md5(f"{ep.path}#{aug_tag}".encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    bright = rng.uniform(0.7, 1.3); contrast = rng.uniform(0.7, 1.3); sat = rng.uniform(0.6, 1.4)
    gains = rng.uniform(0.92, 1.08, size=3)          # mild per-channel colour cast
    scale = rng.uniform(0.75, 1.0); ox, oy = rng.uniform(0, 1), rng.uniform(0, 1)
    blur = rng.rand() < 0.3; gray = rng.rand() < 0.1
    def f(frame):
        h, w = frame.shape[:2]; ch, cw = int(h * scale), int(w * scale)
        y0, x0 = int((h - ch) * oy), int((w - cw) * ox)
        frame = frame[y0:y0 + ch, x0:x0 + cw]
        x = frame.astype(np.float32)
        if gray:
            g = x.mean(axis=2, keepdims=True); x = np.repeat(g, 3, axis=2)
        else:
            m = x.mean(axis=2, keepdims=True); x = m + (x - m) * sat
        x = x * gains
        x = (x - x.mean()) * contrast + x.mean()
        x = x * bright
        x = np.clip(x, 0, 255).astype(np.uint8)
        if blur:
            x = cv2.GaussianBlur(x, (5, 5), 0)
        return x
    return f


def _decode_episode(ep: Episode, feature_stride: int, image_size: int,
                    mean: np.ndarray, std: np.ndarray,
                    camera: str | None = None,
                    crop_mode: str = "squash", aug_tag: str | None = None) -> np.ndarray:
    """Decode + preprocess all strided frames for one episode. Returns (N, C, H, W).

    ``camera`` selects which camera's video + frame_offset to decode. None ==
    the episode's primary video (legacy single-camera behavior).
    ``crop_mode`` selects the frame geometry (see `warp_rm.data.preprocess`).
    """
    video_path, frame_offset = _ep_camera_video(ep, camera)
    n_feat = (ep.n_frames + feature_stride - 1) // feature_stride
    frame_indices = [min(j * feature_stride, ep.n_frames - 1) for j in range(n_feat)]
    # Apply per-camera frame_offset for v3.0 concatenated videos
    abs_indices = [i + frame_offset for i in frame_indices] if frame_offset else frame_indices
    frames_raw = read_frames(video_path, abs_indices)
    aug = _make_episode_aug(ep, aug_tag) if aug_tag else None
    processed = []
    for frame in frames_raw:
        if aug is not None:
            frame = aug(frame)
        frame = resize_frame(frame, image_size, crop_mode)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - mean) / std
        processed.append(frame)

    return np.stack(processed).transpose(0, 3, 1, 2)  # (N, C, H, W)


def _precompute_one_camera(
    episodes: list[Episode],
    encoder: nn.Module,
    device: torch.device,
    feature_stride: int,
    image_size: int,
    cache_dir: str,
    backbone: str,
    batch_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    decode_workers: int,
    mega_batch_episodes: int,
    camera: str | None = None,
    crop_mode: str = "squash",
    aug_tag: str | None = None,
) -> dict:
    """Extract + cache ONE camera's features. Returns
    {str(ep.path): {"cache_path", "n_frames", "n_feat"}}.

    ``camera=None`` is the legacy single-camera path (uses ep.video_path).
    See ``precompute_features`` for the multi-camera wrapper.
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Pre-create per-dataset subdirs so the parallel writer doesn't race.
    seen_datasets = set()
    for ep in episodes:
        vpath, _ = _ep_camera_video(ep, camera)
        dataset = _dataset_name_from_video_path(vpath)
        if dataset not in seen_datasets:
            os.makedirs(Path(cache_dir) / dataset, exist_ok=True)
            seen_datasets.add(dataset)
    t0 = time.time()

    ep_meta, to_extract = {}, []
    for ep in episodes:
        cp = _ep_cache_path(cache_dir, ep, backbone, feature_stride, camera, crop_mode, aug_tag=aug_tag)
        if cp.exists():
            n_feat = int(np.load(str(cp), mmap_mode="r").shape[0])
            ep_meta[str(ep.path)] = {
                "cache_path": str(cp), "n_frames": ep.n_frames, "n_feat": n_feat,
            }
        else:
            to_extract.append(ep)

    cam_tag = f" camera={camera}" if camera is not None else ""
    print(f"Pre-computing {backbone} features{cam_tag}: {len(ep_meta)} cached, "
          f"{len(to_extract)} to extract "
          f"(workers={decode_workers}, gpu_batch={batch_size}, "
          f"mega={mega_batch_episodes} eps)")

    if not to_extract:
        print(f"Precompute done{cam_tag}: {time.time() - t0:.1f}s (all cached)")
        return ep_meta

    encoder.eval().half()
    pool = ThreadPoolExecutor(max_workers=decode_workers)

    # Split into mega-batch chunks
    chunks = []
    for i in range(0, len(to_extract), mega_batch_episodes):
        chunks.append(to_extract[i:i + mega_batch_episodes])

    def _decode_chunk(chunk_eps):
        """Parallel-decode all episodes in a chunk. Returns list of (N,C,H,W) arrays."""
        futures = {
            pool.submit(_decode_episode, ep, feature_stride, image_size, mean, std, camera, crop_mode, aug_tag): i
            for i, ep in enumerate(chunk_eps)
        }
        decoded = [None] * len(chunk_eps)
        for future in as_completed(futures):
            decoded[futures[future]] = future.result()
        return decoded

    # Pipeline: start decoding chunk 0 immediately
    decode_result = [None]  # mutable container for thread result
    decode_event = threading.Event()

    def _decode_async(chunk_eps):
        decode_result[0] = _decode_chunk(chunk_eps)
        decode_event.set()

    # Start first decode
    decode_event.clear()
    t = threading.Thread(target=_decode_async, args=(chunks[0],), daemon=True)
    t.start()

    total_done = 0
    t_decode_wait = 0.0
    t_gpu = 0.0
    t_save = 0.0

    for ci, chunk_eps in enumerate(chunks):
        # Wait for current chunk decode to finish
        td0 = time.time()
        decode_event.wait()
        t_decode_wait += time.time() - td0

        decoded = decode_result[0]
        decode_event.clear()

        # Start decoding NEXT chunk in background (pipeline overlap)
        if ci + 1 < len(chunks):
            t = threading.Thread(target=_decode_async, args=(chunks[ci + 1],), daemon=True)
            t.start()

        # Concatenate frames → mega-batch
        frame_counts = [d.shape[0] for d in decoded]
        mega_frames = np.concatenate(decoded, axis=0)
        del decoded

        # GPU forward pass (fp16 for ~2x throughput on frozen encoder)
        tg0 = time.time()
        all_features = []
        for bi in range(0, len(mega_frames), batch_size):
            batch = mega_frames[bi:bi + batch_size]
            imgs_t = torch.from_numpy(batch).half().to(device)
            with torch.no_grad():
                feats = encoder(imgs_t).float().cpu().numpy()
            all_features.append(feats)
        all_features = np.concatenate(all_features, axis=0)
        t_gpu += time.time() - tg0
        del mega_frames

        # Split and save per episode
        ts0 = time.time()
        offset = 0
        for i, ep in enumerate(chunk_eps):
            n_feat = frame_counts[i]
            cp = _ep_cache_path(cache_dir, ep, backbone, feature_stride, camera, crop_mode, aug_tag=aug_tag)
            np.save(str(cp), all_features[offset:offset + n_feat])
            offset += n_feat
            ep_meta[str(ep.path)] = {
                "cache_path": str(cp), "n_frames": ep.n_frames, "n_feat": n_feat,
            }
        t_save += time.time() - ts0
        del all_features

        total_done += len(chunk_eps)
        elapsed = time.time() - t0
        rate = total_done / elapsed
        eta = (len(to_extract) - total_done) / rate if rate > 0 else 0
        print(f"  [{total_done}/{len(to_extract)}] {elapsed:.0f}s "
              f"({rate:.1f} ep/s, ETA {eta:.0f}s) "
              f"[decode_wait={t_decode_wait:.1f}s gpu={t_gpu:.1f}s save={t_save:.1f}s]")

    pool.shutdown(wait=False)
    print(f"Precompute done{cam_tag}: {time.time() - t0:.1f}s "
          f"({len(to_extract) / (time.time() - t0):.1f} ep/s)")
    return ep_meta


def precompute_features(
    episodes: list[Episode],
    encoder: nn.Module,
    device: torch.device,
    feature_stride: int,
    image_size: int,
    cache_dir: str,
    backbone: str = "dinov3",
    batch_size: int = 1024,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    decode_workers: int = 16,
    mega_batch_episodes: int = 8,
    cameras: list[str] | None = None,
    crop_mode: str = "squash",
    aug_tag: str | None = None,
) -> dict:
    """
    Pre-extract backbone features to disk; skips already-cached episodes.

    Pipelined architecture (see ``_precompute_one_camera``):
      - CPU thread pool decodes mega-batch N+1 while GPU processes mega-batch N
      - Each mega-batch = `mega_batch_episodes` episodes decoded in parallel
      - GPU processes all frames from a mega-batch in one shot (chunked by batch_size)
      - ~100% GPU utilization after first mega-batch is decoded

    Multi-camera: ``cameras`` is a list of camera keys. Each camera is
    extracted + cached separately (one .npy per camera per episode), then the
    per-camera meta is merged into a single ep_meta carrying ``cache_paths``
    (ordered to match ``cameras``). ``cameras=None`` (default) is the legacy
    single-camera path: it caches the episode's primary video and still emits a
    one-element ``cache_paths`` so downstream fusion code is uniform.

    Returns ep_meta dict:
        {str(ep.path): {"cache_path", "cache_paths", "n_frames", "n_feat"}}
      - ``cache_path``: first camera's .npy (back-compat with single-cam readers)
      - ``cache_paths``: list of per-camera .npy paths (len == len(cameras))
      - ``n_feat``: min n_feat across cameras (cameras are aligned; the min
        guards the rare ragged case so fused arrays stay consistent with labels)
    """
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    if std is None:
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # cameras=None → single-camera legacy path (camera=None inside).
    camera_list: list[str | None] = list(cameras) if cameras else [None]

    # Extract each camera independently (reuses the full pipelined extractor).
    per_camera_meta: list[dict] = []
    for cam in camera_list:
        per_camera_meta.append(_precompute_one_camera(
            episodes, encoder, device, feature_stride, image_size, cache_dir,
            backbone, batch_size, mean, std, decode_workers, mega_batch_episodes,
            camera=cam, crop_mode=crop_mode, aug_tag=aug_tag,
        ))

    # Merge per-camera metas into one ep_meta carrying cache_paths.
    ep_meta: dict = {}
    for ep in episodes:
        key = str(ep.path)
        metas = [m.get(key) for m in per_camera_meta]
        if any(m is None for m in metas):
            # An episode missing for some camera (e.g. its video was absent) —
            # skip entirely so we never fuse a partial camera set.
            continue
        cache_paths = [m["cache_path"] for m in metas]
        n_feat = min(m["n_feat"] for m in metas)
        ep_meta[key] = {
            "cache_path": cache_paths[0],   # back-compat single-cam key
            "cache_paths": cache_paths,
            "n_frames": metas[0]["n_frames"],
            "n_feat": n_feat,
        }
    return ep_meta

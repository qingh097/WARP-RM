"""
Episode discovery for LeRobot v2.1 and v3.0 datasets.

Given a LeRobot dataset root (the directory under
``~/.cache/huggingface/lerobot/<org>/<name>/`` that contains ``meta/``,
``data/`` and ``videos/`` subdirectories), construct ``Episode`` objects
pointing at the camera MP4s so the rest of the WARP-RM training stack can
consume them without modification.

Supported on-disk layouts:

  **v2.1** (``codebase_version`` < ``v3.0``):
    meta/info.json              -> video_path template, fps, features
    meta/episodes.jsonl         -> one JSON per line: {episode_index, length, ...}
    videos/chunk-XXX/<key>/episode_YYYYYY.mp4   (one video per episode)

  **v3.0** (``codebase_version`` >= ``v3.0``):
    meta/info.json              -> video_path template, fps, features
    meta/episodes/chunk-XXX/file-YYY.parquet    (sharded episode metadata)
    videos/<key>/chunk-XXX/file-YYY.mp4         (concatenated videos, many episodes per file)

See https://github.com/huggingface/lerobot for the full spec.
"""

import json
from pathlib import Path
from typing import Optional

from .dataset import Episode


def _load_info(repo_path: Path) -> dict:
    info_path = repo_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"LeRobot info.json not found at {info_path}. "
            f"Expected the root of a LeRobot dataset "
            f"(containing meta/, data/, videos/)."
        )
    with info_path.open() as f:
        return json.load(f)


def _is_v3(info: dict) -> bool:
    ver = info.get("codebase_version", "v2.0")
    # v3.0, v3.1, etc.
    return ver >= "v3.0"


def _iter_episode_meta_jsonl(repo_path: Path):
    """v2.1: one JSON object per line in episodes.jsonl."""
    ep_path = repo_path / "meta" / "episodes.jsonl"
    if not ep_path.exists():
        raise FileNotFoundError(f"LeRobot episodes.jsonl not found at {ep_path}")
    with ep_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_episode_meta_parquet(repo_path: Path):
    """v3.0: sharded parquet files under meta/episodes/."""
    import pyarrow.parquet as pq

    episodes_dir = repo_path / "meta" / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(
            f"LeRobot v3.0 episodes dir not found at {episodes_dir}"
        )
    parquet_files = sorted(episodes_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files under {episodes_dir}"
        )
    for pf in parquet_files:
        if pf.name.endswith(".bak"):
            continue
        table = pq.read_table(str(pf))
        df = table.to_pandas()
        for _, row in df.iterrows():
            yield dict(row)


def _iter_episode_meta(repo_path: Path, info: dict):
    if _is_v3(info):
        yield from _iter_episode_meta_parquet(repo_path)
    else:
        yield from _iter_episode_meta_jsonl(repo_path)


def _parse_split_range(split_str: str) -> tuple[int, int]:
    """Parse a LeRobot split range like ``"0:6494"`` into (start, end_exclusive)."""
    parts = split_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"invalid split range: {split_str!r}")
    return int(parts[0]), int(parts[1])


def _detect_episode_list_splits(root: Path) -> dict[str, str]:
    """Detect ``{split}_episodes.json`` files containing explicit episode index
    lists (non-contiguous splits). Returns a splits dict with count-based
    descriptions like ``"6228 eps"``."""
    out: dict[str, str] = {}
    for p in sorted(root.glob("*_episodes.json")):
        name = p.stem.replace("_episodes", "")
        try:
            indices = json.loads(p.read_text())
            if isinstance(indices, list) and indices:
                out[name] = f"{len(indices)} eps"
        except Exception:
            continue
    return out


def _load_split_episode_set(root: Path, split: str) -> Optional[set[int]]:
    """Load a ``{split}_episodes.json`` file as a set of episode indices.
    Returns None if the file doesn't exist."""
    p = root / f"{split}_episodes.json"
    if not p.exists():
        return None
    try:
        indices = json.loads(p.read_text())
        if isinstance(indices, list):
            return set(indices)
    except Exception:
        pass
    return None


def get_splits(repo_path: str) -> dict[str, str]:
    """Return available splits for a dataset.

    Checks info.json ``splits`` first, then falls back to detecting
    ``{split}_episodes.json`` files for non-contiguous splits.
    """
    root = Path(repo_path).expanduser().resolve()
    info = _load_info(root)
    splits = info.get("splits", {})
    ep_list_splits = _detect_episode_list_splits(root)
    if ep_list_splits:
        merged = dict(splits)
        merged.update(ep_list_splits)
        return merged
    return splits


def _resolve_camera_video(
    ep_meta: dict, camera: str, *, root: Path, video_template: str,
    chunks_size: int, fps: int, v3: bool,
) -> tuple[Path, int]:
    """Resolve (video_path, frame_offset) for one camera of one episode.

    Mirrors the per-camera path logic in ``discover_lerobot_episodes`` — each
    camera has its OWN chunk/file/from_timestamp on v3.0, so the frame_offset
    is camera-specific (top and wrist cameras can sit at different offsets in
    their respective concatenated shards)."""
    ep_idx = ep_meta["episode_index"]
    if v3:
        vid_chunk = int(ep_meta.get(f"videos/{camera}/chunk_index", ep_idx // chunks_size))
        vid_file = int(ep_meta.get(f"videos/{camera}/file_index", 0))
        from_ts = float(ep_meta.get(f"videos/{camera}/from_timestamp", 0.0))
        rel_video = video_template.format(
            video_key=camera, chunk_index=vid_chunk, file_index=vid_file,
        )
        frame_offset = round(from_ts * fps)
    else:
        chunk = ep_idx // chunks_size
        rel_video = video_template.format(
            episode_chunk=chunk, video_key=camera, episode_index=ep_idx,
        )
        frame_offset = 0
    return root / rel_video, frame_offset


def discover_lerobot_episodes(
    repo_path: str,
    camera_key: str = "top_camera-images-rgb",
    n: Optional[int] = None,
    require_video: bool = True,
    split: Optional[str] = None,
    cameras: Optional[list[str]] = None,
) -> list[Episode]:
    """Discover episodes in a LeRobot v2.1 or v3.0 dataset.

    Args:
        repo_path: Root directory of the LeRobot dataset.
        camera_key: Video feature key to train on (must exist in meta/info.json
            ``features`` with ``dtype == "video"``). When ``cameras`` is given,
            this is ignored in favor of ``cameras[0]`` as the primary camera.
        n: Optional cap on number of episodes.
        require_video: If True (default), skip episodes whose video file is
            missing. Set to False for precache-mode runs that only read
            cached features — this lets us mount meta/ + data/ from S3 and
            skip the multi-GB videos/ tree.
        split: Optional split name (e.g. ``"train"``, ``"val"``, ``"test"``).
            When set, only episodes whose index falls within the split range
            from ``info.json["splits"]`` are returned.
        cameras: Optional list of camera keys for multi-camera training. When
            given, each Episode's ``camera_videos`` is populated with EVERY
            camera's (video_path, frame_offset), and ``video_path`` /
            ``frame_offset`` point at the FIRST camera (back-compat). When None,
            falls back to ``[camera_key]`` (single-camera, identical to before).

    Returns:
        List of ``Episode`` objects, one per episode, with ``video_path``
        pointing at the (first) requested camera's MP4 and ``n_frames`` taken
        from episode metadata.
    """
    cameras = list(cameras) if cameras else [camera_key]
    primary_camera = cameras[0]
    root = Path(repo_path).expanduser().resolve()
    info = _load_info(root)

    split_indices: Optional[set[int]] = None
    split_lo, split_hi = None, None
    if split is not None:
        ep_list = _load_split_episode_set(root, split)
        if ep_list is not None:
            split_indices = ep_list
        else:
            splits = info.get("splits", {})
            if split not in splits:
                all_splits = list(splits.keys()) + list(_detect_episode_list_splits(root).keys())
                raise KeyError(
                    f"split {split!r} not found. "
                    f"Available splits: {all_splits}"
                )
            split_lo, split_hi = _parse_split_range(splits[split])

    features = info.get("features", {})
    for cam in cameras:
        if cam not in features:
            available = [k for k, v in features.items() if v.get("dtype") == "video"]
            raise KeyError(
                f"camera {cam!r} not found in {root}/meta/info.json. "
                f"Available video keys: {available}"
            )
        if features[cam].get("dtype") != "video":
            raise ValueError(
                f"feature {cam!r} is not a video (dtype={features[cam].get('dtype')!r})"
            )

    video_template: str = info["video_path"]
    chunks_size: int = info.get("chunks_size", 1000)
    fps: int = info.get("fps", 30)
    v3 = _is_v3(info)

    episodes: list[Episode] = []
    skipped_missing = 0
    skipped_empty = 0
    for ep_meta in _iter_episode_meta(root, info):
        ep_idx = ep_meta["episode_index"]
        if split_indices is not None and ep_idx not in split_indices:
            continue
        if split_lo is not None and not (split_lo <= ep_idx < split_hi):
            continue
        length = ep_meta.get("length", 0)
        if length <= 0:
            skipped_empty += 1
            continue

        # Resolve every requested camera's (video_path, frame_offset). Each
        # camera has its own chunk/file/from_timestamp on v3.0 — so per-camera
        # offsets, never the primary's, must drive that camera's decode + cache.
        camera_videos: dict[str, tuple[Path, int]] = {}
        for cam in cameras:
            vp, off = _resolve_camera_video(
                ep_meta, cam, root=root, video_template=video_template,
                chunks_size=chunks_size, fps=fps, v3=v3,
            )
            camera_videos[cam] = (vp, off)

        # Primary camera drives the legacy video_path / frame_offset fields.
        video_path, frame_offset = camera_videos[primary_camera]
        if require_video and any(
            not vp.exists() for vp, _ in camera_videos.values()
        ):
            skipped_missing += 1
            continue

        chunk_for_path = ep_idx // chunks_size
        synthetic_path = root / "data" / f"chunk-{chunk_for_path:03d}" / f"episode_{ep_idx:06d}"
        episodes.append(Episode(
            path=synthetic_path,
            video_path=video_path,
            n_frames=int(length),
            frame_offset=frame_offset,
            episode_index=int(ep_idx),
            # Single-camera (len==1) leaves this as a one-entry dict; downstream
            # caching/_ep_camera_video treats camera=None identically anyway.
            camera_videos=camera_videos,
            task=((ep_meta.get("tasks") or [None])[0] if isinstance(ep_meta.get("tasks"), list) else ep_meta.get("tasks")),
        ))

        if n is not None and len(episodes) >= n:
            break

    if skipped_missing:
        print(f"[lerobot] skipped {skipped_missing} episodes with missing video "
              f"({'/'.join(cameras)})")
    if skipped_empty:
        print(f"[lerobot] skipped {skipped_empty} episodes with zero length")

    return episodes

"""
Episode discovery and PyTorch Datasets for reward model training.

Provides two Dataset flavours:
  - PrecomputedFeatureDataset: backed by disk-cached backbone features (.npy)
  - OnlineVideoDataset: reads frames from MP4 on the fly, returns preprocessed
    image tensors so the backbone runs as part of the training forward pass.

Both delegate trajectory sampling and label generation to pluggable components.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import resize_frame
from .video_reader import read_frames
from .samplers import TrajectorySampler
from .labelers import LabelGenerator


@dataclass
class Episode:
    path: Path
    video_path: Path
    n_frames: int
    frame_offset: int = 0  # v3.0: starting frame within a concatenated video
    episode_index: Optional[int] = None  # v3.0/v2.1: per-dataset episode id; needed for proprio parquet filtering
    # Per-frame progress curve in [0, 1], shape (n_frames,). Attached at
    # discovery time by warp_rm/data/progress_curves.attach_progress_curves
    # under the chosen --progress-shape. None means the labeler should
    # use its legacy uniform-velocity formulation.
    progress_per_frame: Optional[np.ndarray] = None
    # Multi-camera support. When N>1 cameras are requested at discovery time,
    # this maps camera_key -> (video_path, frame_offset) for EVERY camera
    # (the first camera duplicates video_path/frame_offset above, kept for
    # back-compat). Each camera carries its OWN frame_offset because on v3.0
    # concatenated videos different cameras can have different from_timestamps.
    # None / empty means single-camera legacy behavior (use video_path).
    camera_videos: Optional[dict] = None
    task: Optional[str] = None  # language condition (from episodes.jsonl tasks[0])

    def read_frames(self, indices):
        if self.frame_offset:
            indices = [i + self.frame_offset for i in indices]
        return read_frames(self.video_path, indices)


def load_fused_features(meta: dict, fusion: str = "concat") -> np.ndarray:
    """Load + fuse the per-camera feature caches for one episode.

    `meta` is one entry of the ep_meta dict (keyed by str(ep.path)); it carries
    ``cache_paths: list[str]`` (one .npy per camera, all aligned to the same
    n_feat). Single-camera (len==1) returns the camera's array unchanged, so
    the N=1 path is byte-identical to the legacy ``np.load(meta["cache_path"])``.

    Fusion modes:
      - "concat": concatenate cameras along the feature axis → (n_feat, 768*N).
      - "tokens": stack cameras as a token axis → (n_feat, N, 768).
    """
    cache_paths = meta.get("cache_paths")
    if not cache_paths:
        # Legacy single-camera meta (no cache_paths) — fall back to cache_path.
        cache_paths = [meta["cache_path"]]
    if len(cache_paths) == 1:
        # Byte-identical to the legacy single-camera load.
        return np.load(cache_paths[0])
    arrs = [np.load(cp) for cp in cache_paths]
    # Guard against ragged per-camera n_feat (e.g. a camera's video had a
    # different decoded length): trim all cameras to the shortest n_feat so the
    # stack/concat is well-defined and stays aligned to the labels.
    n_feat = min(a.shape[0] for a in arrs)
    arrs = [a[:n_feat] for a in arrs]
    if fusion == "tokens":
        return np.stack(arrs, axis=1)        # (n_feat, N, 768)
    return np.concatenate(arrs, axis=1)      # (n_feat, 768*N)


def discover_episodes(data_dir: str, n: Optional[int] = None,
                      prefer_resized: int | None = 224) -> list[Episode]:
    """Discover all episodes under data_dir.

    If prefer_resized is set, uses pre-resized MP4s (e.g. top_camera-images-rgb_224.mp4)
    when available, falling back to the original.
    """
    episodes = []
    resized_count = 0
    for ep_dir in sorted(Path(data_dir).iterdir()):
        if not ep_dir.is_dir():
            continue
        # Prefer pre-resized video if available and valid
        video_path = None
        if prefer_resized is not None:
            resized_path = ep_dir / f"top_camera-images-rgb_{prefer_resized}.mp4"
            if resized_path.exists() and resized_path.stat().st_size > 1024:
                # Quick validity check: try to read frame count
                _cap = cv2.VideoCapture(str(resized_path))
                _nf = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                _cap.release()
                if _nf > 0:
                    video_path = resized_path
                    resized_count += 1
        if video_path is None:
            video_path = ep_dir / "top_camera-images-rgb.mp4"
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n_frames > 0:
            episodes.append(Episode(path=ep_dir, video_path=video_path, n_frames=n_frames))
    if resized_count > 0:
        print(f"Using pre-resized ({prefer_resized}px) videos for {resized_count}/{len(episodes)} episodes")
    if n is not None:
        episodes = episodes[:n]
    return episodes


def split_episodes(episodes: list[Episode], n_train: int, n_val: int,
                   seed: int = 42) -> tuple[list[Episode], list[Episode]]:
    """Deterministic train/val split."""
    rng = random.Random(seed)
    shuffled = episodes.copy()
    rng.shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val]


def split_episodes_clean_val(
    episodes: list[Episode],
    n_val: int,
    clean_val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Episode], list[Episode]]:
    """Split with validation drawn from the cleanest (shortest) episodes.

    Short episodes are empirically higher-SNR: fast tshirt-folds are smooth,
    consistent-tempo demonstrations with less hesitation / retries / noise.
    A val set held out from the fastest `clean_val_frac` of episodes gives
    cleaner ground-truth signal for composite metrics.

    Returns (train_episodes, val_episodes) where:
      - val: `n_val` episodes deterministically sampled from the shortest
        `clean_val_frac` of the input.
      - train: all remaining episodes (in the order they'd appear after
        the usual deterministic shuffle on `seed`).
    """
    if n_val <= 0:
        return list(episodes), []
    n_total = len(episodes)
    n_clean_pool = max(n_val, int(round(clean_val_frac * n_total)))
    if n_clean_pool >= n_total:
        raise ValueError(
            f"clean_val_frac={clean_val_frac} of {n_total} episodes "
            f"= {n_clean_pool} pool leaves no train data."
        )

    sorted_eps = sorted(episodes, key=lambda e: e.n_frames)
    clean_pool = sorted_eps[:n_clean_pool]

    # Deterministic val pick from the clean pool.
    rng_pick = random.Random(seed)
    pool_shuffled = clean_pool.copy()
    rng_pick.shuffle(pool_shuffled)
    val_episodes = pool_shuffled[:n_val]

    val_ids = {id(e) for e in val_episodes}
    remaining = [e for e in episodes if id(e) not in val_ids]

    # Shuffle train deterministically (same seed as split_episodes for parity).
    rng_tr = random.Random(seed)
    rng_tr.shuffle(remaining)

    print(f"[clean_val] pool=shortest {n_clean_pool}/{n_total} "
          f"(threshold n_frames<={clean_pool[-1].n_frames}) "
          f"-> val={len(val_episodes)} (mean n_frames="
          f"{sum(e.n_frames for e in val_episodes)/max(1,len(val_episodes)):.0f}) "
          f"train_pool={len(remaining)}")
    return remaining, val_episodes


def split_episodes_quality_val(
    episodes: list[Episode],
    quality_tsv_path: str,
    n_val: int,
    quality_val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Episode], list[Episode]]:
    """Split with validation drawn from the highest-Q episodes per a TSV.

    Drop-in replacement for `split_episodes_clean_val` when a per-episode
    quality index (from `scripts/eval/score_episodes.py`) is available.
    Uses the model's own notion of "cleanest" rather than the
    length-based heuristic. Val is held out from the top
    `quality_val_frac` of the input by Q; train is everything else,
    shuffled deterministically on `seed`.

    Rows in the TSV keyed by `Path(ep.path).name`. Episodes missing
    from the TSV are placed at the bottom of the Q ranking (never
    eligible for val), matching the conservative assumption that
    unknown-quality ≠ highest-quality.
    """
    if n_val <= 0:
        return list(episodes), []
    import csv
    q_by_name: dict[str, float] = {}
    with open(quality_tsv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                q_by_name[row["episode"]] = float(row["Q"])
            except (KeyError, ValueError):
                continue
    if not q_by_name:
        raise ValueError(f"quality_tsv has no parseable rows: {quality_tsv_path}")

    # Rank episodes by Q descending; missing → -inf so they never enter val pool.
    ranked = sorted(
        episodes,
        key=lambda e: (-q_by_name.get(Path(e.path).name, float("-inf"))),
    )
    n_total = len(ranked)
    n_pool = max(n_val, int(round(quality_val_frac * n_total)))
    if n_pool >= n_total:
        raise ValueError(
            f"quality_val_frac={quality_val_frac} of {n_total} episodes "
            f"= {n_pool} pool leaves no train data."
        )
    clean_pool = ranked[:n_pool]

    rng_pick = random.Random(seed)
    pool_shuffled = clean_pool.copy()
    rng_pick.shuffle(pool_shuffled)
    val_episodes = pool_shuffled[:n_val]

    val_ids = {id(e) for e in val_episodes}
    remaining = [e for e in episodes if id(e) not in val_ids]
    rng_tr = random.Random(seed)
    rng_tr.shuffle(remaining)

    threshold_q = q_by_name.get(Path(clean_pool[-1].path).name, float("nan"))
    val_q = [q_by_name.get(Path(e.path).name, float("nan")) for e in val_episodes]
    val_q_mean = sum(q for q in val_q if q == q) / max(1, sum(1 for q in val_q if q == q))
    print(f"[quality_val] pool=top-Q {n_pool}/{n_total} "
          f"(threshold Q>={threshold_q:.4f}) "
          f"-> val={len(val_episodes)} (mean Q={val_q_mean:.4f}) "
          f"train_pool={len(remaining)}  tsv={Path(quality_tsv_path).name}")
    return remaining, val_episodes


def build_demo_tokens(ep, ep_meta, demo_index, k, m, fusion="concat", rng=None, exclude_self=True,
                      vel=False, feature_stride=3, source_standard_stride=45, window_size=32):
    """Demo context tokens for k demos of the same task (other episodes), m frames uniformly spanning each demo.

    vel=False (legacy): (k*m, D+2) rows = [feature (D) | normalised time in demo | demo id]
    vel=True:           (k*m, 2D+3) rows = [feature (D) | feature - previous demo feature (D) |
                                            normalised time | dt_std | demo id]
      dt_std = source frames between consecutive demo frames / ((window_size-1)*source_standard_stride),
      i.e. the cumulative-label increment the query labeler would assign to that gap. Together with the
      frame diff this tells the model the demo's RELATIVE VELOCITY (appearance change per unit of label),
      which the legacy layout (normalised time only) never conveyed."""
    rng = rng or random
    task = getattr(ep, "task", None) or ""
    cands = [p for p in demo_index.get(task, []) if (not exclude_self or p != str(ep.path)) and p in ep_meta]
    if not cands:
        cands = [str(ep.path)]
    rows = []
    for di in range(k):
        p = rng.choice(cands)
        arr = load_fused_features(ep_meta[p], fusion)
        if arr.ndim == 3:            # tokens fusion -> use the primary camera for demos
            arr = arr[:, 0]
        idx = np.linspace(0, len(arr) - 1, m).round().astype(int)
        t = (idx / max(len(arr) - 1, 1)).astype(np.float32)
        f = arr[idx].astype(np.float32)
        if vel:
            d = np.zeros_like(f); d[1:] = f[1:] - f[:-1]
            gap_src = np.zeros(m, np.float32); gap_src[1:] = np.diff(idx) * feature_stride
            dt_std = gap_src / float((window_size - 1) * source_standard_stride)
            rows.append(np.concatenate([f, d, t[:, None], dt_std[:, None], np.full((m, 1), di, np.float32)], axis=1))
        else:
            rows.append(np.concatenate([f, t[:, None], np.full((m, 1), di, np.float32)], axis=1))
    return np.concatenate(rows, 0)


class PrecomputedFeatureDataset(Dataset):
    """
    Dataset backed by disk-cached backbone features.

    Delegates trajectory sampling and label generation to pluggable components.
    num_workers=0 recommended to avoid CUDA-context-fork segfault.
    """

    def __init__(
        self,
        episodes: list[Episode],
        ep_meta: dict,
        sampler: TrajectorySampler,
        labeler: LabelGenerator,
        eval_mode: bool = False,
        return_abs_labels: bool = False,
        return_completion: bool = False,
        return_meta: bool = False,
        feature_stride: int = 3,
        fusion: str = "concat",
        return_text: bool = False,
        aug_p: float = 0.0,
        return_demos: bool = False,
        demo_k: int = 2,
        demo_m: int = 12,
        demo_index: dict | None = None,
        demo_vel: bool = False,
    ):
        self.return_demos = return_demos; self.demo_k = int(demo_k); self.demo_m = int(demo_m)
        self.demo_vel = bool(demo_vel)
        self.demo_index = demo_index or {}   # task -> [str(ep.path)] with cached features
        self.return_text = return_text   # append ep_meta["text_emb"] as the LAST batch element
        self.aug_p = aug_p               # prob. of loading an augmented feature cache instead of the clean one
        self.episodes = episodes
        self.ep_meta = ep_meta
        self.sampler = sampler
        self.labeler = labeler
        self.eval_mode = eval_mode
        self.return_abs_labels = return_abs_labels
        self.return_completion = return_completion
        self.return_meta = return_meta
        self.feature_stride = feature_stride
        # Multi-camera fusion mode: 'concat' (early, (T,768*N)) or 'tokens'
        # (late, (T,N,768)). With one camera both reduce to the legacy
        # (T,768) load — see load_fused_features.
        self.fusion = fusion

    def __len__(self):
        return len(self.episodes) * 100

    def __getitem__(self, idx):
        ep = self.episodes[idx % len(self.episodes)]
        meta = self.ep_meta[str(ep.path)]
        n_feat = meta["n_feat"]

        # 1. Sample trajectory
        feat_indices = self.sampler.sample_indices(n_feat)

        # 2. Generate labels (piecewise-aware via ep.progress_per_frame)
        labels = self.labeler.generate(
            feat_indices, ep.progress_per_frame, ep.n_frames,
        )

        # 3. Load features from disk (fusing N cameras when multi-cam).
        augs = meta.get("aug_cache_paths")
        if augs and self.aug_p > 0 and not self.eval_mode and random.random() < self.aug_p:
            _c = random.choice(augs)
            feat_arr = (load_fused_features({"cache_paths": list(_c)}, self.fusion) if isinstance(_c, (list, tuple))
                        else np.load(_c))
        else:
            feat_arr = load_fused_features(meta, self.fusion)
        features = feat_arr[feat_indices].copy()

        out: list = [
            torch.from_numpy(features).float(),
            torch.from_numpy(labels).float(),
        ]

        if self.return_abs_labels:
            abs_labels = np.array(
                [fi * self.feature_stride / max(ep.n_frames - 1, 1) for fi in feat_indices],
                dtype=np.float32,
            )
            out.append(torch.from_numpy(abs_labels).float())

        if self.return_completion:
            comp = np.array(
                [1.0 if fi == n_feat - 1 else 0.0 for fi in feat_indices],
                dtype=np.float32,
            )
            out.append(torch.from_numpy(comp).float())

        if self.return_meta:
            # Integer ep index (0..N-1) into self.episodes + int32 feat_indices.
            # Used by QualityTracker to route predictions back to the
            # right accumulator slot without holding pathlib objects in tensors.
            out.append(torch.tensor(idx % len(self.episodes), dtype=torch.long))
            out.append(torch.tensor(feat_indices, dtype=torch.int32))

        if self.return_demos:
            out.append(torch.from_numpy(build_demo_tokens(ep, self.ep_meta, self.demo_index, self.demo_k, self.demo_m, self.fusion,
                                                          vel=self.demo_vel, feature_stride=self.feature_stride)))
        if self.return_text:
            out.append(torch.from_numpy(np.asarray(meta["text_emb"], dtype=np.float32)))
        return tuple(out)


class OnlineVideoDataset(Dataset):
    """
    Dataset that reads frames directly from MP4 and returns preprocessed image
    tensors.  The backbone forward pass happens in the training loop (on GPU),
    not in the dataset worker, so num_workers > 0 is safe.

    Each __getitem__ call:
      1. Picks an episode
      2. Computes n_feat = ceil(n_frames / feature_stride)
      3. Samples trajectory indices in *feature-space* via the sampler
      4. Converts to source frame indices (feat_idx * feature_stride)
      5. Decodes those frames from the MP4
      6. Resizes, normalises, returns (images [W,C,H,W], labels [W])
    """

    def __init__(
        self,
        episodes: list[Episode],
        sampler: TrajectorySampler,
        labeler: LabelGenerator,
        feature_stride: int = 3,
        image_size: int = 224,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        return_abs_labels: bool = False,
        crop_mode: str = "squash",
    ):
        self.episodes = episodes
        self.sampler = sampler
        self.labeler = labeler
        self.feature_stride = feature_stride
        self.image_size = image_size
        self.mean = mean if mean is not None else np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = std if std is not None else np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.crop_mode = crop_mode
        self.return_abs_labels = return_abs_labels

        # Pre-compute n_feat per episode so sampler can use it
        self._n_feat = [
            (ep.n_frames + feature_stride - 1) // feature_stride
            for ep in episodes
        ]

    def __len__(self):
        return len(self.episodes) * 100

    def __getitem__(self, idx):
        ep_idx = idx % len(self.episodes)
        ep = self.episodes[ep_idx]
        n_feat = self._n_feat[ep_idx]

        # 1. Sample trajectory in feature-space
        feat_indices = self.sampler.sample_indices(n_feat)

        # 2. Generate labels (piecewise-aware via ep.progress_per_frame)
        labels = self.labeler.generate(
            feat_indices, ep.progress_per_frame, ep.n_frames,
        )

        # 3. Convert to source frame indices and decode
        frame_indices = [min(fi * self.feature_stride, ep.n_frames - 1) for fi in feat_indices]
        frames = ep.read_frames(frame_indices)  # list of (H, W, 3) uint8

        # 4. Resize (skip if already correct size), normalise, to CHW tensor
        processed = []
        for frame in frames:
            frame = resize_frame(frame, self.image_size, self.crop_mode)
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - self.mean) / self.std
            processed.append(frame.transpose(2, 0, 1))  # HWC -> CHW

        images = torch.from_numpy(np.stack(processed)).float()  # (W, C, H, W_img)

        if self.return_abs_labels:
            abs_labels = np.array(
                [fi * self.feature_stride / max(ep.n_frames - 1, 1) for fi in feat_indices],
                dtype=np.float32,
            )
            return images, torch.from_numpy(labels).float(), torch.from_numpy(abs_labels).float()

        return images, torch.from_numpy(labels).float()


class CachedVideoLoader:
    """
    High-throughput video loader with a lookahead frame cache.

    Strategy:
      1. At init, a seeded RNG pre-generates the full epoch schedule:
         (episode_idx, sample_seed) for every sample.
      2. A background thread pool looks ahead K batches, replays each
         sample_seed to compute the exact (video_path, frame_idx) pairs,
         decodes them from MP4, and stores preprocessed frames in an
         in-memory dict — deduplicated across samples.
      3. The training loop assembles batches from the cache (instant
         numpy lookup, zero decode latency).
      4. As training advances past a batch, its frames are evicted.

    IID sampling is preserved because episode selection is uniformly random.
    Determinism is preserved because the epoch seed is fixed.

    Usage:
        loader = CachedVideoLoader(episodes, sampler, labeler, ...)
        for images, labels in loader:
            features = backbone(images.reshape(-1, C, H, W))
            ...
    """

    def __init__(
        self,
        episodes: list[Episode],
        sampler: TrajectorySampler,
        labeler: LabelGenerator,
        feature_stride: int = 3,
        image_size: int = 224,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        batch_size: int = 128,
        num_threads: int = 16,
        lookahead_batches: int = 4,
        epoch_seed: int | None = None,
        return_abs_labels: bool = False,
        crop_mode: str = "squash",
    ):
        self.episodes = episodes
        self.sampler = sampler
        self.labeler = labeler
        self.feature_stride = feature_stride
        self.image_size = image_size
        self.mean = mean if mean is not None else np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = std if std is not None else np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.crop_mode = crop_mode
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.lookahead_batches = lookahead_batches
        self.epoch_seed = epoch_seed
        self.return_abs_labels = return_abs_labels
        self._epoch = 0

        self._n_feat = [
            (ep.n_frames + feature_stride - 1) // feature_stride
            for ep in episodes
        ]
        self._n_batches = len(episodes) * 100 // batch_size

    def _replay_sample(self, ep_idx: int, sample_seed: int
                       ) -> tuple[list[int], list[int], np.ndarray]:
        """Replay a sample's RNG to get feat_indices, frame_indices, labels."""
        ep = self.episodes[ep_idx]
        n_feat = self._n_feat[ep_idx]

        # Seed Python's random module for deterministic sampler output
        rng_state = random.getstate()
        random.seed(sample_seed)
        feat_indices = self.sampler.sample_indices(n_feat)
        random.setstate(rng_state)

        labels = self.labeler.generate(
            feat_indices, ep.progress_per_frame, ep.n_frames,
        )
        frame_indices = [min(fi * self.feature_stride, ep.n_frames - 1)
                         for fi in feat_indices]
        return feat_indices, frame_indices, labels

    def _decode_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        """Decode and preprocess a single frame. Returns (C, H, W) float32."""
        frames = read_frames(video_path, [frame_idx])
        frame = frames[0]
        frame = resize_frame(frame, self.image_size, self.crop_mode)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - self.mean) / self.std
        return frame.transpose(2, 0, 1)  # CHW

    def __len__(self):
        return self._n_batches

    def __iter__(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        # --- 1. Generate full epoch schedule + pre-compute sample plans ---
        # Epoch seed varies per epoch for different sampling each pass
        base_seed = self.epoch_seed if self.epoch_seed is not None else random.randint(0, 2**31)
        epoch_seed = base_seed + self._epoch
        self._epoch += 1
        rng = np.random.RandomState(epoch_seed)

        n_samples = self._n_batches * self.batch_size

        # Pre-compute ALL sample plans at once (avoids repeated _replay_sample)
        # Each plan: (ep_idx, video_path_str, frame_indices, feat_indices, labels)
        plans: list[tuple[int, str, list[int], list[int], np.ndarray]] = []
        for _ in range(n_samples):
            ep_idx = int(rng.randint(0, len(self.episodes)))
            sample_seed = int(rng.randint(0, 2**31))

            ep = self.episodes[ep_idx]
            n_feat = self._n_feat[ep_idx]

            rng_state = random.getstate()
            random.seed(sample_seed)
            feat_indices = self.sampler.sample_indices(n_feat)
            random.setstate(rng_state)

            labels = self.labeler.generate(
                feat_indices, ep.progress_per_frame, ep.n_frames,
            )
            frame_indices = [min(fi * self.feature_stride, ep.n_frames - 1)
                             for fi in feat_indices]
            # v3.0 concat shards: indices are stored absolute (post-offset)
            # since the cache key is (vpath, frame_idx) — episodes sharing a
            # shard need distinct keys. Decoder reads abs indices directly.
            if ep.frame_offset:
                frame_indices = [i + ep.frame_offset for i in frame_indices]
            plans.append((ep_idx, str(ep.video_path), frame_indices, feat_indices, labels))

        # Pre-compute per-batch frame sets for fast lookup
        batch_frame_sets: list[set[tuple[str, int]]] = []
        for b in range(self._n_batches):
            fset: set[tuple[str, int]] = set()
            for s in range(self.batch_size):
                _, vpath, fi, _, _ = plans[b * self.batch_size + s]
                for f in fi:
                    fset.add((vpath, f))
            batch_frame_sets.append(fset)

        # --- 2. Frame cache ---
        cache: dict[tuple[str, int], np.ndarray] = {}
        cache_lock = threading.Lock()

        def _decode_video_frames(vpath: str, indices: list[int]):
            """Decode all needed frames from one video in a single read."""
            unique_indices = sorted(set(indices))
            raw_frames = read_frames(vpath, unique_indices)
            results = {}
            for fi, frame in zip(unique_indices, raw_frames):
                frame = resize_frame(frame, self.image_size, self.crop_mode)
                frame = frame.astype(np.float32) / 255.0
                frame = (frame - self.mean) / self.std
                results[(vpath, fi)] = frame.transpose(2, 0, 1)
            return results

        def _prefetch_batch_range(batch_start: int, batch_end: int):
            """Decode all unique frames in [batch_start, batch_end)."""
            needed: set[tuple[str, int]] = set()
            for b in range(batch_start, min(batch_end, self._n_batches)):
                needed |= batch_frame_sets[b]

            with cache_lock:
                to_decode = [k for k in needed if k not in cache]
            if not to_decode:
                return

            # Group by video for efficient batch decode
            by_video: dict[str, list[int]] = {}
            for vpath, fidx in to_decode:
                by_video.setdefault(vpath, []).append(fidx)

            with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                futures = [pool.submit(_decode_video_frames, vp, idxs)
                           for vp, idxs in by_video.items()]
                for future in as_completed(futures):
                    decoded = future.result()
                    with cache_lock:
                        cache.update(decoded)

        # --- 3. Initial prefetch ---
        prefetch_thread = threading.Thread(
            target=_prefetch_batch_range, args=(0, self.lookahead_batches), daemon=True
        )
        prefetch_thread.start()

        # --- 4. Yield batches ---
        for batch_idx in range(self._n_batches):
            prefetch_thread.join()

            # Assemble batch from cache
            images_list = []
            labels_list = []
            abs_labels_list = [] if self.return_abs_labels else None

            for s in range(self.batch_size):
                ep_idx, vpath, frame_indices, feat_indices, labels = plans[batch_idx * self.batch_size + s]
                sample_frames = []
                for fi in frame_indices:
                    key = (vpath, fi)
                    with cache_lock:
                        frame = cache.get(key)
                    if frame is not None:
                        sample_frames.append(frame)
                    else:
                        sample_frames.append(self._decode_frame(vpath, fi))
                images_list.append(np.stack(sample_frames))
                labels_list.append(labels)

                if self.return_abs_labels:
                    ep = self.episodes[ep_idx]
                    abs_labels_list.append(np.array(
                        [fi * self.feature_stride / max(ep.n_frames - 1, 1)
                         for fi in feat_indices], dtype=np.float32))

            images = torch.from_numpy(np.stack(images_list)).float()
            labels = torch.from_numpy(np.stack(labels_list)).float()

            # Prefetch next window
            next_end = batch_idx + 1 + self.lookahead_batches
            prefetch_thread = threading.Thread(
                target=_prefetch_batch_range, args=(batch_idx + 1, next_end), daemon=True
            )
            prefetch_thread.start()

            # Evict: remove frames only needed by past batches
            if batch_idx % self.lookahead_batches == 0 and batch_idx > 0:
                future_needed: set[tuple[str, int]] = set()
                for b in range(batch_idx, min(batch_idx + self.lookahead_batches + 1, self._n_batches)):
                    future_needed |= batch_frame_sets[b]
                with cache_lock:
                    for k in [k for k in cache if k not in future_needed]:
                        del cache[k]

            if self.return_abs_labels:
                abs_labels = torch.from_numpy(np.stack(abs_labels_list)).float()
                yield images, labels, abs_labels
            else:
                yield images, labels

        prefetch_thread.join()

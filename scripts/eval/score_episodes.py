#!/usr/bin/env python3
"""
Score every episode with WARP-RM and summarize its dense per-frame signal.

Runs sliding-window dense inference per episode and writes a per-episode TSV
with the mean per-frame progress velocity (pace) and the std of its first
difference (smoothness). The dense per-frame velocity is the model's actual
output; to inject it back into a LeRobot dataset use write_warp_rm_annotations.py.
Optionally lists the top/bottom-K episodes by mean velocity.

Usage:
    # LeRobot dataset
    uv run python scripts/eval/score_episodes.py \
        --lerobot-repo ~/.cache/huggingface/lerobot/<org>/<dataset> \
        --checkpoint checkpoints/best_model_main_full.pt \
        --output episode_quality.tsv \
        --top-k 10

    # Raw S3 source format
    uv run python scripts/eval/score_episodes.py \
        --data-dir /path/to/episodes \
        --checkpoint checkpoints/best_model.pt \
        --output episode_quality.tsv
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.config import default_feature_cache_dir

from warp_rm.data.dataset import discover_episodes, Episode
from warp_rm.data.lerobot_dataset import discover_lerobot_episodes
from warp_rm.models.backbones import build_backbone
from warp_rm.visualization.inference import (
    bulk_dense_inference,
    has_abs_progress_head,
)
from warp_rm.visualization.renderer import extract_features_on_the_fly
from scripts.eval.render import load_checkpoint, try_load_cached_features


# Per-episode summary of the dense per-frame signal. (Episode-wide quality
# scoring — the old Q/P/S index — is deprecated; we report the raw per-frame
# velocity summary only.)
TSV_COLUMNS = ["episode", "n_frames", "n_feat", "mean_velocity", "delta_v_std"]


@dataclasses.dataclass
class EpisodeSummary:
    n_feat: int
    mean_velocity: float
    delta_v_std: float


def episode_velocity_summary(vel: np.ndarray) -> EpisodeSummary:
    """Summarize an episode's dense per-frame velocity: mean (pace) and the
    std of its first difference (smoothness)."""
    vel = np.asarray(vel, dtype=np.float64)
    n = int(len(vel))
    mean_v = float(vel.mean()) if n else 0.0
    dstd = float(np.std(np.diff(vel))) if n > 1 else 0.0
    return EpisodeSummary(n_feat=n, mean_velocity=mean_v, delta_v_std=dstd)


def format_tsv_row(name: str, n_frames: int, s: EpisodeSummary) -> str:
    return f"{name}\t{n_frames}\t{s.n_feat}\t{s.mean_velocity:.4f}\t{s.delta_v_std:.4f}"


# ── WebUI quality cache interop ──────────────────────────────────────────────
# Format mirrors scripts/webui/server.py's _persist_quality_cache so the webUI
# picks results up on its next checkpoint load. Keep these in sync.


def _webui_ckpt_hash(ckpt_path: Path) -> str:
    stat = ckpt_path.stat()
    raw = f"{ckpt_path.resolve()}:{stat.st_mtime_ns}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _webui_cache_path(cache_dir: Path, ckpt_path: Path) -> Path:
    return cache_dir / f"quality_{_webui_ckpt_hash(ckpt_path)}.json"


def _json_safe(obj):
    """NaN/inf → None so the webUI JSON parse matches server behavior."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _load_webui_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
        scores = obj.get("scores", {})
        return scores if isinstance(scores, dict) else {}
    except Exception:
        return {}


def _write_webui_cache(path: Path, ckpt_path: Path, scores: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ckpt": str(ckpt_path),
        "mtime_ns": ckpt_path.stat().st_mtime_ns,
        "scores": scores,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


@dataclasses.dataclass
class Args:
    """Score every episode and write a per-episode velocity-summary TSV."""

    data_dir: Optional[str] = None
    """Raw episode directory. Mutually exclusive with --lerobot-repo."""

    lerobot_repo: Optional[str] = None
    """Path to a LeRobot v2.1 dataset. Mutually exclusive with --data-dir."""

    camera_key: str = "top_camera-images-rgb"
    """Camera key (LeRobot only)."""

    checkpoint: str = "checkpoints/best_model_main_full.pt"
    """Path to model checkpoint."""

    output: str = "episode_quality.tsv"
    """Where to write the TSV."""

    backbone: str = "dinov3"
    """Backbone model (must match the checkpoint)."""

    feature_stride: Optional[int] = None
    """Feature stride used at extraction time. When None (default),
    auto-resolved from the checkpoint's stamped `feature_stride` so an
    FS=2 ckpt scores against the FS=2 cache automatically. Pass an
    explicit value only to override; mismatching the ckpt's stamp is a
    hard error (would silently produce wrong quality scores)."""

    image_size: int = 224
    """Image size for backbone."""

    cache_dir: Optional[str] = None
    """Feature cache dir. Default: the shared feature cache (WARP_RM_FEATURE_CACHE / ~/.cache/warp_rm/features)."""

    window_size: int = 32
    """Inference window (frames per window). Defaults to 32 to match the
    paper WARP-RM model's training window (paper N=32). The model's
    aggregator caps at 32; smaller windows change the velocity magnitude."""

    n: Optional[int] = None
    """Score only the first N episodes (by discovery order). Omit for all."""

    top_k: int = 10
    """Print the top-K and bottom-K episodes by Q after scoring."""

    gpu: Optional[int] = None
    """GPU ID to use."""

    webui_cache: bool = True
    """Also populate the WARP-RM inspector's per-checkpoint quality cache (keyed by
    ckpt path + mtime). Merges with any existing cache so partial runs are
    additive."""

    webui_cache_dir: str = str(Path.home() / ".cache" / "warp_rm" / "webui_infer")
    """Directory for the webUI quality cache. Must match server's INFER_CACHE."""

    gpu_batch_eps: int = 16
    """Number of episodes packed into one bulk_dense_inference call. Larger
    values amortize Python overhead but hold more features + intermediate
    tensors in memory."""

    gpu_batch_size: int = 4096
    """GPU window batch size inside bulk_dense_inference. Increase on GPUs
    with more VRAM for higher utilization."""

    prefetch_depth: int = 32
    """Feature loads kept in flight at once by the prefetch thread pool. Lets
    feature-loading overlap with GPU work; tune up if disk is slow."""

    prefetch_workers: int = 4
    """Parallel threads used for feature loading."""

    require_video: bool = True
    """Drop episodes whose source MP4 is missing (default True, safe for
    local use). Set False on cloud jobs where only meta+features are
    mounted — dense inference only needs features anyway."""


def main():
    args = tyro.cli(Args)

    if (args.data_dir is None) == (args.lerobot_repo is None):
        raise SystemExit("Pass exactly one of --data-dir or --lerobot-repo.")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, ckpt = load_checkpoint(args.checkpoint, device)
    source_standard_stride = ckpt.get("standard_stride_src", 45)
    ckpt_feature_stride = ckpt.get("feature_stride")
    if args.feature_stride is None:
        if ckpt_feature_stride is None:
            raise SystemExit(
                "checkpoint has no `feature_stride` stamp; pass --feature-stride "
                "explicitly. Pre-2026-04-29 checkpoints predate the stamp and "
                "default to 3 — confirm before scoring."
            )
        feature_stride = int(ckpt_feature_stride)
        print(f"  feature_stride={feature_stride} (auto from checkpoint)")
    else:
        feature_stride = int(args.feature_stride)
        if ckpt_feature_stride is not None and int(ckpt_feature_stride) != feature_stride:
            raise SystemExit(
                f"--feature-stride {feature_stride} does not match the "
                f"checkpoint's stamped feature_stride {int(ckpt_feature_stride)}. "
                "Mismatched strides silently produce wrong scores (cache key "
                "diverges from the model's expected feature density). Drop "
                "the override or load the matching checkpoint."
            )
        print(f"  feature_stride={feature_stride} (from --feature-stride)")
    standard_feat_steps = source_standard_stride // feature_stride
    # Geometry must match training or features land in a different distribution
    # than the model saw. Pre-crop-mode checkpoints carry no stamp -> "squash",
    # which is what they were in fact trained with.
    crop_mode = ckpt.get("crop_mode", "squash")
    print(f"  crop_mode={crop_mode} "
          f"({'from checkpoint' if ckpt.get('crop_mode') else 'legacy default'})")
    enhanced = has_abs_progress_head(model)
    print(f"  standard_stride_src={source_standard_stride}  "
          f"standard_feat_steps={standard_feat_steps}  enhanced={enhanced}")

    # Discover episodes
    if args.lerobot_repo:
        print(f"\nDiscovering LeRobot episodes: {args.lerobot_repo} "
              f"(camera={args.camera_key})")
        episodes = discover_lerobot_episodes(args.lerobot_repo,
                                             require_video=args.require_video,
                                             camera_key=args.camera_key)
    else:
        print(f"\nDiscovering episodes: {args.data_dir}")
        episodes = discover_episodes(args.data_dir)

    if not episodes:
        print("No episodes found.")
        return

    if args.n is not None:
        episodes = episodes[:args.n]
    print(f"Scoring {len(episodes)} episodes.")

    # Lazy-build backbone only if some features are missing
    cache_dir = args.cache_dir or default_feature_cache_dir(args.backbone, feature_stride)
    encoder = None
    backbone_obj = None
    need_encoder = False
    for ep in episodes:
        if try_load_cached_features(ep, cache_dir, args.backbone,
                                    feature_stride, crop_mode) is None:
            need_encoder = True
            break
    if need_encoder:
        print(f"Building {args.backbone} encoder (some features uncached)...")
        backbone_obj, _ = build_backbone(args.backbone)
        backbone_obj = backbone_obj.to(device).eval()
        encoder = backbone_obj

    # Score each episode
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Seed webUI cache from any prior run so this invocation is additive.
    webui_scores: dict[str, dict] = {}
    webui_path: Optional[Path] = None
    if args.webui_cache:
        webui_path = _webui_cache_path(Path(args.webui_cache_dir), Path(args.checkpoint))
        webui_scores = _load_webui_cache(webui_path)
        if webui_scores:
            print(f"Loaded {len(webui_scores)} existing webUI cache entries "
                  f"from {webui_path}")

    cache_dir_path = Path(cache_dir)

    def _cache_feat_path(ep) -> Path:
        # Canonical key (namespaced + crop-mode aware). The old inline md5 here
        # reproduced only the legacy flat absolute-path key, so a center-crop
        # run would have written its features over the squash cache entry.
        from warp_rm.utils.caching import _ep_cache_path
        return _ep_cache_path(str(cache_dir_path), ep, args.backbone,
                              feature_stride, None, crop_mode)

    def _load_features_for(ep):
        """Feature-fetch for one episode; writes back to the disk cache on
        compute-miss so subsequent runs hit. Returns None if no encoder is
        available to compute a miss."""
        feat = try_load_cached_features(ep, cache_dir, args.backbone,
                                        feature_stride, crop_mode)
        if feat is not None:
            return feat
        if encoder is None:
            return None
        feat = extract_features_on_the_fly(
            encoder, ep, device, feature_stride, args.image_size,
            backbone_obj.MEAN, backbone_obj.STD, crop_mode=crop_mode,
        )
        try:
            cache_dir_path.mkdir(parents=True, exist_ok=True)
            np.save(_cache_feat_path(ep), feat)
        except Exception:
            pass
        return feat

    rows: list[tuple[str, int, object]] = []
    with open(out_path, "w") as f:
        f.write("\t".join(TSV_COLUMNS) + "\n")

        # Sliding-window prefetch: keeps up to `prefetch_depth` feature loads in
        # flight so disk/encoder work overlaps with GPU inference.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.prefetch_workers)
        in_flight: dict[int, concurrent.futures.Future] = {}
        next_submit = 0

        def _submit(idx: int) -> None:
            in_flight[idx] = pool.submit(_load_features_for, episodes[idx])

        for k in range(min(args.prefetch_depth, len(episodes))):
            _submit(k)
            next_submit = k + 1

        buffer_eps: list = []
        buffer_feats: list[np.ndarray] = []
        buffer_global_idx: list[int] = []
        i = 0

        def _flush_gpu_batch() -> None:
            if not buffer_feats:
                return
            text_list = None
            if getattr(model, "lang_dim", 0):
                from warp_rm.utils.text import embed_tasks
                emb = embed_tasks([getattr(e, "task", None) or "" for e in buffer_eps], device)
                text_list = [emb[getattr(e, "task", None) or ""] for e in buffer_eps]
            results = bulk_dense_inference(
                model, buffer_feats, device,
                window_size=args.window_size,
                standard_feat_steps=standard_feat_steps,
                gpu_batch_size=args.gpu_batch_size,
                want_abs=enhanced,
                text_list=text_list,
            )
            for gi, ep, (abs_prog, vel, abs_metrics) in zip(buffer_global_idx, buffer_eps, results):
                summary = episode_velocity_summary(vel)
                rows.append((ep.path.name, ep.n_frames, summary))

                row_str = format_tsv_row(ep.path.name, ep.n_frames, summary)
                f.write(row_str + "\n")
                f.flush()

                if args.webui_cache:
                    webui_scores[str(ep.path.resolve())] = _json_safe({
                        "mean_velocity": summary.mean_velocity,
                        "delta_v_std": summary.delta_v_std,
                        "n_feat": summary.n_feat,
                    })

                print(f"  [{gi + 1}/{len(episodes)}] {ep.path.name:<28} "
                      f"mean_vel={summary.mean_velocity:.3f}  "
                      f"n_frames={ep.n_frames}")
            buffer_eps.clear()
            buffer_feats.clear()
            buffer_global_idx.clear()

        try:
            while i < len(episodes):
                feat = in_flight.pop(i).result()
                if next_submit < len(episodes):
                    _submit(next_submit)
                    next_submit += 1

                if feat is None:
                    print(f"  [{i + 1}/{len(episodes)}] skip {episodes[i].path.name}: "
                          f"no cached features and no encoder")
                else:
                    buffer_eps.append(episodes[i])
                    buffer_feats.append(feat)
                    buffer_global_idx.append(i)
                    if len(buffer_feats) >= args.gpu_batch_eps:
                        _flush_gpu_batch()
                i += 1
            _flush_gpu_batch()
        finally:
            pool.shutdown(wait=True)

    print(f"\nWrote {len(rows)} scored episodes -> {out_path.resolve()}")

    if args.webui_cache and webui_path is not None:
        _write_webui_cache(webui_path, Path(args.checkpoint), webui_scores)
        print(f"Wrote webUI cache ({len(webui_scores)} total entries) -> {webui_path}")

    if rows:
        rows.sort(key=lambda r: r[2].mean_velocity, reverse=True)
        k = min(args.top_k, len(rows))
        print(f"\nTop {k} by mean velocity:")
        for name, nf, s in rows[:k]:
            print(f"  mean_vel={s.mean_velocity:.3f}  {name}  ({nf} frames)")
        print(f"\nBottom {k} by mean velocity:")
        for name, nf, s in rows[-k:][::-1]:
            print(f"  mean_vel={s.mean_velocity:.3f}  {name}  ({nf} frames)")


if __name__ == "__main__":
    main()

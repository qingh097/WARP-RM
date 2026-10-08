#!/usr/bin/env python3
"""
Render reward-overlay videos for episodes from any data directory.

Supports two modes:
  - Cached features: uses pre-computed .npy cache (fast, for training data)
  - On-the-fly: extracts backbone features live (for unseen data)

Usage:
    # Render 5 longest episodes from a new data directory (on-the-fly extraction)
    python scripts/render.py \
        --data-dir /path/to/episodes \
        --checkpoint checkpoints/best_model.pt \
        --n 5

    # Render from training data using existing feature cache
    python scripts/render.py \
        --data-dir /path/to/training/episodes \
        --checkpoint checkpoints/best_model.pt \
        --cache-dir /tmp/feat_rm3_dinov3_fs3 \
        --show-gt
"""

import dataclasses
import hashlib
import os
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from warp_rm.data.dataset import discover_episodes, Episode
from warp_rm.models.backbones import build_backbone
from warp_rm.models.aggregators import TransformerAggregator
from warp_rm.visualization.renderer import render_episode


def discover_episodes_by_length(data_dir: str) -> list[Episode]:
    """Return all episodes sorted longest-first."""
    episodes = discover_episodes(data_dir)
    episodes.sort(key=lambda e: e.n_frames, reverse=True)
    return episodes


def load_checkpoint(checkpoint_path: str, device: torch.device,
                    d_model: int = 768, n_heads: int = 8, n_layers: int = 12,
                    dropout: float = 0.15, max_seq_len: int = 32) -> tuple:
    """Load checkpoint and return (model, metadata dict).

    Auto-detects the model config (temporal diffs, attention mode, strides)
    from the checkpoint's stamped metadata and state_dict keys.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    d_model = ckpt.get("d_model", d_model)

    # Multi-camera: rebuild with the stamped fusion + camera count. Legacy
    # ckpts lack these keys → single-cam concat (n_cameras=1, backbone_dim=
    # d_model), identical to before. For concat the backbone_dim is
    # d_model*n_cameras; for tokens it stays d_model (one camera per token).
    fusion = ckpt.get("fusion", "concat")
    n_cameras = int(ckpt.get("n_cameras", 1))
    if fusion == "concat":
        backbone_dim = int(ckpt.get("backbone_dim", d_model * n_cameras))
    else:  # tokens
        backbone_dim = int(ckpt.get("backbone_dim", d_model))

    # Always TransformerAggregator now — the legacy CausalTransformerAggregator
    # baseline was dropped in the WARP-RM column rename. Pre-rename ckpts that
    # omit `rel_bin_centers` keys can't be loaded here; re-train if needed.
    state_keys = set(ckpt["model"].keys())
    # Temporal diffs double the input_proj's input dim relative to the per-token
    # backbone_dim (which already accounts for concat's *N). Compare against
    # backbone_dim, not d_model, so multi-cam concat ckpts detect correctly.
    has_temporal_diffs = any(k for k in state_keys
                            if "input_proj.weight" in k
                            and ckpt["model"][k].shape[-1] == backbone_dim * 2)
    # Attention mode stamped by train.py's extra_ckpt_meta. Absent on
    # legacy ckpts → default to 'causal' for back-compat.
    attention_mode = ckpt.get("attention", "causal")
    use_causal = attention_mode != "bidirectional"
    lang_dim = int(ckpt["model"]["lang_proj.weight"].shape[1]) if "lang_proj.weight" in state_keys else 0
    if lang_dim:
        print(f"  language-conditioned checkpoint (lang_dim={lang_dim})")
    # Demo-conditioned ckpts: rebuild the demo token path, otherwise strict=False
    # silently drops demo_proj/demo_time/demo_id/demo_type and demos are ignored.
    demo_dim = int(ckpt["model"]["demo_proj.weight"].shape[1]) if "demo_proj.weight" in state_keys else 0
    demo_k = int(ckpt["model"]["demo_id.weight"].shape[0]) if "demo_id.weight" in state_keys else 2
    demo_m = int(ckpt.get("demo_m", 12))
    if demo_dim:
        print(f"  demo-conditioned checkpoint (demo_dim={demo_dim}, K={demo_k}, M={demo_m})")
    model = TransformerAggregator(
        lang_dim=lang_dim,
        demo_dim=demo_dim, demo_k=demo_k, demo_m=demo_m,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        dropout=dropout, max_seq_len=max_seq_len,
        backbone_dim=backbone_dim,
        use_temporal_diffs=has_temporal_diffs,
        use_causal_attention=use_causal,
        fusion=fusion,
        n_cameras=n_cameras,
        wrist_pool=int(ckpt.get("wrist_pool", 1)),
    ).to(device)
    print(f"  Detected TransformerAggregator "
          f"(temporal_diffs={has_temporal_diffs}, attention={attention_mode}, "
          f"fusion={fusion}, n_cameras={n_cameras}, backbone_dim={backbone_dim})")

    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # The abs-progress head is built unconditionally by TransformerAggregator,
    # but RMLoss only trains it when the ablation enables it. On a `no_abs`
    # checkpoint the head therefore carries INIT weights and emits a ~flat 0.5
    # (the uniform bin prior) that is uncorrelated — often anti-correlated —
    # with real progress. A bare hasattr() check cannot tell the two apart, so
    # resolve it from the stamped ablation and mark the model. Consumers go
    # through `has_abs_progress_head`, which honours the mark.
    abs_trained = True
    ablation_name = ckpt.get("ablation")
    if ablation_name:
        from warp_rm.core.ablation import get_ablation_configs
        cfg = get_ablation_configs().get(ablation_name)
        if cfg is not None:
            abs_trained = bool(cfg.use_abs_progress)
        else:
            print(f"  WARNING: unknown ablation {ablation_name!r}; assuming the "
                  f"abs head is trained")
    model._warp_rm_abs_head_trained = abs_trained
    if not abs_trained:
        print(f"  abs head: UNTRAINED (ablation={ablation_name}) — suppressed "
              f"from inference outputs")
    print(f"  step={ckpt.get('step', '?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.6f}  "
          f"val_spearman={ckpt.get('val_spearman', float('nan')):.4f}")
    return model, ckpt


def try_load_cached_features(ep: Episode, cache_dir: str,
                             backbone: str, feature_stride: int,
                             crop_mode: str = "squash") -> np.ndarray | None:
    """Try to load cached features for an episode. Returns None if not found.

    Prefers the stable (mount-prefix-agnostic) hash used by training, falls
    back to the legacy absolute-path hash for old caches. Mirrors the
    lookup in src.utils.caching._ep_cache_path — keep in sync."""
    from warp_rm.utils.caching import _ep_cache_path
    cp = _ep_cache_path(cache_dir, ep, backbone, feature_stride, None, crop_mode)
    if cp.exists():
        return np.load(str(cp))
    return None


import tyro


@dataclasses.dataclass
class Args:
    """Render reward-overlay videos for episodes."""

    data_dir: str
    """Path to episode data."""

    checkpoint: str = "checkpoints/best_model.pt"
    """Path to model checkpoint."""

    n: int = 5
    """Number of episodes to render."""

    out_dir: str = "rendered"
    """Output directory for rendered videos."""

    target_h: int = 480
    """Target video height in pixels."""

    fps: float = 15.0
    """Output video FPS."""

    backbone: Literal["dinov3", "clip"] = "dinov3"
    """Backbone model to use."""

    feature_stride: Optional[int] = None
    """Feature stride. When None (default), auto-resolved from the
    checkpoint's stamped `feature_stride`. Pass an explicit value only
    to override; mismatching the ckpt's stamp is a hard error."""

    image_size: int = 224
    """Image size for backbone."""

    cache_dir: Optional[str] = None
    """Feature cache dir. If set, loads cached features when available."""

    show_gt: bool = False
    """Include ground truth row (linear 0->1)."""

    sort_by: Literal["length", "name"] = "length"
    task: Optional[str] = None
    """Task text for language-conditioned checkpoints (applied to every rendered episode)."""
    """How to sort/select episodes."""

    gpu: Optional[int] = None
    """GPU ID to use."""


def main():
    args = tyro.cli(Args)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, ckpt = load_checkpoint(args.checkpoint, device)
    text_t = None
    if getattr(model, "lang_dim", 0):
        from warp_rm.utils.text import embed_tasks
        _task = args.task or ""
        if not args.task:
            print("  WARNING: language-conditioned checkpoint but no --task given; using empty text")
        text_t = torch.from_numpy(embed_tasks([_task], device)[_task]).to(device).unsqueeze(0)
        print(f"  task text: {_task!r}")

    source_standard_stride = ckpt.get("standard_stride_src", 45)
    crop_mode = ckpt.get("crop_mode", "squash")
    ckpt_feature_stride = ckpt.get("feature_stride")
    if args.feature_stride is None:
        if ckpt_feature_stride is None:
            raise SystemExit(
                "checkpoint has no `feature_stride` stamp; pass --feature-stride "
                "explicitly (legacy ckpts predate the stamp)."
            )
        feature_stride = int(ckpt_feature_stride)
        print(f"  feature_stride={feature_stride} (auto from checkpoint)")
    else:
        feature_stride = int(args.feature_stride)
        if ckpt_feature_stride is not None and int(ckpt_feature_stride) != feature_stride:
            raise SystemExit(
                f"--feature-stride {feature_stride} does not match the "
                f"checkpoint's stamped feature_stride {int(ckpt_feature_stride)}. "
                "Drop the override or load the matching checkpoint."
            )
    standard_feat_steps = source_standard_stride // feature_stride

    # Discover episodes
    print(f"\nDiscovering episodes in {args.data_dir}...")
    if args.sort_by == "length":
        episodes = discover_episodes_by_length(args.data_dir)
    else:
        episodes = discover_episodes(args.data_dir)

    if not episodes:
        print("No episodes found. Check --data-dir.")
        return

    selected = episodes[:args.n]
    print(f"Found {len(episodes)} episodes, rendering top {len(selected)}:")
    for ep in selected:
        print(f"  {ep.n_frames:6d} frames  ({ep.n_frames / 30:.1f}s @ 30fps)  {ep.path.name}")

    # Build backbone if we might need on-the-fly extraction
    encoder = None
    backbone_obj = None
    need_encoder = args.cache_dir is None  # definitely need it if no cache
    if not need_encoder and args.cache_dir:
        # Check if all episodes are cached
        for ep in selected:
            if try_load_cached_features(ep, args.cache_dir, args.backbone, feature_stride) is None:
                need_encoder = True
                break

    if need_encoder:
        print(f"\nBuilding {args.backbone} encoder...")
        backbone_obj, _ = build_backbone(args.backbone)
        backbone_obj = backbone_obj.to(device)
        backbone_obj.eval()
        encoder = backbone_obj

    # Render
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, ep in enumerate(selected, 1):
        ep_stem = ep.path.name
        if ep_stem.endswith(".mp4"):
            ep_stem = ep_stem[:-4]
        out_path = str(out_dir / f"{ep_stem}.mp4")
        print(f"\n[{i}/{len(selected)}] {ep.path.name}")

        # Try loading cached features
        feat_arr = None
        if args.cache_dir:
            feat_arr = try_load_cached_features(
                ep, args.cache_dir, args.backbone, feature_stride,
            )
            if feat_arr is not None:
                print(f"  Loaded cached features: {feat_arr.shape}")

        try:
            render_episode(
                model=model,
                episode=ep,
                out_path=out_path,
                device=device,
                feat_arr=feat_arr,
                encoder=encoder,
                feature_stride=feature_stride,
                image_size=args.image_size,
                backbone_mean=backbone_obj.MEAN if backbone_obj else None,
                backbone_std=backbone_obj.STD if backbone_obj else None,
                window_size=32,
                standard_feat_steps=standard_feat_steps,
                target_h=args.target_h,
                out_fps=args.fps,
                show_gt=args.show_gt,
                crop_mode=crop_mode,
                text=text_t,
            )
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\nDone. Videos saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
WARP-RM training entrypoint.

Trains the Warp-Augmented Relative Progress reward model: a frozen DINOv3
backbone + bidirectional Transformer aggregator with a C51 categorical
progress-velocity head, supervised by the time-warp (AR-sampler) augmentation.

The defaults ARE the recipe — `python scripts/train.py --lerobot-repo <ds>`
trains the recommended config with no extra flags: ablation=no_abs (C51 +
temporal diffs + stochastic depth), shortest-frac=0.25, window=32, AR sampler,
bidirectional attention, feature-stride=3, 15k steps, clean-val.

Usage:
    # Recommended recipe (zero flags beyond the dataset):
    python scripts/train.py --lerobot-repo /path/to/lerobot/dataset

    # Paper sim checkpoint config (sss15; selected at step 14,400 of 20k):
    python scripts/train.py --ablation no_abs --feature-stride 1 \
        --source-standard-stride 15 --max-steps 20000 \
        --lerobot-repo /path/to/lerobot/dataset

    # List all ablation configs:
    python scripts/train.py --list-configs
"""

import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.config import *  # noqa: F401, F403
from warp_rm.data.dataset import (
    split_episodes, split_episodes_clean_val,
    PrecomputedFeatureDataset, CachedVideoLoader,
)
from warp_rm.data.lerobot_dataset import discover_lerobot_episodes
from warp_rm.data.samplers import (
    ARSampler, ContinuousWarpSampler, EvalSampler, TrueARSampler, derive_ar_budget,
)
from warp_rm.data.labelers import RelativeCumulativeLabeler, DeltaVelocityLabeler
from warp_rm.data.preprocess import CROP_MODES


def _length_stats(n_frames: list[int]) -> str:
    import statistics
    if not n_frames:
        return "(empty)"
    s = sorted(n_frames)
    def pct(p):
        k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
        return s[k]
    return (
        f"n={len(s)} min={s[0]} p25={pct(0.25)} median={pct(0.5)} "
        f"p75={pct(0.75)} max={s[-1]} mean={statistics.fmean(s):.1f}"
    )


def filter_shortest(episodes, frac: float):
    n_total = len(episodes)
    n_keep = max(1, int(round(frac * n_total)))
    sorted_eps = sorted(episodes, key=lambda e: e.n_frames)
    kept = sorted_eps[:n_keep]
    dropped = sorted_eps[n_keep:]
    threshold = kept[-1].n_frames if kept else None
    print(f"[filter_shortest] frac={frac} -> kept {len(kept)}/{n_total} "
          f"(threshold n_frames<={threshold})")
    print(f"  kept:    {_length_stats([e.n_frames for e in kept])}")
    print(f"  dropped: {_length_stats([e.n_frames for e in dropped])}")
    return kept


def filter_shortest_stratified(episodes, frac: float, counts: dict):
    """Per-object-count stratified shortest filter.

    Group episodes by their object count (counts[str(episode_index)]), then
    within each group keep the shortest max(1, round(frac*len(group))) by
    n_frames. Episodes whose index is missing from `counts` form a separate
    'ungrouped' stratum and are filtered the same way (so the total stays
    ≈ frac*N rather than silently keep-all/drop-all). Returns the union.
    """
    n_total = len(episodes)
    groups: dict = {}
    ungrouped = []
    for e in episodes:
        key = counts.get(str(e.episode_index))
        if key is None:
            ungrouped.append(e)
        else:
            groups.setdefault(key, []).append(e)
    if ungrouped:
        print(f"[filter_shortest_stratified] WARNING: {len(ungrouped)} episodes "
              f"missing from counts -> 'ungrouped' stratum")

    kept = []
    # Sort strata by count for deterministic, readable output; the ungrouped
    # bucket (if any) is reported last under count=<ungrouped>.
    def _sort_key(c):
        return (0, c) if isinstance(c, (int, float)) else (1, str(c))
    for c in sorted(groups, key=_sort_key):
        group = groups[c]
        n_keep = max(1, int(round(frac * len(group))))
        sorted_grp = sorted(group, key=lambda e: e.n_frames)
        kept.extend(sorted_grp[:n_keep])
        print(f"  count={c}: {len(group)} -> keep {n_keep}")
    if ungrouped:
        n_keep = max(1, int(round(frac * len(ungrouped))))
        sorted_grp = sorted(ungrouped, key=lambda e: e.n_frames)
        kept.extend(sorted_grp[:n_keep])
        print(f"  count=<ungrouped>: {len(ungrouped)} -> keep {n_keep}")

    print(f"[filter_shortest_stratified] frac={frac} -> kept {len(kept)}/{n_total}")
    return kept


from warp_rm.models.backbones import build_backbone
from warp_rm.models.aggregators import TransformerAggregator
from warp_rm.core.trainer import Trainer
from warp_rm.core.ablation import AblationConfig, get_ablation_configs
from warp_rm.core.wandb_logger import WandbLogger
from warp_rm.utils.caching import precompute_features
from warp_rm.utils.environment import seed_everything, get_branch_tag

import tyro


@dataclasses.dataclass
class Args:
    """Run ablation experiment."""

    ablation: str = "no_abs"
    """Ablation config name (use --list to see options).

    Default 'no_abs' = the recommended WARP-RM recipe (C51 + temporal diffs +
    stochastic depth, NO absolute-progress head). The abs head tended to
    overfit and add noise in downstream policy (fold) eval, so the recommended
    defaults to dropping it. The abs head remains available via
    --ablation full. The public paper sim checkpoint uses no_abs with
    feature-stride=1, source-standard-stride=15, and a 20k-step schedule
    (the selected checkpoint is at step 14,400). --list shows all configs."""

    mode: Literal["online", "precache"] = "precache"
    """Data loading mode (default: online)."""

    tag: Optional[str] = None
    """Checkpoint tag (default: git branch name)."""

    list_configs: bool = False
    """List all available ablation configs and exit."""

    lerobot_repo: list[str] = dataclasses.field(default_factory=list)
    """Path(s) to one or more LeRobot v2.1 dataset roots. Required; at
    least one must be given. Episodes from all repos are merged before
    filtering."""

    camera_key: str = "top_camera-images-rgb"
    """Video feature key inside the LeRobot dataset to train on.

    Single-camera convenience flag. When --cameras is left at its default,
    this becomes the sole camera (so existing single-cam configs/jobs are
    unaffected). Ignored when --cameras lists more than the default."""

    cameras: str = "top_camera-images-rgb"
    """Comma-separated list of camera feature keys for multi-camera training.

    Default is the single top camera → behavior is byte-identical to before.
    Multi-cam example:
      --cameras top_camera-images-rgb,left_camera-images-rgb,right_camera-images-rgb
    Each camera's DINOv3 features are cached separately and fused per --fusion."""

    fusion: str = "concat"
    """Multi-camera fusion mode (only matters when --cameras lists >1 camera).

    'concat' (default) = early fusion: per-camera features are concatenated on
    the feature axis → (T, 768*N); the aggregator's input_proj resizes to
    backbone_dim=768*N. Sequence length stays T.
    'tokens' = late/attention fusion: each (camera, timestep) is a transformer
    token (sequence length T*N) with a learned per-camera embedding; the N
    camera tokens at each timestep are mean-pooled before the heads, so outputs
    stay (B, T, *)."""

    shortest_frac: float = 0.25
    """Keep only the shortest N% of discovered episodes by n_frames.

    Default 0.25 = shortest-25% (current production). Shortest episodes are
    cleaner demos: smoother tempo, less hesitation, higher SNR. Legacy
    values 0.50 and 0.75 remain supported (earlier SOTA bases). 1.0 keeps all."""

    min_frames: Optional[int] = None
    """Drop episodes shorter than this many SOURCE frames, before every other
    filter. Use it to cut truncated/aborted recordings that `--shortest-frac`
    would otherwise preferentially KEEP (it selects the shortest episodes, so a
    junk 2-second recording survives while a good 45-second demo is dropped).

    A principled floor is one standard-stride window:
    `(window_size - 1) * source_standard_stride + 2 * feature_stride`
    — below that the sampler cannot lay down a standard-pace path and dense
    inference falls back to a rescaled shorter stride."""

    exclude_episode_index: tuple[int, ...] = ()
    """Episode indices to drop outright (e.g. demos contaminated by a human
    reaching into frame). Applied before every other filter. Indices are matched
    per `Episode.episode_index`, which is only unambiguous with a single
    --lerobot-repo; with several repos, prune upstream instead."""

    object_counts_json: Optional[str] = None
    """Path to an object_counts.json. When set, the shortest filter is applied
    per object-count stratum (group by count, keep shortest shortest_frac of
    each) instead of globally. When unset, behavior is unchanged."""

    require_cached: bool = False
    """Drop episodes whose DINOv3 feature cache is missing instead of
    extracting from video. Use on cloud jobs where videos aren't mounted."""

    clean_val: bool = True
    """Hold out the validation set from the shortest clean_val_frac of
    episodes. Short episodes are higher-SNR demos (smooth, consistent
    tempo, less hesitation); evaluating against them gives a cleaner
    composite signal. Default True (matches SOTA recipe). Set --no-clean-val
    for the legacy random-split evaluation — metrics are not comparable across
    clean_val=True/False runs."""

    clean_val_frac: float = 0.15
    """Fraction of the shortest episodes from which val is drawn (when
    --clean-val is set). 0.15 default per user hypothesis: top-15% fastest
    demos are the cleanest."""

    max_steps: Optional[int] = None
    """Override MAX_STEPS. Also retunes the cosine schedule T_0 to match."""

    batch_size: Optional[int] = None
    """Override BATCH_SIZE."""

    lr: Optional[float] = None
    """Override LR."""

    window_size: Optional[int] = 32
    """Override WINDOW_SIZE. Must be <= MAX_SEQ_LEN (32 by default).

    Default 32 matches the new winning recipe (AR + bidir + bs1024 +
    win32). Earlier recipes used 30 or 20; set explicitly to fall back."""

    source_standard_stride: Optional[int] = None
    """Override SOURCE_STANDARD_STRIDE. Lower = denser per-step sampling
    (smaller std_feat_steps = SSS // feature_stride), and finer label scale.
    Also shifts max|label| = max_src_stride / SSS; tune --max-src-stride
    alongside to keep labels in the C51 head's [-3, 3] range."""

    rel_bin_max: Optional[float] = None
    """Override the C51 relative-head support to [-rel_bin_max, rel_bin_max]
    (default [-3, 3]; delta-labeler mode defaults to [-6, 6]). Needed when
    labels exceed the support and clamp — e.g. sss15 on 60Hz data reaches
    |label| ~5 on the longest episodes (2.96% of tokens clamped)."""

    auto_bin_range: bool = False
    """Size the C51 relative-head support to the label distribution the sampler
    actually produces (p99.5 x 1.15, rounded up to 0.05) instead of a fixed
    +-3. OFF by default since 2026-08-19: as a default (shipped 2026-08-15,
    PR #2) it resized the support to +-1.45 at sss15, which parks the curation
    threshold (1.0) at 69% of the half-range with the whole upper tail packed
    against the edge bin. Measured downstream at n=512 (paired, tight-anypart):
    the auto-sized default's arm scored 3.904 vs 4.391 for the fixed +-3
    support with the same seed/recipe/eval (p=2e-08, jointly with the derived
    path budget below). Opt in per-run when the label distribution genuinely
    outgrows +-3 (e.g. sss15 on 60 Hz data); the saturation audit prints
    either way, and an explicit --rel-bin-max still wins."""

    feature_stride: Optional[int] = None
    """Override FEATURE_STRIDE (default 3 — every 3rd source frame produces
    one DINOv3 feature). Lower = denser features, bigger feature cache,
    slower extraction. The cache dir keys on the stride
    (`<root>/<backbone>_fs<stride>/`), and the value is stamped into the
    checkpoint so downstream scoring/inference paths pick it up
    automatically — never assume the global default at score time."""

    crop_mode: str = "squash"
    """Frame geometry when a decoded frame is not already `IMAGE_SIZE` square.

    'squash' (default) resizes straight to 224x224, ignoring aspect ratio —
    correct for datasets already converted to square videos (every LeRobot
    dataset produced by the conversion scripts, which center-crop at encode
    time), and a no-op there since the resize is skipped.

    'center' center-crops to the largest centered square first, matching the
    ffmpeg filter the conversion scripts apply. Use this for datasets stored at
    their native non-square aspect ratio (e.g. 1280x720), where 'squash' would
    compress the horizontal axis by 1.78x. The mode is mixed into the feature
    cache key and stamped into the checkpoint, so scoring/inference reproduce
    it automatically."""

    sampler: str = "ar"
    """Trajectory sampler. 'ar' (default — the WARP recipe) = ARSampler:
    AR(1) log-speed process + Poisson-sampled reversals + path budget.
    'iid' = ARSampler with the speed process swapped for an i.i.d. log-normal
    draw matching the AR(1) marginal (the paper's IID ablation — measures the
    value of the AR temporal correlation); reversal sampling and path budget
    unchanged. Only affects training-time window sampling; not inference."""

    ar_center_stride_sec: float = 1.5
    """(ar only) Centre of per-step stride distribution, in seconds. The
    sampler samples a path budget in [center - half_range, center + half_range]
    seconds-per-step, then an AR(1) speed process modulates individual gaps.

    Fixed 1.5 s default again since 2026-08-19. Pass -1 to DERIVE from
    --source-standard-stride (centre = SSS/fps, the 2026-08-15 PR #2
    behavior): SSS sets the label denominator and this sets how far a window
    travels, so the derivation keeps them in sync —

        label(window) = path_src / ((N-1) * SSS)
        => centred at 1.0  <=>  center_stride_sec = SSS / fps

    but as the DEFAULT the derivation was measured to hurt downstream: at
    sss15 it derives a [0.167, 0.833] s band, narrower on both sides than any
    band this recipe had ever run (the arm that motivated the ship used
    centre 0.5 with half_range LEFT AT 1.0 — band [0, 1.5] s, 37.2% retention;
    the derived band retains 29.3%). Its arm scored 3.904 vs 4.391 for these
    fixed defaults at n=512, same seed/recipe/eval (p=2e-08, jointly with
    auto_bin_range above)."""

    ar_half_range_sec: float = 1.0
    """(ar only) Half-width of the uniform path-budget band around center.
    Fixed 1.0 s default again since 2026-08-19 (band [0.5, 2.5] s — the
    paper's Uniform([1/3 L, 5/3 L]) at the 1.5 s centre). Pass -1 to DERIVE
    as 2/3 * centre (PR #2 behavior; see ar_center_stride_sec for why that
    is no longer the default)."""

    ar_alpha: float = 0.5
    """(ar only) AR(1) autocorrelation coefficient for log-speed process.
    Higher = smoother speed variation across a window (more momentum)."""

    ar_lambda_reversals: float = 1.0
    """(ar only) Poisson rate for in-window reversal changepoints. 0 = always
    monotone direction (before the full-flip); 1 = on average one reversal."""

    ar_p_full_flip: float = 0.5
    """(ar only) Probability of flipping the whole window (backward playback)."""

    attention: str = "bidirectional"
    """Transformer attention type. 'bidirectional' (default — the WARP recipe)
    = full attention across the window. Valid here because the model scores
    already-observed trajectories, not autoregressive generation. 'causal' =
    each token attends only to the past. Stamped into ckpt meta so downstream
    inference rebuilds the model with the same mode."""

    wandb: bool = False
    """Opt in to Weights & Biases logging of train/val curves (default off).
    Requires `uv sync --extra wandb` and being logged in (`wandb login`)."""


def _clip_text_embeds(clip, out):
    """transformers<5 returns a tensor; >=5 returns BaseModelOutputWithPooling."""
    import torch as _torch
    if _torch.is_tensor(out):
        return out
    te = getattr(out, "text_embeds", None)
    if te is not None:
        return te
    return clip.text_projection(out.pooler_output)


def _demo_dense_env():
    """WARP_DEMO_DENSE='n_seg,n_per,stride' -> dense demo runs inside sparse intervals (None = uniform only)."""
    v = os.environ.get("WARP_DEMO_DENSE", "")
    return tuple(int(x) for x in v.split(",")) if v else None


def build_model(ablation: AblationConfig, d_model: int, device: torch.device,
                first_frame_pe_only: bool = False,
                rel_bin_min: float = -3.0,
                rel_bin_max: float = 3.0,
                use_causal_attention: bool = False,
                fusion: str = "concat",
                n_cameras: int = 1) -> torch.nn.Module:
    """Build TransformerAggregator with the ablation flags applied.

    Single model class regardless of ablation. `use_c51=False` ablations are
    handled by RMLoss falling back to Curriculum Huber on the model's
    progress_preds head (the C51 bins are always present but unused when the
    loss is in fallback mode).

    Multi-camera:
      - fusion="concat": cameras concatenated on the feature axis upstream, so
        backbone_dim must be d_model * n_cameras (input_proj resizes itself).
        With n_cameras=1 this is byte-identical to single-camera.
      - fusion="tokens": per-camera tokens; backbone_dim stays d_model (each
        token is one camera's feature) and the aggregator builds T*N tokens.
    """
    backbone_dim = d_model * n_cameras if fusion == "concat" else d_model
    model = TransformerAggregator(
        lang_dim=(512 if os.environ.get("WARP_LANG_COND", "0") == "1" else 0),
        demo_dim=(768 if os.environ.get("WARP_DEMO_COND", "0") == "1" else 0),
        demo_k=int(os.environ.get("WARP_DEMO_K", "2")), demo_m=int(os.environ.get("WARP_DEMO_M", "12")),
        demo_vel=os.environ.get("WARP_DEMO_VEL", "0") == "1",
        demo_dense=_demo_dense_env(),
        d_model=d_model, n_heads=N_HEADS, n_layers=N_LAYERS,
        dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN,
        backbone_dim=backbone_dim,
        use_temporal_diffs=ablation.use_temporal_diffs,
        stochastic_depth_p=ablation.stochastic_depth_p if ablation.use_stochastic_depth else 0.0,
        first_frame_pe_only=first_frame_pe_only,
        rel_bin_min=rel_bin_min,
        rel_bin_max=rel_bin_max,
        use_causal_attention=use_causal_attention,
        fusion=fusion,
        n_cameras=n_cameras,
        wrist_pool=int(os.environ.get("WARP_WRIST_POOL", "1")),
    ).to(device)
    # The abs head is always constructed but only supervised when the ablation
    # enables it; mark it so `has_abs_progress_head` doesn't surface an
    # untrained head's ~constant 0.5 output as a real prediction.
    model._warp_rm_abs_head_trained = bool(ablation.use_abs_progress)
    return model


def run_experiment(ablation: AblationConfig, mode: str = "online",
                   tag: str | None = None,
                   lerobot_repos: list[str] | None = None,
                   max_frames_by_repo: list[str] | None = None,
                   camera_key: str = "top_camera-images-rgb",
                   cameras: list[str] | None = None,
                   fusion: str = "concat",
                   min_frames: int | None = None,
                   exclude_episode_index: tuple[int, ...] = (),
                   shortest_frac: float = 1.0,
                   object_counts_json: str | None = None,
                   max_steps_override: int | None = None,
                   batch_size_override: int | None = None,
                   lr_override: float | None = None,
                   init_from_ckpt: str | None = None,
                   p_edge_anchor: float = 0.10,
                   p_mid_flip: float = 0.20,
                   p_backward: float = 0.50,
                   window_size_override: Optional[int] = None,
                   source_standard_stride_override: Optional[int] = None,
                   feature_stride_override: Optional[int] = None,
                   rel_bin_max_override: Optional[float] = None,
                   auto_bin_range: bool = False,
                   progress_shape: str = "uniform",
                   balance_repos: bool = False,
                   weight_by_length: bool = False,
                   smooth_flip: bool = True,
                   max_src_stride: int | None = None,
                   wandb_project: str | None = None,
                   wandb_entity: str | None = None,
                   disable_wandb: bool = False,
                   require_cached: bool = False,
                   first_frame_pe_only: bool = False,
                   focal_gamma: float = 0.0,
                   focal_on: str = "none",
                   tri_state_weight: float = 0.0,
                   tri_state_eps: float = 0.02,
                   completion_weight: float = 0.0,
                   completion_focal_gamma: float = 2.0,
                   completion_pos_weight: float = 2.0,
                   smoothness_weight: float = 0.0,
                   c51_label_smoothing_sigma: float = 0.0,
                   clean_val: bool = False,
                   clean_val_frac: float = 0.15,
                   labeler_mode: str = "relative",
                   label_anchor_frames: int = 15,
                   sampler_type: str = "continuous",
                   crop_mode: str = "squash",
                   ar_center_stride_sec: float = 1.5,
                   ar_half_range_sec: float = 1.0,
                   ar_alpha: float = 0.5,
                   ar_lambda_reversals: float = 1.0,
                   ar_p_full_flip: float = 0.5,
                   truear_alpha: float = 0.5,
                   truear_sigma_inf: float = math.log(2),
                   truear_lambda_reversals: float = 1.0,
                   truear_p_full_flip: float = 0.5,
                   attention: str = "bidirectional",
                   upload_ckpts_to_s3: bool = True,
                   ckpt_s3_prefix: str | None = None):
    global_start = time.time()
    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize the camera list. Default falls back to [camera_key] so existing
    # single-camera callers (camera_key only) keep working unchanged. With one
    # camera + fusion=concat the whole pipeline is byte-identical to before.
    cameras = list(cameras) if cameras else [camera_key]
    n_cameras = len(cameras)
    if n_cameras > 1 and mode == "online":
        # Online mode encodes a single camera per frame (CachedVideoLoader +
        # the backbone in the train loop). Multi-cam fusion is wired only for
        # precache (per-camera .npy caches → load_fused_features). Fail loud
        # instead of dying with a cryptic matmul shape error at step 1.
        raise ValueError(
            "multi-camera (--cameras with N>1) requires --mode precache; "
            "online mode only supports a single camera"
        )
    if n_cameras > 1:
        print(f"[multi-cam] {n_cameras} cameras={cameras} fusion={fusion}")
    elif fusion != "concat":
        # tokens fusion with a single camera is harmless (T*1 == T tokens) but
        # almost certainly a mistake — warn rather than silently differ.
        print(f"[multi-cam] WARNING: fusion={fusion} with a single camera")

    max_steps = max_steps_override if max_steps_override is not None else MAX_STEPS
    batch_size = batch_size_override if batch_size_override is not None else BATCH_SIZE
    lr = lr_override if lr_override is not None else LR
    # Override-aware feature stride. EVERY FEATURE_STRIDE reference inside
    # run_ablation must use ``feat_stride`` instead — otherwise an override
    # leaks into the wrong place (cache key, model meta, downstream loaders).
    feat_stride = (
        feature_stride_override
        if feature_stride_override is not None
        else FEATURE_STRIDE
    )
    if feat_stride != FEATURE_STRIDE:
        print(f"[feature-stride] override active: {FEATURE_STRIDE} → {feat_stride}")

    print(f"\n{'='*60}")
    print(f"ABLATION: {ablation.name} | Mode: {mode}")
    print(f"  c51={ablation.use_c51} temporal_diffs={ablation.use_temporal_diffs} "
          f"stoch_depth={ablation.use_stochastic_depth} abs={ablation.use_abs_progress}")
    print(f"  batch_size={batch_size} lr={lr:.2e} max_steps={max_steps} "
          f"p_edge_anchor={p_edge_anchor}")
    print(f"{'='*60}\n")

    # Build backbone
    print(f"Building {BACKBONE} backbone...")
    backbone, d_model = build_backbone(BACKBONE)
    backbone = backbone.to(device)
    backbone.eval()

    print(f"Device: {device} | Backbone: {BACKBONE} | D_MODEL: {d_model}")
    branch_tag = tag if tag else get_branch_tag()
    ckpt_s3_prefix_effective = None
    if upload_ckpts_to_s3:
        ckpt_s3_prefix_effective = (
            ckpt_s3_prefix
            or f"s3://{warp_rm_s3_bucket()}/warp_rm/ckpts/{branch_tag}/"
        )
        os.environ["WARP_RM_CKPT_S3_PREFIX"] = ckpt_s3_prefix_effective
        print(f"[ckpt-upload] enabled: {ckpt_s3_prefix_effective}")
    else:
        os.environ.pop("WARP_RM_CKPT_S3_PREFIX", None)
        print("[ckpt-upload] disabled")

    # Discover and split episodes
    print("Discovering episodes...")
    if not lerobot_repos:
        raise ValueError("--lerobot-repo is required (at least one repo path)")
    all_episodes = []
    require_video = mode != "precache"
    # Parse --max-frames-by-repo specs into [(substring, max_frames), ...]
    repo_caps: list[tuple[str, int]] = []
    for spec in (max_frames_by_repo or []):
        if ":" not in spec:
            raise ValueError(
                f"--max-frames-by-repo: bad spec {spec!r}, expected "
                f"'<substring>:<max_frames>'"
            )
        sub, val = spec.rsplit(":", 1)
        repo_caps.append((sub, int(val)))
    for repo in lerobot_repos:
        print(f"LeRobot source: {repo} (cameras={cameras}) "
              f"require_video={require_video}")
        eps = discover_lerobot_episodes(
            repo, camera_key=camera_key, cameras=cameras,
            require_video=require_video,
        )
        n_disc = len(eps)
        # Apply any per-repo length cap whose substring matches this repo's path.
        for sub, max_n in repo_caps:
            if sub in repo:
                kept = [e for e in eps if e.n_frames <= max_n]
                print(f"  --max-frames-by-repo {sub!r}: kept {len(kept)}/{n_disc} "
                      f"(n_frames<={max_n})")
                eps = kept
                break
        print(f"  discovered {n_disc} episodes; after per-repo cap {len(eps)}")
        all_episodes.extend(eps)
    print(f"Found {len(all_episodes)} episodes")

    if exclude_episode_index:
        drop = set(int(i) for i in exclude_episode_index)
        before = len(all_episodes)
        kept_ep = [e for e in all_episodes if int(e.episode_index) not in drop]
        found = drop.intersection(int(e.episode_index) for e in all_episodes)
        missing = drop - found
        print(f"[exclude] dropped {before - len(kept_ep)}/{before} episodes "
              f"by index: {sorted(found)}")
        if missing:
            # Loud, not fatal: a typo'd index silently training on the episode
            # you meant to remove is the failure mode worth surfacing.
            print(f"[exclude] WARNING: indices not present in the dataset: "
                  f"{sorted(missing)}")
        all_episodes = kept_ep

    if min_frames is not None:
        before = len(all_episodes)
        kept_ep = [e for e in all_episodes if e.n_frames >= min_frames]
        dropped = sorted(e.n_frames for e in all_episodes if e.n_frames < min_frames)
        print(f"[min_frames] >= {min_frames} src frames -> kept {len(kept_ep)}/{before}")
        if dropped:
            print(f"  dropped lengths: {dropped}")
        all_episodes = kept_ep
        if not all_episodes:
            raise ValueError(f"min_frames={min_frames} removed every episode")

    if not 0.0 < shortest_frac <= 1.0:
        raise ValueError(f"shortest_frac must be in (0, 1], got {shortest_frac}")
    if crop_mode not in CROP_MODES:
        raise ValueError(f"--crop-mode must be one of {CROP_MODES}, got {crop_mode!r}")
    if object_counts_json is not None:
        with open(object_counts_json) as f:
            _oc = json.load(f)
        counts = _oc["counts"] if isinstance(_oc, dict) and "counts" in _oc else _oc
        print(f"[object_counts] loaded {len(counts)} counts from {object_counts_json}")
        all_episodes = filter_shortest_stratified(all_episodes, shortest_frac, counts)
    elif shortest_frac < 1.0:
        all_episodes = filter_shortest(all_episodes, shortest_frac)

    if require_cached:
        from warp_rm.utils.caching import _ep_cache_path
        cache_dir = default_feature_cache_dir(BACKBONE, feat_stride)
        before = len(all_episodes)
        # Multi-cam: an episode is "cached" only if ALL its cameras are cached.
        # Single-cam keeps camera=None so existing top-camera caches resolve.
        cam_args = cameras if n_cameras > 1 else [None]
        all_episodes = [
            ep for ep in all_episodes
            if all(
                _ep_cache_path(cache_dir, ep, BACKBONE, feat_stride, cam).exists()
                for cam in cam_args
            )
        ]
        print(f"[require_cached] filtered {before} -> {len(all_episodes)} with cached features")

    if clean_val:
        # Clean-val path: carve val out first so train/val can't overlap
        # regardless of downstream filtering. Val is the shortest
        # clean_val_frac of episodes — short demos are higher-SNR.
        train_pool, val_episodes = split_episodes_clean_val(
            all_episodes, N_VAL, clean_val_frac=clean_val_frac, seed=SEED,
        )
        split_source = f"shortest-{clean_val_frac:.0%}-holdout"
        train_episodes = train_pool
        print(f"[clean_val=True] val={split_source}, "
              f"train={len(train_episodes)} from {len(train_pool)} post-val pool")
    else:
        n_train = max(len(all_episodes) - N_VAL, 0)
        train_episodes, val_episodes = split_episodes(all_episodes, n_train, N_VAL, SEED)
        print(f"Train: {len(train_episodes)}, Val: {len(val_episodes)}")

    # Attach per-episode progress curves to BOTH train and val episodes,
    # under the chosen --progress-shape. Sidecar lookup happens here once;
    # downstream labeler calls just read ep.progress_per_frame. Episodes
    # without a sidecar entry silently fall back to uniform — important
    # for future merged-dataset runs where only some episodes are
    # annotated.
    from warp_rm.data.progress_curves import attach_progress_curves
    dataset_roots = [Path(repo).resolve() for repo in (lerobot_repos or [])]
    n_hit, n_miss = attach_progress_curves(
        train_episodes + val_episodes, progress_shape, dataset_roots,
    )
    if progress_shape == "uniform":
        print(f"[progress] shape=uniform — all {n_hit + n_miss} episodes use linear t/T")
    else:
        print(f"[progress] shape={progress_shape} — {n_hit} episodes attached "
              f"from sidecar, {n_miss} fell back to uniform")

    # Build modular components
    window_size = window_size_override if window_size_override is not None else WINDOW_SIZE
    sss = (
        source_standard_stride_override
        if source_standard_stride_override is not None
        else SOURCE_STANDARD_STRIDE
    )
    standard_feat_steps = sss // feat_stride
    stride_options = (
        [s for s in STRIDE_OPTIONS_SRC if s <= max_src_stride]
        if max_src_stride is not None else STRIDE_OPTIONS_SRC
    )
    max_label = max(stride_options) / sss
    # Couple the sampler's path budget to SSS (docs/recipe.md 4a).
    ar_center_stride_sec, ar_half_range_sec, _derived_centre, _derived_half = \
        derive_ar_budget(sss, ar_center_stride_sec, ar_half_range_sec)
    if sampler_type in ("ar", "iid"):
        iid_speed = (sampler_type == "iid")
        print(f"Sampler: {sampler_type} | window_size={window_size} | "
              f"center_stride_sec={ar_center_stride_sec} | "
              f"half_range_sec={ar_half_range_sec} | alpha={ar_alpha} | "
              f"lambda_reversals={ar_lambda_reversals} | "
              f"p_full_flip={ar_p_full_flip} | iid_speed={iid_speed}")
        sampler = ARSampler(
            window_size=window_size,
            feature_stride=feat_stride,
            alpha=ar_alpha,
            center_stride_sec=ar_center_stride_sec,
            half_range_sec=ar_half_range_sec,
            lambda_reversals=ar_lambda_reversals,
            p_full_flip=ar_p_full_flip,
            iid_speed=iid_speed,
        )
    elif sampler_type == "truear":
        print(f"Sampler: truear | window_size={window_size} | "
              f"standard_feat_steps={standard_feat_steps} | alpha={truear_alpha} | "
              f"sigma_inf={truear_sigma_inf:.4f} | "
              f"lambda_reversals={truear_lambda_reversals} | "
              f"p_full_flip={truear_p_full_flip}")
        sampler = TrueARSampler(
            window_size=window_size,
            feature_stride=feat_stride,
            source_standard_stride=sss,
            alpha=truear_alpha,
            sigma_inf=truear_sigma_inf,
            lambda_reversals=truear_lambda_reversals,
            p_full_flip=truear_p_full_flip,
        )
    else:
        print(f"Sampler: continuous | stride_options_src={stride_options} "
              f"| max|label|={max_label:.2f} | p_edge_anchor={p_edge_anchor}")
        sampler = ContinuousWarpSampler(
            window_size=window_size,
            feature_stride=feat_stride,
            stride_options_src=stride_options,
            standard_feat_steps=standard_feat_steps,
            p_backward=p_backward,
            p_mid_flip=p_mid_flip,
            p_edge_anchor=p_edge_anchor,
            smooth_flip=smooth_flip,
        )

    if hasattr(sampler, "path_min_feat"):
        # The path budget is a REQUEST in feature frames that the sampler clips
        # to the episode. A band sitting above the episode length truncates the
        # AR(1) speed process and biases labels low. Print it every run so that
        # can never again be found late.
        _lens = np.asarray(
            [((e.n_frames + feat_stride - 1) // feat_stride) for e in train_episodes],
            dtype=np.float64)
        _p50 = float(np.median(_lens)) if _lens.size else 0.0
        _p90 = float(np.quantile(_lens, 0.9)) if _lens.size else 0.0
        _lo, _hi = float(sampler.path_min_feat), float(sampler.path_max_feat)
        _over = 0.0 if _hi <= _lo else min(1.0, max(0.0, (_hi - max(_lo, _p50)) / (_hi - _lo)))
        _src = (f"derived from sss={sss}" if _derived_centre else "explicit")
        _srch = ("derived" if _derived_half else "explicit")
        print(f"[path-budget] centre={ar_center_stride_sec:.4f}s ({_src}) "
              f"half={ar_half_range_sec:.4f}s ({_srch}) "
              f"-> band [{_lo:.0f}, {_hi:.0f}] feat frames "
              f"| standard window = {standard_feat_steps * (window_size - 1)} "
              f"| train n_feat p50={_p50:.0f} p90={_p90:.0f} "
              f"| {100.0 * _over:.0f}% of the band exceeds p50 (clipped to the episode)")

    if labeler_mode == "delta":
        labeler = DeltaVelocityLabeler(
            feature_stride=feat_stride,
            label_anchor_frames=label_anchor_frames,
        )
        print(f"[labeler] delta mode, anchor={label_anchor_frames} src frames "
              f"(= {label_anchor_frames / 30:.2f} s @ 30 fps)")
    else:
        labeler = RelativeCumulativeLabeler(
            feature_stride=feat_stride,
            source_standard_stride=sss,
        )

    eval_sampler = EvalSampler(
        window_size=WINDOW_SIZE,
        feature_stride=feat_stride,
        stride_options_src=STRIDE_OPTIONS_SRC,
        standard_feat_steps=standard_feat_steps,
    )

    # ── Data loading ─────────────────────────────────────────────────────
    needs_abs = ablation.use_abs_progress
    needs_completion = completion_weight > 0
    ep_meta = None
    train_backbone = None

    if mode == "online":
        train_loader = CachedVideoLoader(
            train_episodes, sampler=sampler, labeler=labeler,
            feature_stride=feat_stride, image_size=IMAGE_SIZE,
            mean=backbone.MEAN, std=backbone.STD,
            batch_size=batch_size, num_threads=DECODE_THREADS,
            lookahead_batches=LOOKAHEAD_BATCHES,
            epoch_seed=SEED, return_abs_labels=needs_abs,
            crop_mode=crop_mode,
        )
        train_backbone = backbone
        print(f"Online mode: threads={DECODE_THREADS}, lookahead={LOOKAHEAD_BATCHES}")
    else:
        # Pre-compute features (pipelined CPU decode + GPU inference)
        # For large datasets, pre-run with multi-GPU sharding:
        #   python scripts/precompute_features.py --data-dir ... --multi-gpu 3,5,0
        cache_dir = default_feature_cache_dir(BACKBONE, feat_stride)
        episodes_for_precompute = list({
            str(ep.path): ep
            for ep in (train_episodes + val_episodes)
        }.values())
        # Single-camera passes cameras=None so the cache key uses camera=None
        # (byte-identical to legacy top-camera caches). Multi-cam caches each
        # camera separately and ep_meta carries cache_paths for fusion.
        precompute_cameras = cameras if n_cameras > 1 else None
        ep_meta = precompute_features(
            episodes_for_precompute, backbone, device,
            feat_stride, IMAGE_SIZE, cache_dir=cache_dir,
            backbone=BACKBONE, batch_size=PRECACHE_BATCH_SIZE,
            mean=backbone.MEAN, std=backbone.STD,
            decode_workers=PRECACHE_DECODE_WORKERS,
            mega_batch_episodes=PRECACHE_MEGA_BATCH,
            cameras=precompute_cameras,
            crop_mode=crop_mode,
        )
        # ---- visual augmentation: K extra feature caches per episode ----
        _n_aug = int(os.environ.get("WARP_AUG_VARIANTS", "0"))
        for _k in range(_n_aug):
            _m = precompute_features(
                episodes_for_precompute, backbone, device,
                feat_stride, IMAGE_SIZE, cache_dir=cache_dir,
                backbone=BACKBONE, batch_size=PRECACHE_BATCH_SIZE,
                mean=backbone.MEAN, std=backbone.STD,
                decode_workers=PRECACHE_DECODE_WORKERS,
                mega_batch_episodes=PRECACHE_MEGA_BATCH,
                cameras=precompute_cameras,
                crop_mode=crop_mode, aug_tag=str(_k),
            )
            for _p, _v in _m.items():
                # single-cam: a path; multi-cam: the per-camera list (fused at load time like the clean caches)
                ep_meta.setdefault(_p, {}).setdefault("aug_cache_paths", []).append(_v.get("cache_paths") or _v["cache_path"])
        if _n_aug:
            print(f"[aug] {_n_aug} augmented feature cache(s) per episode; sampled with p={os.environ.get('WARP_AUG_P', '0.5')}")
        # ---- language condition: frozen CLIP text embedding per task ----
        if os.environ.get("WARP_LANG_COND", "0") == "1":
            from transformers import CLIPModel, CLIPTokenizer
            _tm = os.environ.get("WARP_TEXT_MODEL", "openai/clip-vit-base-patch16")
            _tok = CLIPTokenizer.from_pretrained(_tm); _clip = CLIPModel.from_pretrained(_tm).to(device).eval()
            _tasks = sorted({(ep.task or "") for ep in episodes_for_precompute})
            _emb = {}
            with torch.no_grad():
                for _i in range(0, len(_tasks), 64):
                    _b = _tasks[_i:_i + 64]
                    _t = _tok(_b, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
                    _e = torch.nn.functional.normalize(_clip_text_embeds(_clip, _clip.get_text_features(**_t)).float(), dim=-1).cpu().numpy()
                    for _s, _v in zip(_b, _e): _emb[_s] = _v.astype(np.float32)
            for ep in episodes_for_precompute:
                ep_meta[str(ep.path)]["text_emb"] = _emb[ep.task or ""]
            print(f"[lang] {len(_tasks)} unique task strings embedded with {_tm} (dim {next(iter(_emb.values())).shape[0]})")
            del _clip; torch.cuda.empty_cache()
        _didx = {}
        if os.environ.get("WARP_DEMO_COND", "0") == "1":
            for ep in episodes_for_precompute:
                _didx.setdefault(ep.task or "", []).append(str(ep.path))
            print(f"[demo] index over {len(_didx)} tasks; K={os.environ.get('WARP_DEMO_K','2')} demos x M={os.environ.get('WARP_DEMO_M','12')} frames, dropout {os.environ.get('WARP_DEMO_DROPOUT','0.5')}")
        torch.cuda.empty_cache()
        precompute_elapsed = time.time() - global_start
        print(f"Precompute: {precompute_elapsed:.0f}s (not counted against budget)")

        train_dataset = PrecomputedFeatureDataset(
            train_episodes, ep_meta, sampler=sampler, labeler=labeler,
            return_abs_labels=needs_abs, return_completion=needs_completion,
            return_meta=False,
            feature_stride=feat_stride,
            fusion=fusion,
            return_text=os.environ.get("WARP_LANG_COND", "0") == "1",
            return_demos=os.environ.get("WARP_DEMO_COND", "0") == "1",
            demo_k=int(os.environ.get("WARP_DEMO_K", "2")), demo_m=int(os.environ.get("WARP_DEMO_M", "12")),
            demo_vel=os.environ.get("WARP_DEMO_VEL", "0") == "1",
            demo_dense=_demo_dense_env(),
            demo_index=_didx,
            aug_p=float(os.environ.get("WARP_AUG_P", "0.5")) if int(os.environ.get("WARP_AUG_VARIANTS", "0")) > 0 else 0.0,
        )

        need_weighted = (
            (balance_repos and lerobot_repos and len(lerobot_repos) > 1)
            or weight_by_length
        )
        if need_weighted:
            from torch.utils.data import WeightedRandomSampler
            # Start from uniform per-ep weights, then multiply in any active
            # reweight factors (repo balance, episode length) so they compose.
            ep_weight = [1.0] * len(train_episodes)
            if balance_repos and lerobot_repos and len(lerobot_repos) > 1:
                repo_counts: dict[str, int] = {}
                ep_repos: list[str] = []
                for ep in train_episodes:
                    ep_path_str = str(ep.path)
                    owner = "_other"
                    for repo in lerobot_repos:
                        repo_tail = repo.rstrip("/").split("/")[-1]
                        if repo_tail in ep_path_str:
                            owner = repo_tail
                            break
                    ep_repos.append(owner)
                    repo_counts[owner] = repo_counts.get(owner, 0) + 1
                n_repos = max(1, len(repo_counts))
                for i, r in enumerate(ep_repos):
                    ep_weight[i] *= 1.0 / (n_repos * repo_counts[r])
                print(f"[balance_repos] per-repo counts={repo_counts}")
            if weight_by_length:
                lengths = [ep.n_frames for ep in train_episodes]
                mean_L = sum(lengths) / max(1, len(lengths))
                # w = L_ref / L_ep clipped to [0.5, 2.0]: short eps → upweight.
                for i, L in enumerate(lengths):
                    lw = mean_L / max(1, L)
                    lw = min(2.0, max(0.5, lw))
                    ep_weight[i] *= lw
                print(
                    f"[weight_by_length] mean_L={mean_L:.0f} "
                    f"min_ep_w={min(ep_weight):.3f} max_ep_w={max(ep_weight):.3f}"
                )
            # Expand to match PrecomputedFeatureDataset's __len__ (n_eps * 100).
            expanded = ep_weight * 100
            ws = WeightedRandomSampler(
                expanded, num_samples=len(expanded), replacement=True,
            )
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=ws,
                num_workers=0, pin_memory=False,
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=False,
            )

    # Model
    # Delta-mode widens the C51 relative-head bin range. Per-step labels can
    # reach ±(max_src_stride / label_anchor_frames); with max_src_stride=90
    # and anchor=15 that's ±6. Use ±6 as the default delta-mode range.
    auto_bin_max = None
    if sampler is not None and labeler is not None:
        import random as _rnd
        _state = _rnd.getstate()
        _rnd.seed(0xB1A5)
        _probe = train_episodes[: min(64, len(train_episodes))]
        _labels = []
        for _ep in _probe:
            _nf = (_ep.n_frames + feat_stride - 1) // feat_stride
            for _ in range(32):
                _fi = sampler.sample_indices(_nf)
                _labels.append(np.abs(np.asarray(
                    labeler.generate(_fi, _ep.progress_per_frame, _ep.n_frames),
                    dtype=np.float64)))
        _rnd.setstate(_state)
        _a = np.concatenate(_labels)
        _p995 = float(np.quantile(_a, 0.995))
        _fit = float(np.ceil(_p995 * 1.15 * 20) / 20)  # +15% margin, round to 0.05
        print(f"[label-audit] |label|: p50={np.quantile(_a, 0.5):.3f} "
              f"p90={np.quantile(_a, 0.9):.3f} p99={np.quantile(_a, 0.99):.3f} "
              f"p99.5={_p995:.3f} max={_a.max():.3f}  (n={_a.size} over {len(_probe)} eps)")
        print(f"[label-audit] clamped by the +-3.0 support: "
              f"{100.0 * (_a > 3.0).mean():.2f}% of labels "
              f"(a support of +-{_fit:.2f} would fit)")
        if auto_bin_range:
            auto_bin_max = _fit

    if labeler_mode == "delta":
        rel_bin_min_use, rel_bin_max_use = -6.0, 6.0
        print(f"[model] delta mode: rel_bin range = [{rel_bin_min_use}, {rel_bin_max_use}]")
    else:
        rel_bin_min_use, rel_bin_max_use = -3.0, 3.0
    if auto_bin_max is not None:
        rel_bin_min_use, rel_bin_max_use = -auto_bin_max, auto_bin_max
        print(f"[label-audit] --auto-bin-range: support resized to "
              f"[{rel_bin_min_use:.2f}, {rel_bin_max_use:.2f}]")
    if rel_bin_max_override is not None:
        rel_bin_min_use, rel_bin_max_use = -rel_bin_max_override, rel_bin_max_override
        print(f"[model] rel_bin override: range = [{rel_bin_min_use}, {rel_bin_max_use}]")
    use_causal_attention = attention != "bidirectional"
    print(f"[model] attention={attention} (use_causal={use_causal_attention})")
    model = build_model(
        ablation, d_model, device,
        first_frame_pe_only=first_frame_pe_only,
        rel_bin_min=rel_bin_min_use,
        rel_bin_max=rel_bin_max_use,
        use_causal_attention=use_causal_attention,
        fusion=fusion,
        n_cameras=n_cameras,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_enc_params = sum(p.numel() for p in backbone.parameters())
    print(f"Trainable: {n_params / 1e6:.2f}M | Encoder (frozen): {n_enc_params / 1e6:.1f}M")

    # Warm-start from a prior checkpoint if requested. Loads model weights
    # only — fresh optimizer + scheduler state. Use case: train phase 1 with
    # one curation strategy, then continue with another from the prior weights
    # (e.g., shortest25 length-curated → rescan top_k quality-curated).
    if init_from_ckpt:
        ckpt_path = Path(init_from_ckpt)
        if not ckpt_path.exists():
            # On cloud workers init_from_ckpt may be a bare filename inside
            # checkpoints/ that --source-ckpt-tag pulled from S3 — resolve.
            alt = Path("checkpoints") / ckpt_path.name
            if alt.exists():
                ckpt_path = alt
            else:
                raise FileNotFoundError(f"--init-from-ckpt: {init_from_ckpt} not found")
        print(f"[init] loading model weights from {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[init]   missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[init]   unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
        prior_step = ckpt.get("step") if isinstance(ckpt, dict) else None
        print(f"[init]   warm-started from {ckpt_path.name} (prior step={prior_step})")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    cosine_t0 = max(int(3 * max_steps), 1000)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cosine_t0, T_mult=1, eta_min=lr * 0.01,
    )

    # Wandb (opt-in-by-default; no-op if unauthenticated)
    wandb_logger = WandbLogger(
        project=wandb_project,
        run_name=branch_tag,
        entity=wandb_entity,
        disabled=disable_wandb,
        config={
            "tag": branch_tag,
            "ablation": ablation.name,
            "use_c51": ablation.use_c51,
            "use_temporal_diffs": ablation.use_temporal_diffs,
            "use_stochastic_depth": ablation.use_stochastic_depth,
            "use_abs_progress": ablation.use_abs_progress,
            "sampler": sampler_type,
            "shortest_frac": shortest_frac,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "lr": lr,
            "window_size": window_size,
            "feature_stride": feat_stride,
            "max_src_stride": max_src_stride,
            "p_edge_anchor": p_edge_anchor,
        },
        tags=[ablation.name, "continuous", f"steps={max_steps}"],
    )

    # Train
    print("Starting training...")
    trainer = Trainer(
        lang_cond=(os.environ.get("WARP_LANG_COND", "0") == "1"),
        demo_cond=(os.environ.get("WARP_DEMO_COND", "0") == "1"),
        demo_dropout=float(os.environ.get("WARP_DEMO_DROPOUT", "0.5")),
        demo_index=_didx,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_episodes=val_episodes,
        ep_meta=ep_meta,
        labeler=labeler,
        device=device,
        eval_sampler=eval_sampler,
        ablation_config=ablation,
        base_lr=lr,
        warmup_steps=WARMUP_STEPS,
        max_steps=max_steps,
        total_budget=TOTAL_BUDGET,
        eval_every=EVAL_EVERY,
        grad_clip=GRAD_CLIP,
        branch_tag=branch_tag,
        window_size=window_size,
        standard_feat_steps=standard_feat_steps,
        backbone=train_backbone,
        feature_stride=feat_stride,
        image_size=IMAGE_SIZE,
        wandb_logger=wandb_logger,
        focal_gamma=focal_gamma,
        focal_on=focal_on,
        tri_state_weight=tri_state_weight,
        tri_state_eps=tri_state_eps,
        completion_weight=completion_weight,
        completion_focal_gamma=completion_focal_gamma,
        completion_pos_weight=completion_pos_weight,
        smoothness_weight=smoothness_weight,
        c51_label_smoothing_sigma=c51_label_smoothing_sigma,
        train_sampler=sampler,
        label_mode=labeler_mode,
        label_anchor_frames=label_anchor_frames,
    )

    # Stamp delta-mode metadata so downstream loaders (webUI, scoring,
    # pair_fill) can route to the right dense-inference aggregator. Absent on
    # legacy ckpts → ``label_mode`` defaults to 'relative' in consumers.
    trainer.extra_ckpt_meta = {
        "label_mode": labeler_mode,
        "label_anchor_frames": label_anchor_frames if labeler_mode == "delta" else None,
        "rel_bin_min": rel_bin_min_use,
        "rel_bin_max": rel_bin_max_use,
        "standard_stride_src": sss,   # preserved for back-compat
        "crop_mode": crop_mode,
        "use_abs_progress": ablation.use_abs_progress,
        # AR-sampler stride calibration. Absent on pre-2026-08 checkpoints, which
        # leaves their supervision scale unverifiable after the fact: this must
        # track sss/fps or every label rescales by 45/sss (docs/recipe.md §4a).
        "ar_center_stride_sec": ar_center_stride_sec,
        "ar_half_range_sec": ar_half_range_sec,
        "min_frames": min_frames,
        "excluded_episode_index": list(exclude_episode_index),
        "feature_stride": feat_stride,
        "window_size": window_size,
        "sampler": sampler_type,
        "attention": attention,  # 'causal' | 'bidirectional'
        "progress_shape": progress_shape,
        # Multi-camera: downstream scoring/inference rebuilds the model with the
        # same fusion + camera count and fuses features the same way. Absent on
        # legacy ckpts → consumers default to single-cam concat.
        "cameras": cameras,
        "fusion": fusion,
        "n_cameras": n_cameras,
        "wrist_pool": int(os.environ.get("WARP_WRIST_POOL", "1")),
        "demo_k": int(os.environ.get("WARP_DEMO_K", "2")), "demo_m": int(os.environ.get("WARP_DEMO_M", "12")),
        "demo_vel": os.environ.get("WARP_DEMO_VEL", "0") == "1",
        "demo_dense": _demo_dense_env(),
        "demo_xor": os.environ.get("WARP_DEMO_XOR", "0") == "1",
        "backbone_dim": model.backbone_dim,
    }

    # Structured per-run JSONL logger (logs/runs/<date>/<tag>.jsonl).
    try:
        from warp_rm.core.run_logger import RunLogger
        trainer.run_logger = RunLogger(
            tag=branch_tag,
            config={
                "ablation": ablation.name,
                "use_c51": ablation.use_c51,
                "use_temporal_diffs": ablation.use_temporal_diffs,
                "use_stochastic_depth": ablation.use_stochastic_depth,
                "use_abs_progress": ablation.use_abs_progress,
                "sampler": sampler_type,
                "shortest_frac": shortest_frac,
                "max_steps": max_steps,
                "batch_size": batch_size,
                "lr": lr,
                "window_size": window_size,
                "feature_stride": feat_stride,
                "max_src_stride": max_src_stride,
                "p_edge_anchor": p_edge_anchor,
            },
        )
    except ImportError:
        pass

    _ = (clean_val, clean_val_frac)  # already consumed above

    best_metrics = trainer.train()

    # Final report
    total_time = time.time() - global_start
    n_params_all = n_params + n_enc_params
    print("\n" + "=" * 60)
    print(f"ABLATION: {ablation.name}")
    for key, val in best_metrics.to_dict().items():
        print(f"{key:30s} {val:.6f}")
    print(f"{'composite':30s} {best_metrics.composite_score:.6f}")
    print(f"training_seconds: {total_time:.1f}")
    print(f"num_params_M:     {n_params_all / 1e6:.1f}")
    print("=" * 60)

    return best_metrics


def main():
    args = tyro.cli(Args)

    configs = get_ablation_configs()

    if args.list_configs:
        print("Available ablation configs:")
        for name, cfg in configs.items():
            flags = []
            if cfg.use_c51: flags.append("c51")
            if cfg.use_temporal_diffs: flags.append("temporal_diffs")
            if cfg.use_stochastic_depth: flags.append("stoch_depth")
            if cfg.use_abs_progress: flags.append("abs")
            print(f"  {name:20s} -> {', '.join(flags) if flags else '(none — plain baseline)'}")
        return

    if args.ablation not in configs:
        print(f"Unknown ablation '{args.ablation}'. Available: {list(configs.keys())}")
        sys.exit(1)

    # Parse --cameras comma list. If left at the default (single top camera),
    # fall back to --camera-key so existing single-cam invocations are
    # unaffected even if they only set --camera-key.
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if cameras == [Args.cameras] and args.camera_key != Args.camera_key:
        # User overrode --camera-key but not --cameras → honor --camera-key.
        cameras = [args.camera_key]

    run_experiment(
        configs[args.ablation],
        mode=args.mode,
        tag=args.tag,
        lerobot_repos=args.lerobot_repo,
        camera_key=args.camera_key,
        cameras=cameras,
        fusion=args.fusion,
        min_frames=args.min_frames,
        exclude_episode_index=args.exclude_episode_index,
        shortest_frac=args.shortest_frac,
        object_counts_json=args.object_counts_json,
        require_cached=args.require_cached,
        clean_val=args.clean_val,
        clean_val_frac=args.clean_val_frac,
        max_steps_override=args.max_steps,
        batch_size_override=args.batch_size,
        lr_override=args.lr,
        window_size_override=args.window_size,
        source_standard_stride_override=args.source_standard_stride,
        feature_stride_override=args.feature_stride,
        rel_bin_max_override=args.rel_bin_max,
        auto_bin_range=args.auto_bin_range,
        sampler_type=args.sampler,
        crop_mode=args.crop_mode,
        ar_center_stride_sec=args.ar_center_stride_sec,
        ar_half_range_sec=args.ar_half_range_sec,
        ar_alpha=args.ar_alpha,
        ar_lambda_reversals=args.ar_lambda_reversals,
        ar_p_full_flip=args.ar_p_full_flip,
        attention=args.attention,
        # Weights & Biases logging: opt-in (default off).
        disable_wandb=not args.wandb,
        wandb_project="warp-rm",
        # Checkpoint S3 upload is off by default. Cloud launchers can
        # re-enable it by setting WARP_RM_CKPT_S3_PREFIX in the environment.
        upload_ckpts_to_s3=False,
    )


if __name__ == "__main__":
    main()

"""
Dense inference with velocity integration and absolute progress head support.

Reconstructs absolute progress [0, 1] from the model's relative cumulative
predictions by:
  1. Sliding standard-stride windows across the episode (max density)
  2. Extracting per-step velocities, averaging across overlapping windows
  3. Smoothing velocity, integrating via cumsum
  4. Min-max normalizing to [0, 1]

For enhanced models with abs_progress_head, also provides direct absolute
progress predictions and C51 uncertainty metrics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d


@torch.no_grad()
def dense_inference_relative(
    model: nn.Module,
    feat_arr: np.ndarray,
    device: torch.device,
    window_size: int = 20,
    standard_feat_steps: int = 15,
    batch_size: int = 512,
    text: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct absolute progress from relative model predictions.

    Args:
        model: Temporal aggregator (takes (B, T, D) -> (B, T))
        feat_arr: (n_feat, D) pre-computed backbone features
        device: Torch device
        window_size: Number of frames per window
        standard_feat_steps: Reference stride between frames in each window.
            If the episode is shorter than a full standard-stride window, the
            stride is automatically reduced so at least one window fits, and
            per-step velocities are rescaled by standard_feat_steps/stride_used
            so magnitudes stay comparable to standard-stride episodes.
        batch_size: Inference batch size

    Returns:
        abs_progress: (n_feat,) float32 in [0, 1] (velocity-integrated, normalized — no synthetic Gaussian smoothing)
        avg_velocity: (n_feat,) float32 per-step velocity (positive = forward, negative = rewind)
    """
    model.eval()
    n_feat = len(feat_arr)
    N = window_size
    std_step = standard_feat_steps

    max_start = n_feat - (N - 1) * std_step - 1
    if max_start <= 0 and n_feat >= N + 1:
        # Episode shorter than a full standard-stride window. Shrink the stride
        # so at least one window fits and the model produces real predictions
        # instead of a uniform-velocity stub. Velocities are rescaled below so
        # magnitudes stay comparable to standard-stride episodes.
        std_step = max(1, (n_feat - 2) // (N - 1))
        max_start = n_feat - (N - 1) * std_step - 1
        print(f"  [dense_inference] short episode ({n_feat} feat frames): "
              f"stride {standard_feat_steps} -> {std_step}, {max_start + 1} windows")

    if max_start <= 0:
        return (np.linspace(0, 1, n_feat, dtype=np.float32),
                np.ones(n_feat, dtype=np.float32) / n_feat)

    vel_scale = float(standard_feat_steps) / float(std_step)

    # Collect all window start positions (scan every feature step for max density)
    starts = list(range(0, max_start + 1))
    all_fi = [[min(s + i * std_step, n_feat - 1) for i in range(N)] for s in starts]
    all_preds: list[np.ndarray] = []

    for bi in range(0, len(all_fi), batch_size):
        batch_fi = all_fi[bi:bi + batch_size]
        batch_feat = np.stack([feat_arr[fi] for fi in batch_fi])
        feats_t = torch.from_numpy(batch_feat).float().to(device)
        out = model(feats_t, text=(text.expand(feats_t.shape[0], -1) if text is not None else None))
        preds = (out[0] if isinstance(out, tuple) else out).cpu().numpy()
        all_preds.append(preds)

    all_preds_arr = np.concatenate(all_preds, axis=0)

    # Aggregate per-step velocities across overlapping windows
    vel_sum = np.zeros(n_feat, dtype=np.float64)
    vel_count = np.zeros(n_feat, dtype=np.int64)

    for fi, preds in zip(all_fi, all_preds_arr):
        for i in range(N - 1):
            f_start = fi[i]
            f_end = fi[i + 1]
            span = f_end - f_start
            if span <= 0:
                continue
            vel = (preds[i + 1] - preds[i]) * (N - 1) * vel_scale
            end_c = min(f_end, n_feat)
            vel_sum[f_start:end_c] += vel
            vel_count[f_start:end_c] += 1

    # Average velocities; interpolate any gaps
    valid = vel_count > 0
    avg_vel = np.zeros(n_feat, dtype=np.float64)
    avg_vel[valid] = vel_sum[valid] / vel_count[valid]
    if valid.any() and not valid.all():
        xi = np.arange(n_feat)
        avg_vel = np.interp(xi, xi[valid], avg_vel[valid])

    # Cumulative integral -> absolute progress (no synthetic Gaussian smoothing).
    # Removed prior pre-/post-integration gaussian_filter1d calls so the curve
    # reflects the model's raw per-window aggregated velocity, matching how
    # the abs-head overlay is rendered (also no smoothing).
    abs_progress = np.cumsum(avg_vel)

    # Min-max normalize
    p_min, p_max = abs_progress.min(), abs_progress.max()
    if p_max > p_min:
        abs_progress = (abs_progress - p_min) / (p_max - p_min)
    else:
        abs_progress = np.linspace(0, 1, n_feat)
    abs_progress = np.clip(abs_progress, 0.0, 1.0)

    return abs_progress.astype(np.float32), avg_vel.astype(np.float32)


def has_abs_progress_head(model: nn.Module) -> bool:
    """True only if the model has a C51 absolute-progress head *that was trained*.

    `TransformerAggregator` always constructs `abs_progress_head`, but `RMLoss`
    only supervises it when the ablation sets `use_abs_progress` (i.e. NOT for
    `no_abs`, `c51_only`, `baseline`, ...). An unsupervised head keeps its init
    weights and emits a near-constant 0.5 — the uniform prior over its bins —
    which is meaningless and frequently anti-correlated with real progress.

    `scripts/eval/render.py:load_checkpoint` stamps `_warp_rm_abs_head_trained`
    from the checkpoint's ablation. Models without the mark (built in-process,
    or legacy) default to True, preserving the original behavior.
    """
    if not (hasattr(model, "abs_progress_head") and hasattr(model, "abs_bin_centers")):
        return False
    return bool(getattr(model, "_warp_rm_abs_head_trained", True))


# ── Shared helpers for fused / bulk paths ────────────────────────────────────


def _plan_episode_windows(
    n_feat: int, window_size: int, standard_feat_steps: int
) -> tuple[list[list[int]], int, float]:
    """Build window frame-index lists for one episode.

    Returns:
        all_fi: list of length-N frame-index lists; empty if the episode is
            too short to produce any window.
        std_step: the actual stride used (may be smaller than requested for
            short episodes).
        vel_scale: rescale factor so short-episode velocities stay comparable
            to standard-stride ones.
    """
    N = window_size
    std_step = standard_feat_steps
    max_start = n_feat - (N - 1) * std_step - 1
    if max_start <= 0 and n_feat >= N + 1:
        std_step = max(1, (n_feat - 2) // (N - 1))
        max_start = n_feat - (N - 1) * std_step - 1

    if max_start <= 0:
        return [], std_step, 1.0

    vel_scale = float(standard_feat_steps) / float(std_step)
    starts = list(range(0, max_start + 1))
    all_fi = [[min(s + i * std_step, n_feat - 1) for i in range(N)] for s in starts]
    return all_fi, std_step, vel_scale


def _aggregate_relative(
    all_fi: list[list[int]],
    progress_preds: np.ndarray,  # (n_windows, N)
    n_feat: int,
    N: int,
    std_step: int,
    vel_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate window-local velocities into a per-frame absolute progress.

    Returns (abs_progress, avg_velocity), both (n_feat,) float32.

    Implementation: each (window, step) pair adds a constant ``vel`` to the
    range ``[f_start, f_end)``. Using the diff-then-cumsum trick, that's
    equivalent to ``+vel`` at ``f_start``, ``-vel`` at ``f_end``, then a single
    prefix sum — O(n_windows·N) scalar ops instead of O(n_windows·N·span).
    """
    fi_arr = np.asarray(all_fi, dtype=np.int64)  # (n_windows, N)
    f_starts = fi_arr[:, :-1].ravel()
    f_ends = np.minimum(fi_arr[:, 1:], n_feat).ravel()
    vel_flat = ((progress_preds[:, 1:] - progress_preds[:, :-1]) * (N - 1) * vel_scale).ravel()

    spans = f_ends - f_starts
    valid_s = spans > 0
    f_starts = f_starts[valid_s]
    f_ends = f_ends[valid_s]
    vel_flat = vel_flat[valid_s]

    diff_vel = (
        np.bincount(f_starts, weights=vel_flat, minlength=n_feat + 1)
        - np.bincount(f_ends, weights=vel_flat, minlength=n_feat + 1)
    )
    diff_cnt = (
        np.bincount(f_starts, minlength=n_feat + 1)
        - np.bincount(f_ends, minlength=n_feat + 1)
    )
    vel_sum = np.cumsum(diff_vel[:n_feat])
    vel_count = np.cumsum(diff_cnt[:n_feat])

    valid = vel_count > 0
    avg_vel = np.zeros(n_feat, dtype=np.float64)
    avg_vel[valid] = vel_sum[valid] / vel_count[valid]
    if valid.any() and not valid.all():
        xi = np.arange(n_feat)
        avg_vel = np.interp(xi, xi[valid], avg_vel[valid])

    # Cumulative integral -> absolute progress (no synthetic Gaussian smoothing).
    abs_progress = np.cumsum(avg_vel)
    p_min, p_max = abs_progress.min(), abs_progress.max()
    if p_max > p_min:
        abs_progress = (abs_progress - p_min) / (p_max - p_min)
    else:
        abs_progress = np.linspace(0, 1, n_feat)
    abs_progress = np.clip(abs_progress, 0.0, 1.0)

    return abs_progress.astype(np.float32), avg_vel.astype(np.float32)


def _aggregate_delta(
    all_fi: list[list[int]],
    delta_preds: np.ndarray,  # (n_windows, N) in label_anchor_frames units per step
    n_feat: int,
    label_anchor_frames: int,
    feature_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-step delta predictions into a per-feat-frame velocity
    trace and integrate to abs_progress.

    Delta-mode contract: prediction ``preds[j]`` at feat-index ``fi[j]`` is
    the per-step velocity in "anchor" units (e.g. at anchor=15 src frames,
    preds[j]=2.0 means the motion between fi[j-1] and fi[j] spanned 30 src
    frames — 2x the anchor interval).

    Aggregation: each (window, step j>=1) deposits a constant per-feat-step
    velocity ``preds[j] / (fi[j] - fi[j-1])`` across feat-frames in the
    half-open range [fi[j-1], fi[j]). Average across overlapping windows
    then cumsum to get abs_progress. No per-step-diff trick needed — the
    predictions ARE already deltas.
    """
    fi_arr = np.asarray(all_fi, dtype=np.int64)  # (n_windows, N)
    f_starts = fi_arr[:, :-1].ravel()
    f_ends = np.minimum(fi_arr[:, 1:], n_feat).ravel()
    # preds_j is the prediction for step j (j>=1), giving motion from fi[j-1] to fi[j].
    preds_j = delta_preds[:, 1:].ravel()
    spans_feat = (fi_arr[:, 1:] - fi_arr[:, :-1]).ravel()

    valid = spans_feat > 0
    f_starts = f_starts[valid]
    f_ends = f_ends[valid]
    vel_per_feat = preds_j[valid] / spans_feat[valid]

    # Same diff-cumsum trick as _aggregate_relative: constant vel over span
    diff_vel = (
        np.bincount(f_starts, weights=vel_per_feat, minlength=n_feat + 1)
        - np.bincount(f_ends, weights=vel_per_feat, minlength=n_feat + 1)
    )
    diff_cnt = (
        np.bincount(f_starts, minlength=n_feat + 1)
        - np.bincount(f_ends, minlength=n_feat + 1)
    )
    vel_sum = np.cumsum(diff_vel[:n_feat])
    vel_count = np.cumsum(diff_cnt[:n_feat])

    valid_mask = vel_count > 0
    avg_vel = np.zeros(n_feat, dtype=np.float64)
    avg_vel[valid_mask] = vel_sum[valid_mask] / vel_count[valid_mask]
    if valid_mask.any() and not valid_mask.all():
        xi = np.arange(n_feat)
        avg_vel = np.interp(xi, xi[valid_mask], avg_vel[valid_mask])

    # Cumulative integral -> absolute progress (no synthetic Gaussian smoothing).
    abs_progress = np.cumsum(avg_vel)
    p_min, p_max = abs_progress.min(), abs_progress.max()
    if p_max > p_min:
        abs_progress = (abs_progress - p_min) / (p_max - p_min)
    else:
        abs_progress = np.linspace(0, 1, n_feat)
    abs_progress = np.clip(abs_progress, 0.0, 1.0)

    return abs_progress.astype(np.float32), avg_vel.astype(np.float32)


@torch.no_grad()
def dense_inference_delta(
    model: nn.Module,
    feat_arr: np.ndarray,
    device: torch.device,
    window_size: int = 32,
    standard_feat_steps: int = 15,
    label_anchor_frames: int = 15,
    feature_stride: int = 3,
    batch_size: int = 512,
    text: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense inference for delta-labeled models.

    Each window yields N predictions, each a per-step delta (in anchor units).
    We spread these to per-feat-step velocity, average across overlapping
    windows, cumsum to recover abs_progress. Unlike relative-mode dense
    inference, no per-step-diff trick is needed — the model outputs velocities
    directly.

    Args mirror ``dense_inference_relative`` plus:
        label_anchor_frames: anchor used in DeltaVelocityLabeler training.
            Must match the checkpoint's ``label_anchor_frames`` metadata field.
        feature_stride: source-frame stride between adjacent feat indices.

    Returns (abs_progress, avg_velocity) — both (n_feat,) float32.
    """
    model.eval()
    n_feat = len(feat_arr)
    N = window_size

    all_fi, _std_step, _vel_scale = _plan_episode_windows(n_feat, N, standard_feat_steps)
    if not all_fi:
        return (np.linspace(0, 1, n_feat, dtype=np.float32),
                np.ones(n_feat, dtype=np.float32) / max(n_feat, 1))

    all_preds: list[np.ndarray] = []
    for bi in range(0, len(all_fi), batch_size):
        batch_fi = all_fi[bi:bi + batch_size]
        batch_feat = np.stack([feat_arr[fi] for fi in batch_fi])
        feats_t = torch.from_numpy(batch_feat).float().to(device)
        out = model(feats_t, text=(text.expand(feats_t.shape[0], -1) if text is not None else None))
        preds = (out[0] if isinstance(out, tuple) else out).cpu().numpy()
        all_preds.append(preds)
    delta_preds = np.concatenate(all_preds, axis=0)

    return _aggregate_delta(
        all_fi, delta_preds, n_feat,
        label_anchor_frames=label_anchor_frames,
        feature_stride=feature_stride,
    )


_ABS_KEYS = ("abs_progress", "abs_entropy", "abs_msp", "abs_margin", "abs_energy", "rel_entropy")


def _aggregate_absolute(
    all_fi: list[list[int]],
    per_window: dict[str, np.ndarray],  # each (n_windows, N)
    n_feat: int,
) -> dict[str, np.ndarray]:
    """Scatter per-window metrics to per-feature averages.

    Vectorized: flattens the per-window arrays into (n_windows·N,) streams and
    uses ``np.bincount`` to accumulate sums + counts per feature index in one
    C-level pass per key. The window-index planning already clamps into
    ``[0, n_feat-1]``, so no mask is required.
    """
    fi_flat = np.asarray(all_fi, dtype=np.int64).ravel()  # (n_windows * N,)
    counts = np.bincount(fi_flat, minlength=n_feat).astype(np.float64)

    valid = counts > 0
    if not valid.any():
        return {k: np.zeros(n_feat, dtype=np.float32) for k in _ABS_KEYS}

    result: dict[str, np.ndarray] = {}
    for k in _ABS_KEYS:
        weights = per_window[k].ravel().astype(np.float64, copy=False)
        s = np.bincount(fi_flat, weights=weights, minlength=n_feat)
        arr = np.zeros(n_feat, dtype=np.float32)
        arr[valid] = (s[valid] / counts[valid]).astype(np.float32)
        if not valid.all():
            xi = np.arange(n_feat)
            arr = np.interp(xi, xi[valid], arr[valid]).astype(np.float32)
        result[k] = arr
    return result


def _short_episode_abs_stub(n_feat: int) -> dict[str, np.ndarray]:
    return {
        "abs_progress": np.linspace(0, 1, n_feat, dtype=np.float32),
        "abs_entropy": np.zeros(n_feat, dtype=np.float32),
        "abs_msp": np.ones(n_feat, dtype=np.float32),
        "abs_margin": np.ones(n_feat, dtype=np.float32),
        "abs_energy": np.zeros(n_feat, dtype=np.float32),
        "rel_entropy": np.zeros(n_feat, dtype=np.float32),
    }


def _forward_batch(
    model: nn.Module,
    batch_feat_np: np.ndarray,
    device: torch.device,
    want_abs: bool,
    text: np.ndarray | None = None,
    demos: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run one model forward pass and extract rel + (optionally) abs outputs.

    Returns a dict with 'progress_preds' always, plus the 6 ``_ABS_KEYS`` when
    ``want_abs`` and the model has the abs head. All values are numpy on CPU.
    """
    feats_t = torch.from_numpy(batch_feat_np).float().to(device)
    text_t = torch.from_numpy(np.asarray(text, dtype=np.float32)).to(device) if text is not None else None
    demos_t = torch.from_numpy(np.asarray(demos, dtype=np.float32)).to(device) if demos is not None else None
    progress_preds, backbone_out, rel_logits = model(feats_t, text=text_t, demos=demos_t)
    out: dict[str, np.ndarray] = {"progress_preds": progress_preds.cpu().numpy()}

    if not want_abs:
        return out

    abs_logits = model.abs_progress_head(backbone_out)
    abs_probs = F.softmax(abs_logits, dim=-1)
    abs_log_probs = F.log_softmax(abs_logits, dim=-1)
    abs_preds = (abs_probs * model.abs_bin_centers).sum(dim=-1)
    abs_entropy = -(abs_probs * abs_log_probs).sum(dim=-1)
    abs_msp = abs_probs.max(dim=-1).values
    abs_top2 = abs_probs.topk(2, dim=-1).values
    abs_margin = abs_top2[..., 0] - abs_top2[..., 1]
    abs_energy = -torch.logsumexp(abs_logits, dim=-1)

    rel_probs = F.softmax(rel_logits, dim=-1)
    rel_log_probs = F.log_softmax(rel_logits, dim=-1)
    rel_entropy = -(rel_probs * rel_log_probs).sum(dim=-1)

    out.update({
        "abs_progress": abs_preds.cpu().numpy(),
        "abs_entropy": abs_entropy.cpu().numpy(),
        "abs_msp": abs_msp.cpu().numpy(),
        "abs_margin": abs_margin.cpu().numpy(),
        "abs_energy": abs_energy.cpu().numpy(),
        "rel_entropy": rel_entropy.cpu().numpy(),
    })
    return out


@torch.no_grad()
def dense_inference_both(
    model: nn.Module,
    feat_arr: np.ndarray,
    device: torch.device,
    window_size: int = 20,
    standard_feat_steps: int = 15,
    batch_size: int = 4096,
    want_abs: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Fused single-pass equivalent of ``dense_inference_relative`` + ``_absolute``.

    Runs each window batch through the model exactly once (half the GPU work)
    and derives rel + abs outputs from the same forward pass.

    Args:
        want_abs: If None, auto-detects from the model. If True on a model
            without an abs head, returns an empty dict.

    Returns:
        abs_progress: (n_feat,) velocity-integrated, normalized progress
        avg_velocity: (n_feat,) per-step velocity
        abs_metrics:  dict of (n_feat,) arrays — the 6 ``_ABS_KEYS`` if the
            abs head is present and requested, else empty.
    """
    model.eval()
    n_feat = len(feat_arr)
    N = window_size

    if want_abs is None:
        want_abs = has_abs_progress_head(model)
    elif want_abs:
        want_abs = has_abs_progress_head(model)

    all_fi, std_step, vel_scale = _plan_episode_windows(n_feat, N, standard_feat_steps)
    if not all_fi:
        abs_stub = _short_episode_abs_stub(n_feat) if want_abs else {}
        return (
            np.linspace(0, 1, n_feat, dtype=np.float32),
            np.ones(n_feat, dtype=np.float32) / max(n_feat, 1),
            abs_stub,
        )

    # Per-window stashes.
    progress_windows: list[np.ndarray] = []
    abs_stash: dict[str, list[np.ndarray]] = {k: [] for k in _ABS_KEYS} if want_abs else {}

    for bi in range(0, len(all_fi), batch_size):
        batch_fi = all_fi[bi:bi + batch_size]
        batch_feat = np.stack([feat_arr[fi] for fi in batch_fi])
        out = _forward_batch(model, batch_feat, device, want_abs)
        progress_windows.append(out["progress_preds"])
        if want_abs:
            for k in _ABS_KEYS:
                abs_stash[k].append(out[k])

    progress_preds_all = np.concatenate(progress_windows, axis=0)
    abs_progress, avg_vel = _aggregate_relative(
        all_fi, progress_preds_all, n_feat, N, std_step, vel_scale
    )

    if want_abs:
        per_window_abs = {k: np.concatenate(abs_stash[k], axis=0) for k in _ABS_KEYS}
        abs_metrics = _aggregate_absolute(all_fi, per_window_abs, n_feat)
    else:
        abs_metrics = {}

    return abs_progress, avg_vel, abs_metrics


@torch.no_grad()
def bulk_dense_inference(
    model: nn.Module,
    feat_list: list[np.ndarray],
    device: torch.device,
    window_size: int = 20,
    standard_feat_steps: int = 15,
    gpu_batch_size: int = 4096,
    want_abs: bool | None = None,
    text_list: list[np.ndarray] | None = None,
    demo_list: list[np.ndarray] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]]:
    """Cross-episode batching: pack windows from many episodes into GPU-sized
    batches so the GPU is saturated on every forward pass instead of being
    launched once per (tiny) episode.

    Post-processing aggregation still runs per-episode on CPU — it's cheap
    compared to the forward passes.

    Args:
        feat_list: list of (n_feat_i, D) feature arrays, one per episode.
            All must have the same D.

    Returns:
        list aligned with ``feat_list``: each entry is
        ``(abs_progress, avg_velocity, abs_metrics)``.
    """
    model.eval()
    N = window_size

    if want_abs is None:
        want_abs = has_abs_progress_head(model)
    elif want_abs:
        want_abs = has_abs_progress_head(model)

    # Plan windows per episode.
    plans: list[tuple[list[list[int]], int, float]] = []
    # Flatten: for each planned window, remember (ep_idx, within-ep window idx).
    flat_owner: list[tuple[int, int]] = []
    for ep_idx, feat_arr in enumerate(feat_list):
        all_fi, std_step, vel_scale = _plan_episode_windows(len(feat_arr), N, standard_feat_steps)
        plans.append((all_fi, std_step, vel_scale))
        for w_idx in range(len(all_fi)):
            flat_owner.append((ep_idx, w_idx))

    # Per-episode stashes: lists of per-batch numpy slices to be concatenated after the GPU loop.
    ep_progress: list[list[np.ndarray]] = [[] for _ in feat_list]
    ep_abs: list[dict[str, list[np.ndarray]]] = [
        {k: [] for k in _ABS_KEYS} if want_abs else {} for _ in feat_list
    ]

    # Stream through the GPU in large batches.
    total = len(flat_owner)
    cursor = 0
    while cursor < total:
        end = min(cursor + gpu_batch_size, total)
        # Assemble the batch's feature tensor + track per-item ownership.
        # trailing dims may be (D,) for single/concat features or (n_cam, D) for tokens fusion
        batch_feats = np.empty((end - cursor, N) + tuple(feat_list[0].shape[1:]), dtype=feat_list[0].dtype)
        owners_this_batch: list[tuple[int, int]] = flat_owner[cursor:end]
        for local_i, (ep_idx, w_idx) in enumerate(owners_this_batch):
            fi = plans[ep_idx][0][w_idx]
            batch_feats[local_i] = feat_list[ep_idx][fi]
        batch_text = (np.stack([text_list[ep_idx] for (ep_idx, _w) in owners_this_batch])
                      if text_list is not None else None)
        batch_demos = (np.stack([demo_list[ep_idx] for (ep_idx, _w) in owners_this_batch])
                       if demo_list is not None else None)
        out = _forward_batch(model, batch_feats, device, want_abs, text=batch_text, demos=batch_demos)
        # We'd like to scatter now by episode. Group local indices by owner episode.
        by_ep: dict[int, list[int]] = {}
        for local_i, (ep_idx, _w_idx) in enumerate(owners_this_batch):
            by_ep.setdefault(ep_idx, []).append(local_i)
        for ep_idx, local_is in by_ep.items():
            idxs = np.asarray(local_is, dtype=np.int64)
            ep_progress[ep_idx].append(out["progress_preds"][idxs])
            if want_abs:
                for k in _ABS_KEYS:
                    ep_abs[ep_idx][k].append(out[k][idxs])

        cursor = end

    # Per-episode aggregation on CPU.
    results: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = []
    for ep_idx, feat_arr in enumerate(feat_list):
        n_feat = len(feat_arr)
        all_fi, std_step, vel_scale = plans[ep_idx]
        if not all_fi:
            abs_stub = _short_episode_abs_stub(n_feat) if want_abs else {}
            results.append((
                np.linspace(0, 1, n_feat, dtype=np.float32),
                np.ones(n_feat, dtype=np.float32) / max(n_feat, 1),
                abs_stub,
            ))
            continue

        progress_all = np.concatenate(ep_progress[ep_idx], axis=0)
        abs_prog, avg_vel = _aggregate_relative(
            all_fi, progress_all, n_feat, N, std_step, vel_scale
        )
        if want_abs:
            per_window_abs = {k: np.concatenate(ep_abs[ep_idx][k], axis=0) for k in _ABS_KEYS}
            abs_metrics = _aggregate_absolute(all_fi, per_window_abs, n_feat)
        else:
            abs_metrics = {}
        results.append((abs_prog, avg_vel, abs_metrics))

    return results


@torch.no_grad()
def dense_inference_absolute(
    model: nn.Module,
    feat_arr: np.ndarray,
    device: torch.device,
    window_size: int = 20,
    standard_feat_steps: int = 15,
    batch_size: int = 512,
    text: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """
    Dense inference using the absolute progress head and C51 uncertainty metrics.

    Only works for TransformerAggregator with abs_progress_head.
    Returns empty dict if model doesn't have the head.

    Args:
        model: Enhanced temporal aggregator with abs_progress_head
        feat_arr: (n_feat, D) pre-computed backbone features
        device: Torch device
        window_size: Number of frames per window
        standard_feat_steps: Stride between frames in each window
        batch_size: Inference batch size

    Returns:
        dict with keys:
            abs_progress: (n_feat,) float32 direct absolute progress from C51 head
            abs_entropy:  (n_feat,) float32 predictive entropy (higher = more uncertain)
            abs_msp:      (n_feat,) float32 max softmax probability (higher = more confident)
            abs_margin:   (n_feat,) float32 margin between top-2 probs
            abs_energy:   (n_feat,) float32 energy score
            rel_entropy:  (n_feat,) float32 relative head entropy
        Returns empty dict if model doesn't have abs_progress_head.
    """
    if not has_abs_progress_head(model):
        return {}

    model.eval()
    n_feat = len(feat_arr)
    N = window_size
    std_step = standard_feat_steps

    max_start = n_feat - (N - 1) * std_step - 1
    if max_start <= 0:
        return {
            "abs_progress": np.linspace(0, 1, n_feat, dtype=np.float32),
            "abs_entropy": np.zeros(n_feat, dtype=np.float32),
            "abs_msp": np.ones(n_feat, dtype=np.float32),
            "abs_margin": np.ones(n_feat, dtype=np.float32),
            "abs_energy": np.zeros(n_feat, dtype=np.float32),
            "rel_entropy": np.zeros(n_feat, dtype=np.float32),
        }

    starts = list(range(0, max_start + 1))
    all_fi = [[min(s + i * std_step, n_feat - 1) for i in range(N)] for s in starts]

    # Accumulate predictions + uncertainty per feature position
    keys = ["abs_progress", "abs_entropy", "abs_msp", "abs_margin", "abs_energy", "rel_entropy"]
    sums = {k: np.zeros(n_feat, dtype=np.float64) for k in keys}
    counts = np.zeros(n_feat, dtype=np.float64)

    for bi in range(0, len(all_fi), batch_size):
        batch_fi = all_fi[bi:bi + batch_size]
        batch_feat = np.stack([feat_arr[fi] for fi in batch_fi])
        feats_t = torch.from_numpy(batch_feat).float().to(device)

        # Forward: get backbone_out and rel_logits
        _, backbone_out, rel_logits = model(feats_t, text=(text.expand(feats_t.shape[0], -1) if text is not None else None))

        # Absolute progress: C51 softmax expectation
        abs_logits = model.abs_progress_head(backbone_out)  # (B, T, n_abs_bins)
        abs_probs = F.softmax(abs_logits, dim=-1)
        abs_preds = (abs_probs * model.abs_bin_centers).sum(dim=-1)  # (B, T)

        # Uncertainty metrics from abs head
        abs_log_probs = F.log_softmax(abs_logits, dim=-1)
        abs_entropy = -(abs_probs * abs_log_probs).sum(dim=-1)  # (B, T)
        abs_msp = abs_probs.max(dim=-1).values  # (B, T)
        abs_top2 = abs_probs.topk(2, dim=-1).values
        abs_margin = abs_top2[..., 0] - abs_top2[..., 1]  # (B, T)
        abs_energy = -torch.logsumexp(abs_logits, dim=-1)  # (B, T)

        # Relative head entropy
        rel_probs = F.softmax(rel_logits, dim=-1)
        rel_log_probs = F.log_softmax(rel_logits, dim=-1)
        rel_entropy = -(rel_probs * rel_log_probs).sum(dim=-1)  # (B, T)

        # Move to numpy
        batch_vals = {
            "abs_progress": abs_preds.cpu().numpy(),
            "abs_entropy": abs_entropy.cpu().numpy(),
            "abs_msp": abs_msp.cpu().numpy(),
            "abs_margin": abs_margin.cpu().numpy(),
            "abs_energy": abs_energy.cpu().numpy(),
            "rel_entropy": rel_entropy.cpu().numpy(),
        }

        # Accumulate per feature position
        for w_idx, fi in enumerate(batch_fi):
            for j, fj in enumerate(fi):
                if fj < n_feat:
                    for k in keys:
                        sums[k][fj] += batch_vals[k][w_idx, j]
                    counts[fj] += 1

    # Average and return
    valid = counts > 0
    result = {}
    for k in keys:
        arr = np.zeros(n_feat, dtype=np.float32)
        arr[valid] = (sums[k][valid] / counts[valid]).astype(np.float32)
        if valid.any() and not valid.all():
            xi = np.arange(n_feat)
            arr = np.interp(xi, xi[valid], arr[valid]).astype(np.float32)
        result[k] = arr

    return result

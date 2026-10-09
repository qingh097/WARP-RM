"""
Episode rendering pipeline.

Renders side-by-side videos: source video + reward model visualization.
Supports both cached features and on-the-fly extraction.
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..data.dataset import Episode
from ..data.preprocess import resize_frame
from ..data.video_reader import read_frames, clear_video_cache
from .inference import dense_inference_relative, dense_inference_absolute, has_abs_progress_head
from .plotting import (
    render_base_plot_with_gt, render_base_plot_no_gt,
    render_base_plot_enhanced,
    draw_scrubber, draw_value_dots, draw_zoomed_velocity,
    compose_frame, hex_to_bgr, C_RECON, C_GT, C_VEL_POS, C_ABS_DIRECT,
)


def extract_features_on_the_fly(
    encoder: nn.Module,
    episode: Episode,
    device: torch.device,
    feature_stride: int = 3,
    image_size: int = 224,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    batch_size: int = 256,
    crop_mode: str = "squash",
) -> np.ndarray:
    """
    Extract backbone features on-the-fly (no cache required).

    ``crop_mode`` must match what the checkpoint was trained with, or the
    features land in a different distribution than the model expects — see
    `warp_rm.data.preprocess`.

    Returns (n_feat, D) float32 array matching the training cache format.
    """
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    if std is None:
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    n_feat = (episode.n_frames + feature_stride - 1) // feature_stride
    frame_indices = [min(j * feature_stride, episode.n_frames - 1) for j in range(n_feat)]
    # v3.0 concat shards: shift relative indices by frame_offset so we read
    # the requested episode's frames, not the shard's first episode. The
    # canonical training path does this in warp_rm/utils/caching.py:120-121.
    if episode.frame_offset:
        frame_indices = [i + episode.frame_offset for i in frame_indices]

    frames_raw = read_frames(episode.video_path, frame_indices)
    clear_video_cache()

    encoder.eval()
    ep_features = []
    for bi in range(0, len(frames_raw), batch_size):
        batch_raw = frames_raw[bi:bi + batch_size]
        processed = []
        for frame in batch_raw:
            frame = resize_frame(frame, image_size, crop_mode)
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - mean) / std
            processed.append(frame)
        imgs_t = torch.from_numpy(
            np.stack(processed).transpose(0, 3, 1, 2)
        ).float().to(device)
        with torch.no_grad():
            feats = encoder(imgs_t).cpu().numpy()
        ep_features.append(feats)

    return np.concatenate(ep_features, axis=0)


def render_episode(
    model: nn.Module,
    episode: Episode,
    out_path: str,
    device: torch.device,
    feat_arr: np.ndarray | None = None,
    encoder: nn.Module | None = None,
    feature_stride: int = 3,
    image_size: int = 224,
    backbone_mean: np.ndarray | None = None,
    backbone_std: np.ndarray | None = None,
    window_size: int = 20,
    standard_feat_steps: int = 15,
    target_h: int = 480,
    out_fps: float = 15.0,
    show_gt: bool = False,
    crop_mode: str = "squash",
    text=None,
    demos=None,
):
    """
    Full rendering pipeline: features -> dense inference -> video with plots.

    Args:
        model: Temporal aggregator model
        episode: Episode to render
        out_path: Output video path
        device: Torch device
        feat_arr: Pre-computed features (n_feat, D). If None, extracts on-the-fly.
        encoder: Backbone encoder (required if feat_arr is None)
        feature_stride: Feature extraction stride
        image_size: Image size for backbone
        backbone_mean/std: Normalization params for backbone
        window_size: Inference window size
        standard_feat_steps: Standard stride in feature steps
        target_h: Output video height
        out_fps: Output video FPS
        show_gt: If True, include ground truth row (linear 0->1)
        crop_mode: Frame geometry; must match the checkpoint's training mode
    """
    if not episode.video_path.exists():
        print(f"  Video not found: {episode.video_path}")
        return

    # 1. Get features
    if feat_arr is None:
        if encoder is None:
            raise ValueError("Either feat_arr or encoder must be provided")
        print(f"  Extracting features ({episode.n_frames} src frames)...")
        feat_arr = extract_features_on_the_fly(
            encoder, episode, device, feature_stride, image_size,
            backbone_mean, backbone_std, crop_mode=crop_mode,
        )

    print(f"  Running dense inference ({feat_arr.shape[0]} feature frames)...")
    abs_progress, raw_velocity = dense_inference_relative(
        model, feat_arr, device,
        window_size=window_size,
        standard_feat_steps=standard_feat_steps,
        text=text, demos=demos,
    )

    # Check if model has abs_progress_head for enhanced rendering
    has_abs_head = has_abs_progress_head(model)
    unc_dict = {}
    if has_abs_head:
        print(f"  Running dense inference for absolute progress + uncertainty...")
        unc_dict = dense_inference_absolute(
            model, feat_arr, device,
            window_size=window_size,
            standard_feat_steps=standard_feat_steps,
            text=text, demos=demos,
        )

    # 2. Interpolate to source-frame resolution
    n_frames = episode.n_frames
    t_src = np.linspace(0, 1, n_frames, dtype=np.float32)
    t_feat = np.linspace(0, 1, len(abs_progress), dtype=np.float32)
    abs_frames = np.interp(t_src, t_feat, abs_progress).astype(np.float32)
    vel_frames = np.interp(t_src, t_feat, raw_velocity).astype(np.float32)

    # Interpolate abs head outputs if available
    dabs_frames = None
    abs_entropy_frames = None
    abs_msp_frames = None
    abs_margin_frames = None
    abs_energy_frames = None
    if unc_dict:
        dabs_frames = np.interp(t_src, t_feat, unc_dict["abs_progress"]).astype(np.float32)
        abs_entropy_frames = np.interp(t_src, t_feat, unc_dict["abs_entropy"]).astype(np.float32)
        abs_msp_frames = np.interp(t_src, t_feat, unc_dict["abs_msp"]).astype(np.float32)
        abs_margin_frames = np.interp(t_src, t_feat, unc_dict["abs_margin"]).astype(np.float32)
        abs_energy_frames = np.interp(t_src, t_feat, unc_dict["abs_energy"]).astype(np.float32)

    # 3. Open source video
    cap = cv2.VideoCapture(str(episode.video_path))
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if total_src <= 0 or src_fps <= 0:
        print("  Bad video metadata, skipping.")
        cap.release()
        return

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_w = int(src_w * target_h / src_h)
    plot_w = int(vid_w * 0.85)

    # 4. Render base plot
    vel_min = float(vel_frames.min())
    vel_max = float(vel_frames.max())
    t_arr = np.linspace(0, 1, n_frames)

    if has_abs_head and dabs_frames is not None:
        # Enhanced 4-row plot: reconstructed abs, direct abs, velocity, zoomed velocity
        base_plot, ax_x0, ax_x1, ax_bboxes, zoom_rect, vel_ylim = render_base_plot_enhanced(
            abs_frames, vel_frames, plot_w, target_h,
            title=episode.path.name, direct_abs=dabs_frames,
        )
        # Dots on first 3 rows (recon, direct_abs, velocity)
        signals = [(t_arr, abs_frames), (t_arr, dabs_frames), (t_arr, vel_frames)]
        colors = [hex_to_bgr(C_RECON), hex_to_bgr(C_ABS_DIRECT), (200, 200, 200)]
        ylims = [(0.0, 1.0), (0.0, 1.0), vel_ylim]
        dot_bboxes = ax_bboxes[:3]
    elif show_gt:
        gt_frames = np.linspace(0, 1, n_frames, dtype=np.float32)
        base_plot, ax_x0, ax_x1, ax_bboxes = render_base_plot_with_gt(
            abs_frames, vel_frames, gt_frames,
            plot_w, target_h, title=episode.path.name,
        )
        signals = [(t_arr, gt_frames), (t_arr, abs_frames), (t_arr, vel_frames)]
        colors = [hex_to_bgr(C_GT), hex_to_bgr(C_RECON), (200, 200, 200)]
        ylims = [(0.0, 1.0), (0.0, 1.0),
                 (min(vel_min, 0.0), max(vel_max, 0.0) or 1.0)]
        zoom_rect = None
        vel_ylim = None
        dot_bboxes = ax_bboxes
    else:
        base_plot, ax_x0, ax_x1, ax_bboxes, zoom_rect, vel_ylim = render_base_plot_no_gt(
            abs_frames, vel_frames,
            plot_w, target_h, title=episode.path.name,
        )
        signals = [(t_arr, abs_frames), (t_arr, vel_frames)]
        colors = [hex_to_bgr(C_RECON), (200, 200, 200)]
        ylims = [(0.0, 1.0), vel_ylim]
        dot_bboxes = ax_bboxes[:2]

    # Precompute confidence from abs head entropy (for text overlay)
    n_abs_bins = 50
    max_abs_entropy = float(np.log(n_abs_bins))
    confidence_frames = None
    if abs_entropy_frames is not None:
        confidence_frames = np.clip(1.0 - abs_entropy_frames / max_abs_entropy, 0.0, 1.0)
    # Get text position for uncertainty overlay (row 1 in enhanced mode)
    conf_label_x = None
    conf_label_y = None
    if has_abs_head and len(ax_bboxes) >= 2:
        row1_ytop, _ = ax_bboxes[1]
        conf_label_x = int(ax_x1) - 220
        conf_label_y = int(row1_ytop) + 18

    # 5. Write video frame-by-frame
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (vid_w + plot_w, target_h))

    frame_step = max(1, round(src_fps / out_fps))
    frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            t_norm = frame_idx / max(total_src - 1, 1)
            plot_frame = draw_scrubber(base_plot, t_norm, ax_x0, ax_x1)
            plot_frame = draw_value_dots(plot_frame, t_norm, ax_x0, ax_x1,
                                         dot_bboxes, signals, colors, ylims)
            if zoom_rect is not None and vel_ylim is not None:
                plot_frame = draw_zoomed_velocity(
                    plot_frame, t_norm, vel_frames, zoom_rect, vel_ylim,
                )
            # Uncertainty text overlay for enhanced models
            if confidence_frames is not None and conf_label_x is not None:
                fi = min(int(t_norm * (len(confidence_frames) - 1)),
                         len(confidence_frames) - 1)
                conf_pct = confidence_frames[fi] * 100.0
                msp_val = abs_msp_frames[fi]
                mar_val = abs_margin_frames[fi]
                ene_val = abs_energy_frames[fi]
                unc_text = f"Conf:{conf_pct:.0f}% MSP:{msp_val:.2f} Margin:{mar_val:.2f} E:{ene_val:.1f}"
                cv2.putText(plot_frame, unc_text, (conf_label_x, conf_label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

            ts = f"{frame_idx / src_fps:.1f}s / {total_src / src_fps:.1f}s"
            cv2.putText(frame, ts, (10, target_h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            combined = compose_frame(frame, plot_frame, target_h)
            writer.write(combined)
            written += 1
        frame_idx += 1

    cap.release()
    writer.release()

    # 6. Re-encode for compatibility
    tmp = out_path + ".tmp.mp4"
    os.rename(out_path, tmp)
    ret = os.system(
        f'ffmpeg -y -i "{tmp}" -vcodec libx264 -crf 35 -preset fast '
        f'-vf scale=-2:{target_h} -pix_fmt yuv420p '
        f'"{out_path}" -loglevel error'
    )
    if ret == 0:
        os.remove(tmp)
    else:
        os.rename(tmp, out_path)

    print(f"  Done: {written} frames -> {out_path}")

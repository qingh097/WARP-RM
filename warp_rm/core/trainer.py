"""
Enhanced training loop supporting both original WARP-RM baseline and
the enhanced model with C51/temporal diffs/stochastic depth/abs progress.

Controlled by AblationConfig to enable/disable individual features.
"""

import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from ..data.dataset import Episode, load_fused_features
from ..data.samplers import EvalSampler
from ..data.labelers import LabelGenerator
from ..data.video_reader import read_frames
from .metrics import EvalMetrics, MetricTracker
from .ablation import AblationConfig
from .loss import RMLoss
from .run_logger import RunLogger
from .wandb_logger import WandbLogger


class Trainer:
    """
    Training loop for ablation experiments.

    Handles both:
      - (Legacy "baseline" was CausalTransformerAggregator + Huber; dropped in WARP-RM column rename.)
      - Enhanced: TransformerAggregator with C51 + temporal diffs

    The AblationConfig gates which features are active.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader,  # DataLoader or CachedVideoLoader
        val_episodes: list[Episode],
        ep_meta: dict | None,
        labeler: LabelGenerator,
        device: torch.device,
        eval_sampler: EvalSampler,
        ablation_config: AblationConfig,
        # Training params
        base_lr: float = 1e-4,
        warmup_steps: int = 1000,
        max_steps: int = 999999,
        total_budget: float = 900.0,
        eval_every: int = 100,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        branch_tag: str = "unknown",
        # Feature params for dense scan
        window_size: int = 20,
        standard_feat_steps: int = 15,
        # Online mode
        backbone: nn.Module | None = None,
        feature_stride: int = 3,
        image_size: int = 224,
        wandb_logger: WandbLogger | None = None,
        focal_gamma: float = 0.0,
        focal_on: str = "none",
        tri_state_weight: float = 0.0,
        tri_state_eps: float = 0.02,
        completion_weight: float = 0.0,
        completion_focal_gamma: float = 2.0,
        completion_pos_weight: float = 2.0,
        smoothness_weight: float = 0.0,
        c51_label_smoothing_sigma: float = 0.0,
        train_sampler=None,
        eval_window_k: int = 16,
        label_mode: str = "relative",
        label_anchor_frames: int = 15,
        run_logger: RunLogger | None = None,
        lang_cond: bool = False,
        demo_cond: bool = False,
        demo_dropout: float = 0.5,
        demo_index: dict | None = None,
    ):
        self.lang_cond = lang_cond
        self.demo_cond = demo_cond; self.demo_dropout = float(demo_dropout); self.demo_index = demo_index or {}
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_episodes = val_episodes
        self.ep_meta = ep_meta
        # Multi-camera fusion mode, read from the model so eval fuses the same
        # way training does. Defaults to "concat" (== single-camera behavior)
        # for models built before the multi-cam flag existed.
        self.fusion = getattr(model, "fusion", "concat")
        self.labeler = labeler
        self.device = device
        self.eval_sampler = eval_sampler
        self.ablation = ablation_config
        self.loss_fn = RMLoss(
            ablation_config,
            focal_gamma=focal_gamma, focal_on=focal_on,
            tri_state_weight=tri_state_weight, tri_state_eps=tri_state_eps,
            completion_weight=completion_weight,
            completion_focal_gamma=completion_focal_gamma,
            completion_pos_weight=completion_pos_weight,
            smoothness_weight=smoothness_weight,
            c51_label_smoothing_sigma=c51_label_smoothing_sigma,
        )
        self.train_sampler = train_sampler
        self.eval_window_k = eval_window_k
        self.label_mode = label_mode
        self.label_anchor_frames = int(label_anchor_frames)
        self.run_logger = run_logger

        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.total_budget = total_budget
        self.eval_every = eval_every
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        self.branch_tag = branch_tag
        self.window_size = window_size
        self.standard_feat_steps = standard_feat_steps
        self.wandb = wandb_logger or WandbLogger(project=None)

        # Online mode
        self.backbone = backbone
        self.feature_stride = feature_stride
        self.image_size = image_size
        if backbone is not None:
            self.backbone_mean = backbone.MEAN
            self.backbone_std = backbone.STD

        self.best_val_loss = float("inf")
        self.best_metrics = EvalMetrics(
            loss=float("inf"), vel_spearman=0.0, vel_sign_acc_all=0.0,
            mid_rewind_sign=0.0, fwd_sign=0.0, rew_sign=0.0,
            cum_sign=0.0, cum_pos_sign=0.0, cum_neg_sign=0.0,
            vel_corr=0.0, spearman=0.0,
        )

    def train(self) -> EvalMetrics:
        """Run the full training loop. Returns best metrics."""
        train_start = time.time()
        step = 0
        self._train_iter = iter(self.train_loader)
        uses_abs = self.ablation.use_abs_progress
        if self.run_logger is not None:
            self.run_logger.start()

        while step < self.max_steps:
            elapsed = time.time() - train_start
            if elapsed > self.total_budget:
                print(f"Budget exceeded ({elapsed:.0f}s), stopping")
                break

            self.model.train()
            if self._train_iter is None:
                self._train_iter = iter(self.train_loader)
            try:
                batch = next(self._train_iter)
            except StopIteration:
                self._train_iter = iter(self.train_loader)
                batch = next(self._train_iter)

            completion_targets = None
            abs_labels = None
            batch_data, labels = batch[0], batch[1]
            # Remaining tensors appear in order: abs_labels?, completion?
            # — same order the dataset appended them.
            extras = list(batch[2:])
            text = extras.pop().to(self.device) if (self.lang_cond and extras) else None
            demos = extras.pop().to(self.device) if (self.demo_cond and extras) else None
            if demos is not None and random.random() < self.demo_dropout:
                demos = None   # demo dropout: the model must also work from text / frames alone
            if uses_abs and extras:
                abs_labels = extras.pop(0).to(self.device)
            if self.loss_fn.completion_weight > 0 and extras:
                completion_targets = extras.pop(0).to(self.device)

            batch_data = batch_data.to(self.device)
            labels = labels.to(self.device)

            # Online mode: run frozen backbone on images
            if self.backbone is not None:
                B, W, C, H, W_img = batch_data.shape
                with torch.no_grad():
                    features = self.backbone(batch_data.view(B * W, C, H, W_img))
                features = features.view(B, W, -1)
            else:
                features = batch_data

            # ── Forward ─────────────────────────────────────────────────────
            if self.ablation.is_baseline:
                # Original WARP-RM: returns (B, T) only
                rewards = self.model(features, text=text, demos=demos)
                progress_preds = rewards
                backbone_out = None
                rel_logits = None
            else:
                progress_preds, backbone_out, rel_logits = self.model(features, text=text, demos=demos)

            # ── Loss ────────────────────────────────────────────────────────
            current_lr = self.optimizer.param_groups[0]["lr"]

            sample_weights = None

            loss, components = self.loss_fn(
                progress_preds=progress_preds,
                backbone_out=backbone_out,
                rel_logits=rel_logits,
                model=self.model,
                labels=labels,
                abs_labels=abs_labels,
                completion_targets=completion_targets,
                current_lr=current_lr,
                base_lr=self.base_lr,
                sample_weights=sample_weights,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            # Warmup then scheduler
            if step < self.warmup_steps:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.base_lr * (step + 1) / self.warmup_steps
            else:
                self.scheduler.step()

            if step % 200 == 0:
                parts = [f"Step {step:6d} | loss: {loss.item():.5f}"]
                if not self.ablation.is_baseline:
                    parts.append(f"prog: {components['progress']:.5f}")
                    if self.ablation.use_abs_progress:
                        parts.append(f"abs: {components['abs']:.5f}")
                    if self.loss_fn.completion_weight > 0:
                        parts.append(f"comp: {components.get('completion', 0.0):.5f}")
                    if self.loss_fn.smoothness_weight > 0:
                        parts.append(f"sm: {components.get('smoothness', 0.0):.5f}")
                parts.append(f"lr: {current_lr:.2e}")
                parts.append(f"t: {time.time() - train_start:.0f}s")
                print(" | ".join(parts))
                payload = {
                    "train/loss": loss.item(),
                    "train/lr": current_lr,
                    "train/elapsed_s": time.time() - train_start,
                }
                if not self.ablation.is_baseline:
                    payload["train/loss_progress"] = components["progress"]
                    if self.ablation.use_abs_progress:
                        payload["train/loss_abs"] = components["abs"]
                    if self.loss_fn.completion_weight > 0:
                        payload["train/loss_completion"] = components.get("completion", 0.0)
                    if self.loss_fn.smoothness_weight > 0:
                        payload["train/loss_smoothness"] = components.get("smoothness", 0.0)
                self.wandb.log(payload, step=step)

            if step < 3:
                print(f"  DEBUG labels: mean={labels.mean():.3f} std={labels.std():.3f} "
                      f"min={labels.min():.3f} max={labels.max():.3f} | "
                      f"preds: mean={progress_preds.mean():.3f} std={progress_preds.std():.3f}")

            if step % self.eval_every == 0 and step > 0:
                metrics = self.evaluate()
                new_best = self._maybe_save_checkpoint(metrics, step)
                self.wandb.log_metrics(metrics, step=step, prefix="val")

                if self.run_logger is not None:
                    self.run_logger.eval(step=step, metrics=metrics.to_dict())
                    if new_best:
                        ckpt_tag = f"{self.branch_tag}_{self.ablation.name}"
                        self.run_logger.checkpoint(
                            step=step,
                            path=os.path.join(self.checkpoint_dir, f"best_model_{ckpt_tag}.pt"),
                            composite=metrics.composite_score,
                        )

            step += 1

        # Final evaluation
        print("\nFinal evaluation...")
        final_metrics = self.evaluate()
        self.wandb.log_metrics(final_metrics, step=step, prefix="final")

        self.wandb.finish()
        training_seconds = time.time() - train_start
        returned = (self.best_metrics
                    if self.best_metrics.composite_score > final_metrics.composite_score
                    else final_metrics)
        if returned is self.best_metrics:
            print(f"Using best checkpoint (composite {self.best_metrics.composite_score:.3f})")
        if self.run_logger is not None:
            self.run_logger.finish(
                status="succeeded",
                final_metrics=returned.to_dict(),
                training_seconds=training_seconds,
            )
            self.run_logger.spawn_uploader()
        return returned

    def _encode_frames_online(self, ep: Episode, feat_indices: list[int]) -> torch.Tensor:
        """Read frames from MP4, preprocess, and run through backbone. Returns (1, W, D)."""
        import cv2
        frame_indices = [min(fi * self.feature_stride, ep.n_frames - 1) for fi in feat_indices]
        # v3.0 concat shards: shift by frame_offset so we read this episode's
        # frames, not the shard's first episode. Same fix as caching.py:120-121.
        if ep.frame_offset:
            frame_indices = [i + ep.frame_offset for i in frame_indices]
        frames = read_frames(ep.video_path, frame_indices)
        processed = []
        for frame in frames:
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - self.backbone_mean) / self.backbone_std
            processed.append(frame.transpose(2, 0, 1))
        imgs = torch.from_numpy(np.stack(processed)).float().to(self.device)
        with torch.no_grad():
            feats = self.backbone(imgs)
        return feats.unsqueeze(0)

    @torch.no_grad()
    def evaluate(self) -> EvalMetrics:
        """Evaluate on 8 window types x 2 windows x 20 episodes = 320 windows."""
        self.model.eval()
        tracker = MetricTracker()

        for ep in self.val_episodes:
            if self.backbone is not None:
                n_feat = (ep.n_frames + self.feature_stride - 1) // self.feature_stride
            else:
                meta = self.ep_meta[str(ep.path)]
                feat_arr = load_fused_features(meta, self.fusion)
                n_feat = meta["n_feat"]

            for fi, is_mid in self.eval_sampler.generate_eval_windows(n_feat):
                if self.backbone is not None:
                    features = self._encode_frames_online(ep, fi)
                else:
                    features = torch.from_numpy(feat_arr[fi]).float().unsqueeze(0).to(self.device)

                labels_np = self.labeler.generate(fi, ep.progress_per_frame, ep.n_frames)
                labels = torch.tensor([labels_np], dtype=torch.float32).to(self.device)

                if self.ablation.is_baseline:
                    preds = self.model(features, text=self._text_for(ep, features.shape[0]), demos=self._demos_for(ep, features.shape[0]))
                else:
                    preds, _, _ = self.model(features, text=self._text_for(ep, features.shape[0]), demos=self._demos_for(ep, features.shape[0]))

                loss_val = F.huber_loss(preds, labels, delta=0.01).item()
                tracker.update(preds[0].cpu().numpy(), labels_np, is_mid, loss_val)

        metrics = tracker.compute()
        metrics = self._dense_scan_spearman(metrics)
        metrics.print_summary()
        return metrics

    def _demos_for(self, ep, n: int = 1):
        if not getattr(self, "demo_cond", False) or not self.ep_meta:
            return None
        from ..data.dataset import build_demo_tokens
        import random as _r
        toks = build_demo_tokens(ep, self.ep_meta, self.demo_index, self.model.demo_k, self.model.demo_m,
                                 getattr(self.model, "fusion", "concat"), rng=_r.Random(0), vel=getattr(self.model, "demo_vel", False))
        t = torch.from_numpy(toks).to(self.device).unsqueeze(0)
        return t.expand(n, -1, -1)

    def _text_for(self, ep, n: int = 1):
        if not getattr(self, "lang_cond", False):
            return None
        meta = self.ep_meta[str(ep.path)] if self.ep_meta else {}
        emb = meta.get("text_emb")
        if emb is None:
            return None
        t = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(self.device).unsqueeze(0)
        return t.expand(n, -1)

    def _dense_scan_trace(self, ep, std_step: int) -> tuple[np.ndarray, np.ndarray, int]:
        """Run a forward-only dense sliding-window scan on one episode with the
        given per-step stride. Returns (raw_preds, pred_counts, n_feat).

        Every frame index `j` gets a predicted progress value as the mean over
        windows that covered it. Unvisited frames have pred_counts[j] == 0.
        """
        if self.backbone is not None:
            n_feat = (ep.n_frames + self.feature_stride - 1) // self.feature_stride
            feat_arr = None
        else:
            meta = self.ep_meta[str(ep.path)]
            feat_arr = load_fused_features(meta, self.fusion)
            n_feat = meta["n_feat"]

        scan_step = max(1, std_step * 3)
        raw_preds = np.zeros(n_feat, dtype=np.float32)
        pred_counts = np.zeros(n_feat, dtype=np.float32)

        stop = max(1, n_feat - (self.window_size - 1) * std_step)
        for sf in range(0, stop, scan_step):
            fi = [min(sf + i * std_step, n_feat - 1) for i in range(self.window_size)]
            if self.backbone is not None:
                feats = self._encode_frames_online(ep, fi)
            else:
                feats = torch.from_numpy(feat_arr[fi]).float().unsqueeze(0).to(self.device)
            if self.ablation.is_baseline:
                ps = self.model(feats, text=self._text_for(ep, feats.shape[0]), demos=self._demos_for(ep, feats.shape[0]))[0].cpu().numpy()
            else:
                ps = self.model(feats, text=self._text_for(ep, feats.shape[0]), demos=self._demos_for(ep, feats.shape[0]))[0][0].cpu().numpy()
            for j, fj in enumerate(fi):
                if fj < n_feat:
                    raw_preds[fj] += ps[j]
                    pred_counts[fj] += 1
        return raw_preds, pred_counts, n_feat

    def _window_spearman_one(self, ep, ep_idx: int) -> float:
        """Mean per-window Spearman(pred, label) across K training-sampler
        windows on one episode.

        Replaces the forward-only dense-scan-with-linear-GT computation.
        Uses the same sampler as training (self.train_sampler) and real
        labels from self.labeler — so the metric evaluates the model on
        the exact distribution it was trained on, with no linear-progress
        assumption.

        Deterministic via per-episode seeding so the eval is reproducible.
        """
        if self.train_sampler is None:
            return 0.0
        if self.backbone is not None:
            n_feat = (ep.n_frames + self.feature_stride - 1) // self.feature_stride
            feat_arr = None
        else:
            meta = self.ep_meta[str(ep.path)]
            feat_arr = load_fused_features(meta, self.fusion)
            n_feat = meta["n_feat"]

        # Seed deterministically so each eval evaluates on the same windows.
        rng_state = random.getstate()
        random.seed(0xEEEE + ep_idx * 7919)
        try:
            sps = []
            for _ in range(self.eval_window_k):
                fi = self.train_sampler.sample_indices(n_feat)
                labels = self.labeler.generate(fi, ep.progress_per_frame, ep.n_frames)
                if self.backbone is not None:
                    feats = self._encode_frames_online(ep, fi)
                else:
                    feats = torch.from_numpy(feat_arr[fi]).float().unsqueeze(0).to(self.device)
                if self.ablation.is_baseline:
                    ps = self.model(feats, text=self._text_for(ep, feats.shape[0]), demos=self._demos_for(ep, feats.shape[0]))[0].cpu().numpy()
                else:
                    ps = self.model(feats, text=self._text_for(ep, feats.shape[0]), demos=self._demos_for(ep, feats.shape[0]))[0][0].cpu().numpy()
                if np.std(ps) < 1e-8 or np.std(labels) < 1e-8:
                    continue
                sp, _ = spearmanr(ps, labels)
                if not np.isnan(sp):
                    sps.append(float(sp))
        finally:
            random.setstate(rng_state)
        return float(np.mean(sps)) if sps else 0.0

    def _speed_invariance_one(self, ep) -> float:
        """Speed-invariance score for a single episode.

        Run two dense-scans at different strides (1x and 2x standard) and
        measure pearson correlation of the prediction traces on the shared
        set of covered frames. A model that keys off video *content* should
        give identical traces regardless of playback stride → corr = 1.0.
        A model that overfit to stride distribution → corr < 1.0.
        """
        raw_1, count_1, _ = self._dense_scan_trace(ep, std_step=self.standard_feat_steps)
        raw_2, count_2, _ = self._dense_scan_trace(ep, std_step=2 * self.standard_feat_steps)
        both = (count_1 > 0) & (count_2 > 0)
        if both.sum() < 5:
            return 0.0
        t1 = raw_1[both] / count_1[both]
        t2 = raw_2[both] / count_2[both]
        if np.std(t1) < 1e-8 or np.std(t2) < 1e-8:
            return 0.0
        corr = float(np.corrcoef(t1, t2)[0, 1])
        return corr if not np.isnan(corr) else 0.0

    def _dense_scan_spearman(self, metrics: EvalMetrics) -> EvalMetrics:
        """Computes val_spearman (window-based, training-sampler, real labels)
        + val_speed_invariance (stride 1x vs 2x self-consistency).

        val_spearman used to be a forward-only dense-scan with linear pseudo-
        GT — philosophically wrong (real progress isn't linear-in-frame and
        forward-only can't test rewind/mid-flip). Replaced with mean
        per-window Spearman across K windows sampled from the training
        sampler, on each of the 20 clean-val episodes, using real labels.

        speed_invariance stays self-consistency — it doesn't use GT.
        """
        eps_sample = self.val_episodes
        per_ep_sp = [
            self._window_spearman_one(ep, ep_idx=i)
            for i, ep in enumerate(eps_sample)
        ]
        per_ep_si = [self._speed_invariance_one(ep) for ep in eps_sample]
        val_spearman = float(np.mean(per_ep_sp)) if per_ep_sp else 0.0
        speed_inv = float(np.mean(per_ep_si)) if per_ep_si else 0.0

        return EvalMetrics(
            loss=metrics.loss,
            vel_spearman=metrics.vel_spearman,
            vel_sign_acc_all=metrics.vel_sign_acc_all,
            mid_rewind_sign=metrics.mid_rewind_sign,
            fwd_sign=metrics.fwd_sign,
            rew_sign=metrics.rew_sign,
            cum_sign=metrics.cum_sign,
            cum_pos_sign=metrics.cum_pos_sign,
            cum_neg_sign=metrics.cum_neg_sign,
            vel_corr=metrics.vel_corr,
            spearman=val_spearman,
            speed_invariance=speed_inv,
            vel_magnitude_ratio=metrics.vel_magnitude_ratio,
            pred_magnitude_mean=metrics.pred_magnitude_mean,
            label_magnitude_mean=metrics.label_magnitude_mean,
            vel_net_magnitude_ratio=metrics.vel_net_magnitude_ratio,
            pred_net_magnitude_mean=metrics.pred_net_magnitude_mean,
            label_net_magnitude_mean=metrics.label_net_magnitude_mean,
        )

    def _maybe_save_checkpoint(self, metrics: EvalMetrics, step: int):
        """Save checkpoint if composite score improved. Returns True if saved."""
        score = metrics.composite_score
        best_score = self.best_metrics.composite_score
        is_better = score > best_score or (score == best_score and metrics.loss < self.best_val_loss)

        if is_better:
            self.best_val_loss = metrics.loss
            self.best_metrics = metrics
            print(f"  *** New best ({score:.3f}): vel_sp={metrics.vel_spearman:.3f} "
                  f"d[all:{metrics.vel_sign_acc_all:.3f} fwd:{metrics.fwd_sign:.3f} "
                  f"rew:{metrics.rew_sign:.3f}] "
                  f"c[all:{metrics.cum_sign:.3f} +:{metrics.cum_pos_sign:.3f} "
                  f"-:{metrics.cum_neg_sign:.3f}] "
                  f"cum_sp:{metrics.spearman:.3f}")

            tag = f"{self.branch_tag}_{self.ablation.name}"
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            ckpt = {
                "model": self.model.state_dict(),
                "step": step,
                "ablation": self.ablation.name,
                **metrics.to_dict(),
                "branch_tag": self.branch_tag,
            }
            # Extra metadata stamped by train_ablation (e.g. label_mode='delta',
            # label_anchor_frames=15, rel_bin_min/max for delta-mode ckpts).
            # Needed by downstream dense inference (score_dataset, webUI,
            # pair_fill) to pick the right aggregator.
            extra = getattr(self, "extra_ckpt_meta", {}) or {}
            ckpt.update(extra)
            ckpt_path = os.path.join(self.checkpoint_dir, f"best_model_{tag}.pt")
            torch.save(ckpt, ckpt_path)
            self._prune_old_checkpoints(tag)
            # Periodic mid-training S3 upload (gated on the WARP_RM_CKPT_S3_PREFIX env var).
            # Fire-and-forget — overwrites s3://…/best_model_{tag}.pt on every new best so
            # downstream consumers can grab the current-best ckpt without waiting for end-
            # of-job emit_result. Non-fatal if aws cli or network fails.
            s3_prefix = os.environ.get("WARP_RM_CKPT_S3_PREFIX")
            if s3_prefix:
                dest = s3_prefix.rstrip("/") + f"/best_model_{tag}.pt"
                try:
                    subprocess.Popen(
                        ["aws", "s3", "cp", ckpt_path, dest, "--no-progress"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass
            return True
        return False

    def _prune_old_checkpoints(self, tag: str, max_keep: int = 5):
        """Keep only the N most recent checkpoints + best."""
        ckpt_dir = Path(self.checkpoint_dir)
        if not ckpt_dir.exists():
            return
        ckpts = sorted(ckpt_dir.glob(f"*{tag}*.pt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        best = ckpt_dir / f"best_model_{tag}.pt"
        keep = set(ckpts[:max_keep]) | ({best} if best.exists() else set())
        for c in ckpts:
            if c not in keep:
                try:
                    c.unlink()
                except Exception:
                    pass

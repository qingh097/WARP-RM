"""Transformer aggregator used by WARP-RM.

The aggregator combines:
  - Temporal diffs: concatenate [features, features_t - features_{t-1}] → 2x input dim
  - C51 distributional reward head (categorical bins + softmax expectation)
  - Stochastic depth (random layer drop)
  - Absolute progress auxiliary C51 head
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerAggregator(nn.Module):
    """
    Bidirectional transformer with temporal diffs, C51 distributional output,
    stochastic depth, and absolute progress auxiliary head.

    Architecture:
      - Temporal diff computation: [f_t, f_t - f_{t-1}] → backbone_dim * 2
      - Linear input projection (backbone_dim * 2 → d_model, no bias)
      - Fixed sinusoidal positional encoding
      - TransformerEncoder with optional causal masking and stochastic depth
      - C51 reward head: Linear → softmax expectation over bins
      - C51 absolute progress head: Linear → softmax expectation over bins
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 12,
        dropout: float = 0.15,
        max_seq_len: int = 32,
        backbone_dim: int = 768,
        # Temporal diffs
        use_temporal_diffs: bool = True,
        lang_dim: int = 0,
        # C51 relative progress bins
        n_rel_bins: int = 30,
        rel_bin_min: float = -3.0,
        rel_bin_max: float = 3.0,
        # C51 absolute progress bins
        n_abs_bins: int = 50,
        # Stochastic depth
        stochastic_depth_p: float = 0.1,
        # First-frame-only positional encoding.
        first_frame_pe_only: bool = False,
        use_causal_attention: bool = False,
        # Multi-camera fusion mode.
        fusion: str = "concat",
        n_cameras: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.backbone_dim = backbone_dim
        self.n_layers = n_layers
        self.stochastic_depth_p = stochastic_depth_p
        self.use_temporal_diffs = use_temporal_diffs
        self.lang_dim = int(lang_dim)
        if self.lang_dim > 0:
            # language condition: one extra token (projected text embedding) prepended
            self.lang_proj = nn.Linear(self.lang_dim, d_model)
            self.lang_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.first_frame_pe_only = first_frame_pe_only
        self.use_causal_attention = use_causal_attention
        self.fusion = fusion
        self.n_cameras = n_cameras

        # In "tokens" mode, input_proj operates per-camera token.
        input_dim = backbone_dim * 2 if use_temporal_diffs else backbone_dim
        self.input_proj = nn.Linear(input_dim, d_model, bias=False)

        # Learned per-camera embedding (added to each projected camera token so
        # the transformer can distinguish cameras). Only present in "tokens"
        # mode — concat/single state_dicts must have no camera_embed.* key.
        if fusion == "tokens":
            self.camera_embed = nn.Embedding(n_cameras, d_model)

        # Fixed sinusoidal positional encoding.
        pe = torch.zeros(1, max_seq_len, d_model)
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        if first_frame_pe_only:
            # Keep only t=0 row; zero the rest.
            mask = torch.zeros(1, max_seq_len, 1)
            mask[0, 0, 0] = 1.0
            pe = pe * mask
        self.register_buffer("pos_embed", pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── C51 categorical head for relative progress ──────────────────────
        self.n_rel_bins = n_rel_bins
        self.rel_bin_min = rel_bin_min
        self.rel_bin_max = rel_bin_max
        self.reward_head = nn.Linear(d_model, n_rel_bins)
        self.register_buffer(
            "rel_bin_centers",
            torch.linspace(rel_bin_min, rel_bin_max, n_rel_bins),
        )

        # ── C51 categorical head for absolute progress ──────────────────────
        self.n_abs_bins = n_abs_bins
        self.abs_progress_head = nn.Linear(d_model, n_abs_bins)
        self.register_buffer(
            "abs_bin_centers",
            torch.linspace(0.0, 1.0, n_abs_bins),
        )

        # Per-token 3-way sign classification head.
        self.tri_state_head = nn.Linear(d_model, 3)

        # Per-token completion classification head.
        self.completion_head = nn.Linear(d_model, 1)

    def _build_token_sequence(self, features: torch.Tensor) -> torch.Tensor:
        """Late-fusion ("tokens") token construction.

        Args:
            features: (B, T, N, backbone_dim) — N per-camera feature vectors
                per timestep.
        Returns:
            x: (B, T*N, d_model) — one transformer token per (timestep, camera),
               with PER-CAMERA temporal diffs, a learned camera embedding, and
               the shared sinusoidal time PE (same time index across a
               timestep's N camera tokens).
        """
        B, T, N, _ = features.shape

        # Per-camera temporal diffs along TIME: token for (t, c) is
        # [f_{c,t}, f_{c,t} - f_{c,t-1}]. Diff is over dim=1 (time), per camera.
        if self.use_temporal_diffs:
            diffs = torch.zeros_like(features)
            diffs[:, 1:, :, :] = features[:, 1:, :, :] - features[:, :-1, :, :]
            x = self.input_proj(torch.cat([features, diffs], dim=-1))  # (B,T,N,d)
        else:
            x = self.input_proj(features)  # (B, T, N, d_model)

        # Learned per-camera embedding.
        cam_ids = torch.arange(N, device=features.device)
        x = x + self.camera_embed(cam_ids).view(1, 1, N, self.d_model)

        # Shared time positional encoding.
        x = x + self.pos_embed[:, :T, :].unsqueeze(2)  # (1,T,1,d) broadcast

        return x.reshape(B, T * N, self.d_model)

    def forward(
        self,
        features: torch.Tensor,
        text: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, T, backbone_dim) pre-computed backbone features for
                the "concat"/single-camera path, OR (B, T, N, backbone_dim) for
                the "tokens" (late-fusion) path.
        Returns:
            progress_preds: (B, T) relative progress predictions (softmax expectation)
            backbone_out:   (B, T, d_model) transformer output (for aux heads)
            rel_logits:     (B, T, n_rel_bins) raw C51 logits

        Output shapes are (B, T, *) in BOTH paths: in "tokens" mode the N camera
        tokens at each timestep are mean-pooled after the transformer, so the
        heads / labeler / loss are untouched.
        """
        tokens_mode = self.fusion == "tokens" and features.dim() == 4
        if tokens_mode:
            B, T, N, _ = features.shape
            x = self._build_token_sequence(features)  # (B, T*N, d_model)
            seq_len = T * N
        else:
            # concat / single-camera path — IDENTICAL to the original forward.
            _, T, _ = features.shape

            # Temporal diffs: [f_t, f_t - f_{t-1}]
            if self.use_temporal_diffs:
                diffs = torch.zeros_like(features)
                diffs[:, 1:, :] = features[:, 1:, :] - features[:, :-1, :]
                x = self.input_proj(torch.cat([features, diffs], dim=-1))
            else:
                x = self.input_proj(features)

            x = x + self.pos_embed[:, :T, :]
            seq_len = T

        has_lang = self.lang_dim > 0 and text is not None
        if has_lang:
            tok = self.lang_proj(text.float().to(x.device)).unsqueeze(1) + self.lang_pos  # (B,1,d)
            x = torch.cat([tok, x], dim=1)
            seq_len = seq_len + 1
        if self.use_causal_attention:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=features.device)
            is_causal = True
        else:
            mask = None
            is_causal = False

        # Stochastic depth: random layer drop during training
        if self.training and self.stochastic_depth_p > 0:
            for layer in self.transformer.layers:
                if random.random() < self.stochastic_depth_p:
                    continue  # skip this layer
                x = layer(x, src_mask=mask, is_causal=is_causal)
        else:
            x = self.transformer(x, mask=mask, is_causal=is_causal)

        # Tokens mode: mean-pool the N camera tokens at each timestep so the
        # per-timestep heads see one (B, T, d_model) feature per timestep.
        if has_lang:
            x = x[:, 1:]  # drop the language token before the per-timestep heads
        if tokens_mode:
            x = x.reshape(B, T, N, self.d_model).mean(dim=2)  # (B, T, d_model)

        backbone_out = x  # (B, T, d_model)

        # C51: logits -> softmax expectation for continuous prediction
        rel_logits = self.reward_head(x)  # (B, T, n_rel_bins)
        progress_preds = (F.softmax(rel_logits, dim=-1) * self.rel_bin_centers).sum(dim=-1)

        return progress_preds, backbone_out, rel_logits

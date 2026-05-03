# -*- coding: utf-8 -*-
"""
Rotary Position Embedding (RoPE) for irregular time series.

Adapted from the RoFormer paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
(https://arxiv.org/abs/2104.09864)

Key adaptation for irregular time series:
- Standard RoPE uses integer position indices
- This version uses continuous timestamps as position, enabling
  natural handling of non-uniform time intervals

Usage:
    rope = ContinuousRoPE(d_model)
    q, k = rope(q, k, timestamps)  # timestamps: [B, L] or [B, L, 1]
"""
import torch
import torch.nn as nn
import math


class ContinuousRoPE(nn.Module):
    """
    Continuous Rotary Position Embedding for irregular time series.

    Instead of using discrete position indices (0, 1, 2, ...),
    this module uses continuous timestamps to compute rotation angles,
    naturally handling irregular time intervals.

    Args:
        d_model: dimension of the model (must be even)
        max_freq: maximum frequency for the rotation (default: 10000.0)
        learnable: if True, frequency bases are learnable parameters
    """
    def __init__(self, d_model, max_freq=10000.0, learnable=False):
        super(ContinuousRoPE, self).__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        self.d_model = d_model
        self.half_dim = d_model // 2

        # Frequency bases: theta_i = 1 / (max_freq^(2i/d))
        freq_bases = 1.0 / (max_freq ** (torch.arange(0, self.half_dim).float() / self.half_dim))

        if learnable:
            self.freq_bases = nn.Parameter(freq_bases)
        else:
            self.register_buffer('freq_bases', freq_bases)

    def _compute_rotation(self, timestamps):
        """
        Compute rotation angles from continuous timestamps.

        Args:
            timestamps: [B, L] or [B, L, 1] - continuous time values

        Returns:
            cos_angles: [B, L, half_dim]
            sin_angles: [B, L, half_dim]
        """
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)  # [B, L]

        # angles = timestamps * freq_bases: [B, L, half_dim]
        angles = timestamps.unsqueeze(-1) * self.freq_bases.unsqueeze(0).unsqueeze(0)

        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        return cos_angles, sin_angles

    def _rotate_half(self, x):
        """
        Rotate half of the hidden dims of x.
        x: [..., d_model] -> split into [..., half_dim] pairs and rotate
        """
        x1 = x[..., :self.half_dim]
        x2 = x[..., self.half_dim:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, timestamps):
        """
        Apply rotary position embedding to query and key tensors.

        Args:
            q: [B, H, L, D] or [B, L, D] - query tensor
            k: [B, H, L, D] or [B, L, D] - key tensor
            timestamps: [B, L] or [B, L, 1] - continuous timestamps

        Returns:
            q_rotated: same shape as q
            k_rotated: same shape as k
        """
        cos_angles, sin_angles = self._compute_rotation(timestamps)

        # Handle multi-head case: [B, H, L, D]
        if q.dim() == 4:
            cos_angles = cos_angles.unsqueeze(1)  # [B, 1, L, half_dim]
            sin_angles = sin_angles.unsqueeze(1)

        # Expand cos/sin to full dimension by repeating
        cos_full = torch.cat([cos_angles, cos_angles], dim=-1)  # [..., d_model]
        sin_full = torch.cat([sin_angles, sin_angles], dim=-1)

        q_rotated = q * cos_full + self._rotate_half(q) * sin_full
        k_rotated = k * cos_full + self._rotate_half(k) * sin_full

        return q_rotated, k_rotated


class RoPEAttention(nn.Module):
    """
    Self-attention with Rotary Position Embedding for irregular time series.

    Replaces standard positional embedding with RoPE applied to Q and K.

    Args:
        d_model: model dimension
        n_heads: number of attention heads
        dropout: attention dropout rate
        max_freq: maximum frequency for RoPE
        learnable_freq: if True, frequency bases are learnable
    """
    def __init__(self, d_model, n_heads=1, dropout=0.1, max_freq=10000.0, learnable_freq=False):
        super(RoPEAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = ContinuousRoPE(self.head_dim, max_freq=max_freq, learnable=learnable_freq)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, timestamps, attn_mask=None):
        """
        Args:
            x: [B, L, D] - input features
            timestamps: [B, L] or [B, L, 1] - continuous timestamps
            attn_mask: optional attention mask

        Returns:
            out: [B, L, D]
        """
        B, L, D = x.shape
        H = self.n_heads

        q = self.q_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        k = self.k_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k, timestamps)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        return out

# -*- coding: utf-8 -*-
"""
PatchTST adapter for TiWeaver project.
Used with linear interpolation to convert irregular data to regular data,
then apply PatchTST for forecasting.

Interface: forward(batch_x, x_mask, x_mark, y_mask, y_mark) -> outputs [B, pred_len, enc_in]
"""
import torch
import torch.nn as nn
import math


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class FullAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        H = self.n_heads
        q = self.q_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.attention = FullAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Model(nn.Module):
    """
    PatchTST adapter for TiWeaver.
    Performs linear interpolation on irregular input to create regular grid,
    then applies PatchTST forecasting.

    Matches the TiWeaver Model interface:
      - forward(batch_x, x_mask, x_mark, y_mask, y_mark) -> outputs [B, pred_len, enc_in]
      - self.setting: str for checkpoint path
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.seq_len = args.model.seq_len
        self.pred_len = args.model.pred_len
        self.enc_in = args.model.enc_in
        d_model = getattr(args.model, 'd_model', 64)
        d_ff = getattr(args.model, 'd_ff', 128)
        n_heads = getattr(args.model, 'n_heads', 2)
        e_layers = getattr(args.model, 'e_layers', 3)
        dropout = getattr(args.model, 'dropout', 0.1)
        patch_len = getattr(args.model, 'min_patch_size', 16)
        stride = patch_len

        # Patch embedding
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, stride, dropout)

        # Encoder
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ])
        self.encoder_norm = nn.Sequential(
            Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
        )

        # Prediction Head
        self.head_nf = d_model * int((self.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(self.enc_in, self.head_nf, self.pred_len, head_dropout=dropout)

        self.setting = self._get_setting(args)

    def _get_setting(self, args):
        return 'PatchTST_LinearInterp_{}_lr{}_{}_sl{}_pl{}'.format(
            args.data.data_path,
            args.train.lr,
            args.model.loss.criterion,
            args.model.seq_len,
            args.model.pred_len,
        )

    def _linear_interpolate(self, x, x_mask, x_mark, len_seq=None):
        """
        Linearly interpolate irregular time series to a regular grid.
        Args:
            x:      [B, L, C] - values (0 where masked)
            x_mask: [B, L, C] - binary mask (1=observed, 0=missing)
            x_mark: [B, L] - timestamps (normalized)
        Returns:
            x_regular: [B, seq_len, C] - interpolated regular grid
        """
        B, L, C = x.shape
        device = x.device

        # Create regular time grid
        t_regular = torch.linspace(0, 1, len_seq, device=device).unsqueeze(0).expand(B, -1)  # [B, seq_len]

        x_regular = torch.zeros(B, len_seq, C, device=device)

        for b in range(B):
            for c in range(C):
                mask_bc = x_mask[b, :, c].bool()
                if mask_bc.sum() < 2:
                    # Not enough points for interpolation, use mean or zero
                    if mask_bc.sum() == 1:
                        x_regular[b, :, c] = x[b, mask_bc, c].item()
                    continue

                t_obs = x_mark[b, mask_bc]  # observed timestamps
                v_obs = x[b, mask_bc, c]    # observed values

                # Sort by time
                sort_idx = torch.argsort(t_obs)
                t_obs = t_obs[sort_idx]
                v_obs = v_obs[sort_idx]

                # Linear interpolation using searchsorted
                t_query = t_regular[b]
                idx = torch.searchsorted(t_obs, t_query).clamp(1, len(t_obs) - 1)
                t0 = t_obs[idx - 1]
                t1 = t_obs[idx]
                v0 = v_obs[idx - 1]
                v1 = v_obs[idx]

                dt = (t1 - t0).clamp(min=1e-8)
                w = (t_query - t0) / dt
                w = w.clamp(0, 1)
                x_regular[b, :, c] = v0 + w * (v1 - v0)

        return x_regular

    def forward(self, batch_x, x_mask=None, x_mark=None, y_mask=None, y_mark=None):
        """
        Args:
            batch_x: [B, seq_len, enc_in]
            x_mask:  [B, seq_len, enc_in]
            x_mark:  [B, seq_len]
            y_mask, y_mark: unused
        Returns:
            outputs: [B, pred_len, enc_in]
        """
        # Step 1: Linear interpolation to regular grid
        if x_mask is not None and x_mark is not None:
            x = self._linear_interpolate(batch_x, x_mask, x_mark, self.seq_len)
        else:
            x = batch_x

        # Step 2: Instance normalization
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x /= stdev

        # Step 3: PatchTST encoding
        x = x.permute(0, 2, 1)  # [B, C, L]
        enc_out, n_vars = self.patch_embedding(x)  # [B*C, num_patches, d_model]

        for layer in self.encoder:
            enc_out = layer(enc_out)
        enc_out = self.encoder_norm(enc_out)

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, C, d_model, num_patches]

        # Step 4: Prediction
        dec_out = self.head(enc_out)  # [B, C, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, C]

        # De-normalization
        # dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        # dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

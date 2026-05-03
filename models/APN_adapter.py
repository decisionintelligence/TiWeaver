# -*- coding: utf-8 -*-
"""
APN (Attention Patch Network) adapter for TiWeaver project.
Wraps the APN model to match the TiWeaver Model interface.

Original: others/APN/models/APN.py
Interface: forward(batch_x, x_mask, x_mark, y_mask, y_mark) -> outputs [B, pred_len, enc_in]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class AttentionPatchAggregation(nn.Module):
    def __init__(self, N, P, S, te_dim, hid_dim, history, dropout_rate=0.1):
        super().__init__()
        self.N = N
        self.P = P
        self.S = max(history / P, 1e-6) if S is None else S
        self.history = history
        self.hid_dim = hid_dim
        self.te_dim = te_dim
        self.feature_dim = 1 + te_dim
        self.delta_left_params = nn.Parameter(torch.zeros(N, P))
        self.raw_log_width_params = nn.Parameter(torch.full((N, P), math.log(self.S)))
        self.tau_params = nn.Parameter(torch.zeros(N))
        self.projection_layer = nn.Linear(self.feature_dim, self.hid_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim * 2, hid_dim)
        )
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, t_stacked, x_with_te, mask_stacked):
        current_device = t_stacked.device
        B_N, L_obs_pad, _ = t_stacked.shape
        B = B_N // self.N
        patch_centers = torch.linspace(self.S / 2, self.history - self.S / 2, self.P, device=current_device)
        base_left_boundaries = (patch_centers - self.S / 2).unsqueeze(0)
        t_left_n_p = base_left_boundaries + self.delta_left_params
        width_learned_n_p = torch.exp(self.raw_log_width_params) + 1e-6
        t_right_n_p = t_left_n_p + width_learned_n_p
        current_variable_taus = F.softplus(self.tau_params).unsqueeze(-1) + 1e-6
        t_left_b_n = t_left_n_p.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        t_right_b_n = t_right_n_p.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        taus_b_n = current_variable_taus.unsqueeze(0).expand(B, -1, -1).reshape(B_N, 1).unsqueeze(-1)
        t_raw_b_n = t_stacked.transpose(-1, -2)
        weights_raw = torch.sigmoid((t_right_b_n - t_raw_b_n) / taus_b_n) * \
                      torch.sigmoid((t_raw_b_n - t_left_b_n) / taus_b_n)
        mask_b_n = mask_stacked.transpose(-1, -2)
        temporal_weights = weights_raw * mask_b_n
        sum_weights = temporal_weights.sum(dim=-1, keepdim=True) + 1e-9
        weighted_features_sum = torch.bmm(temporal_weights, x_with_te)
        h_patches_avg = weighted_features_sum / sum_weights
        h_patches_proj = self.projection_layer(h_patches_avg)
        h_patches = self.norm(h_patches_proj + self.ffn(h_patches_proj))
        return h_patches


class IMTS_SubModel(nn.Module):
    def __init__(self, enc_in, d_model, pred_len, te_dim=16, npatch=8, dropout=0.1):
        super(IMTS_SubModel, self).__init__()
        self.hid_dim = d_model
        self.te_dim = te_dim
        self.N = enc_in
        self.P = npatch
        self.dropout_rate = dropout
        self.batch_size = None

        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, self.te_dim - 1)

        self.patching = AttentionPatchAggregation(
            N=self.N, P=self.P, S=None,
            te_dim=self.te_dim, hid_dim=self.hid_dim,
            history=1.0, dropout_rate=self.dropout_rate
        )

        self.patch_pos_enc = PositionalEncoding(self.hid_dim, max_len=self.P)
        self.var_queries = nn.Parameter(torch.randn(1, self.N, 1, self.hid_dim))
        self.aggregation_norm = nn.LayerNorm(self.hid_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.hid_dim + self.te_dim, self.hid_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hid_dim * 2, 1)
        )

    def LearnableTE(self, tt):
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def forward(self, x, x_mark, x_mask, y_mark):
        B, L_obs, N_vars = x.shape
        self.batch_size = B

        time_features = x_mark[:, :, [0]]
        X_stacked = x.permute(0, 2, 1).reshape(B * N_vars, L_obs, 1)
        mask_stacked = x_mask.permute(0, 2, 1).reshape(B * N_vars, L_obs, 1)
        time_features_stacked = time_features.repeat(1, 1, N_vars).permute(0, 2, 1).reshape(B * N_vars, L_obs, 1)

        te_his = self.LearnableTE(time_features_stacked)
        X_with_te = torch.cat([X_stacked, te_his], dim=-1)

        # Patching
        h_patches_stacked = self.patching(time_features_stacked, X_with_te, mask_stacked)
        h_patches_stacked_pe = self.patch_pos_enc(h_patches_stacked)
        h_patches_updated = h_patches_stacked_pe.view(B, N_vars, self.P, self.hid_dim)
        attn_scores = torch.matmul(self.var_queries, h_patches_updated.transpose(-1, -2)) * (self.hid_dim ** -0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        h_final = torch.matmul(attn_weights, h_patches_updated).squeeze(-2)
        h_final = self.aggregation_norm(h_final)

        # Decode
        time_steps_to_predict = y_mark[:, :, [0]]
        L_pred = time_steps_to_predict.shape[1]
        h_expanded = h_final.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)
        time_steps_to_predict_exp = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N_vars, 1, 1)
        te_pred = self.LearnableTE(time_steps_to_predict_exp)
        decoder_input = torch.cat([h_expanded, te_pred], dim=-1)
        outputs_raw = self.decoder(decoder_input)
        outputs = outputs_raw.squeeze(-1).permute(0, 2, 1)  # [B, L_pred, N_vars]
        return outputs


class Model(nn.Module):
    """
    APN adapter for TiWeaver.
    Matches the TiWeaver Model interface:
      - forward(batch_x, x_mask, x_mark, y_mask, y_mark) -> outputs [B, pred_len, enc_in]
      - self.setting: str for checkpoint path
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.pred_len = args.model.pred_len
        self.enc_in = args.model.enc_in

        # APN hyperparams with defaults
        d_model = getattr(args.model, 'd_model', 64)
        te_dim = getattr(args.model, 'apn_te_dim', 16)
        npatch = getattr(args.model, 'apn_npatch', 8)
        dropout = getattr(args.model, 'dropout', 0.1)

        self.model = IMTS_SubModel(
            enc_in=self.enc_in,
            d_model=d_model,
            pred_len=self.pred_len,
            te_dim=te_dim,
            npatch=npatch,
            dropout=dropout,
        )

        self.setting = self._get_setting(args)

    def _get_setting(self, args):
        return 'APN_{}_lr{}_dm{}_{}_sl{}_pl{}'.format(
            args.data.data_path,
            args.train.lr,
            args.model.d_model,
            args.model.loss.criterion,
            args.model.seq_len,
            args.model.pred_len,
        )

    def forward(self, batch_x, x_mask=None, x_mark=None, y_mask=None, y_mark=None):
        """
        Args:
            batch_x: [B, seq_len, enc_in]
            x_mask:  [B, seq_len, enc_in]
            x_mark:  [B, seq_len] or [B, seq_len, 1]
            y_mask:  [B, pred_len, enc_in]
            y_mark:  [B, pred_len] or [B, pred_len, 1]
        Returns:
            outputs: [B, pred_len, enc_in]
        """
        # Ensure x_mark and y_mark have shape [B, L, 1]
        if x_mark.dim() == 2:
            x_mark = x_mark.unsqueeze(-1)
        if y_mark.dim() == 2:
            y_mark = y_mark.unsqueeze(-1)

        outputs = self.model(batch_x, x_mark, x_mask, y_mark)
        return outputs

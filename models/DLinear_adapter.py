# -*- coding: utf-8 -*-
"""
DLinear (Decomposition-Linear) adapter for TiWeaver project.
Wraps the LTSF-Linear DLinear model to match the TiWeaver Model interface.

Original: others/LTSF-Linear/models/DLinear.py
Interface: forward(batch_x, x_mask, x_mark, y_mask, y_mark) -> outputs [B, pred_len, enc_in]

Note: DLinear is a regular time series model. For irregular data, NaN values
in batch_x are replaced with 0 before feeding into the model.
"""
import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block."""
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    DLinear adapter for TiWeaver.
    Matches the TiWeaver Model interface:
      - forward(batch_x, x_mask, x_mark, y_mask, y_mark) -> outputs [B, pred_len, enc_in]
      - self.setting: str for checkpoint path
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.seq_len = args.model.seq_len
        self.pred_len = args.model.pred_len
        self.channels = args.model.enc_in

        # Decomposition Kernel Size
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)

        # Channel-independent linear layers
        self.Linear_Seasonal = nn.ModuleList()
        self.Linear_Trend = nn.ModuleList()
        for i in range(self.channels):
            self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
            self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

        self.setting = self._get_setting(args)

    def _get_setting(self, args):
        return 'DLinear_{}_lr{}_{}_sl{}_pl{}'.format(
            args.data.data_path,
            args.train.lr,
            args.model.loss.criterion,
            args.model.seq_len,
            args.model.pred_len,
        )

    def forward(self, batch_x, x_mask=None, x_mark=None, y_mask=None, y_mark=None):
        """
        Args:
            batch_x: [B, seq_len, enc_in] - input (NaN already replaced with 0)
            x_mask, x_mark, y_mask, y_mark: ignored by DLinear
        Returns:
            outputs: [B, pred_len, enc_in]
        """
        # DLinear only uses batch_x
        x = batch_x  # [B, seq_len, C]

        # Decompose
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)  # [B, C, seq_len]
        trend_init = trend_init.permute(0, 2, 1)

        # Channel-independent prediction
        seasonal_output = torch.zeros(
            [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
            dtype=seasonal_init.dtype
        ).to(seasonal_init.device)
        trend_output = torch.zeros(
            [trend_init.size(0), trend_init.size(1), self.pred_len],
            dtype=trend_init.dtype
        ).to(trend_init.device)

        for i in range(self.channels):
            seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
            trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # [B, pred_len, C]

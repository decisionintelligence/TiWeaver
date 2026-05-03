from torch import nn
import torch
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F

from models.layers.Embedding import positional_encoding
from models.layers.RevIN import RevIN
from models.layers.mamba_ssm_test.modules.mamba_simple import Mamba


class Model(nn.Module):
    def __init__(self, args):

        super().__init__()
        self.setting = self.get_setting(args)

        self.features = args.model.input_dim
        self.seq_len = args.model.seq_len
        self.d_model = args.model.d_model
        # Input encoding: projection of feature vectors onto a d-dim vector space
        self.W_P = nn.Linear(self.seq_len, self.d_model)

        # Residual dropout
        self.dropout = nn.Dropout(args.model.dropout)

        self.mamba = nn.ModuleList(
            [Mamba(
                d_model=self.features,
                d_state=16,  # SSM state expansion factor
                d_conv=3,  # Local convolution width
                expand=1,
                use_fast_path=False
            )
                for i in range(args.model.e_layers)
            ]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(self.d_model) for i in range(args.model.e_layers)]
        )

        self.W_pos = positional_encoding('zero', True, self.features, self.d_model)
        self.revin = RevIN(num_features=self.features)
        self.head = PredictionHead(self.d_model, args.model.pred_len, args.model.dropout)

    def get_setting(self, args):
        setting = 'Mamba_{}_{}_data{}_lr{}_loss{}_sl{}_pl{}'.format(
            args.data.data_path,
            args.data.freq,
            args.train.lr,
            f"{int(args.model.loss.pred_loss)}",
            args.model.loss.criterion,
            args.model.seq_len,
            args.model.pred_len,
        )
        return setting

    def forward(self, x) -> Tensor:
        bs, seq_len, features = x.shape
        x = self.revin(x, "norm")

        x = rearrange(x, 'b l d -> b d l')
        x = self.W_P(x)  # x: [bs x features x d_model]
        z = self.dropout(x + self.W_pos)  # z: [bs x features x d_model]
        z = rearrange(z, 'b d l -> b l d')

        # Encoder & residual
        for layer, bn in zip(self.mamba, self.bns):
            z = layer(z) + z  # z: [bs x d_model x features]
            z = bn(z.view(-1, self.d_model)).view(bs, self.d_model, -1)

        z = rearrange(z, 'b l d -> b d l')
        z = self.head(z)
        z = rearrange(z, 'b d l -> b l d')
        z = self.revin(z, 'denorm')

        return z

class PredictionHead(nn.Module):
    def __init__(self, d_model, forecast_len, head_dropout=0, flatten=False):
        super().__init__()
        self.flatten = flatten

        self.linear = nn.Linear(d_model, forecast_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.linear(x)  # x: [bs x features x d_model]
        x = self.dropout(x)

        return x
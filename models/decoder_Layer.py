import math
import torch
import torch.nn as nn
from torch.nn import init
import time
import torch.nn.functional as F
from models.layers.Embedding import *


class Transformer_Decoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        layer_number,
    ):
        super(Transformer_Decoder, self).__init__()

        self.d_model = d_model
        self.layer_number = layer_number

        self.inter_var_attention = nn.MultiheadAttention(d_model, 1, dropout=0.1, batch_first=True)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        ##FFN
        self.d_ff = d_ff
        self.dropout = nn.Dropout(0.1)
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.d_ff, self.d_model, bias=True),
        )

    def forward(self, x): #n_vars * [bs x num_patches' x hidden_dim]
        K = len(x)

        ####inter var Attention#####
        inter_out = []
        for k in range(K):

            var_x = x[k]
            
            other_patches = []
            for j in range(K):
                if j != k:  # 排除当前变量
                    other_patches.append(x[j])
            other_patches = torch.cat(other_patches, dim=1)  # [bs x total_other_patches x hidden_dim]
            
            inter_var_out, _ = self.inter_var_attention(var_x, other_patches, other_patches)  # [b, patch_num, dim]
            inter_var_out = self.norm(self.dropout(inter_var_out) + var_x)
            inter_var_out = torch.mean(inter_var_out, dim=1)  # [b, dim]
            inter_out.append(inter_var_out)
        out = torch.stack(inter_out, dim=0)  # [nvar, b, dim]

        ##FFN
        # out = self.dropout(out)
        out = self.ff(out) + out
            
        return out.permute(1, 0, 2)  # [b, nvar, dim]


import math
import numpy as np
import torch
from torch import nn
from einops import rearrange

from models.layers.graph_layer import Encoder
from models.layers.Transformer_EncDec import Encoder as FormerEncoder, EncoderLayer as FormerEncoderLayer
from models.layers.Transformer_EncDec import Decoder, DecoderLayer
from models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from models import TimeCHEAT, Pathformer


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        # Multiple n_patches for MOE
        # self.n_patches_list = args.model.n_patches_list  # List of different n_patches values
        # self.num_experts = len(self.n_patches_list)
        # self.experts = nn.ModuleList()
        
        # Create patch ranges and ref points for each expert
        # self.patch_ranges = []
        # self.ref_points_list = []
        # self.encoders = nn.ModuleList()
        # self.position_embeddings = nn.ModuleList()
        # self.formers = nn.ModuleList()
        # self.flattens = nn.ModuleList()
        # self.linears = nn.ModuleList()
        
        self.args = args
        
        self.cor_model = TimeCHEAT.Model(args)
        self.main_model = Pathformer.Model(args)
        
        self.main_projection = nn.Linear(args.model.d_model, 1)
        
        self.patch_size = args.model.ref_points // args.model.n_patches
        
        self.embbeding = nn.Linear(self.patch_size, args.model.d_model)
        
        self.output_projection = nn.Linear(args.model.d_model * args.model.n_patches, args.model.pred_len)

        # self.decoder = Decoder(
        #         [
        #             DecoderLayer(
        #                 AttentionLayer(
        #                     FullAttention(True, args.model.former_factor, attention_dropout=args.model.dropout,
        #                                 output_attention=False),
        #                     args.model.d_model, args.model.attn_head),
        #                 AttentionLayer(
        #                     FullAttention(False, args.model.former_factor, attention_dropout=args.model.dropout,
        #                                 output_attention=False),
        #                     args.model.d_model, args.model.attn_head),
        #                 args.model.d_model,
        #                 args.model.d_ff,
        #                 dropout=args.model.dropout,
        #                 activation=args.model.former_activation,
        #             )
        #             for l in range(1)
        #         ],
        #         norm_layer=torch.nn.LayerNorm(args.model.d_model),
        #         projection=nn.Linear(args.model.d_model, args.model.d_model, bias=True)
        #     )
        self.patch_attention = nn.MultiheadAttention(args.model.d_model, 1, dropout=0.1, batch_first=True)

        self.setting = self.get_setting(args)
        
    def get_setting(self, args):
        setting = 'pathCHEAT_{}_{}_{}_data{}_lr{}_loss{}_sl{}_pl{}_patch{}_rp{}'.format(
            args.des,
            args.data.data_path,
            args.data.freq,
            args.train.lr,
            f"{int(args.model.loss.pred_loss)}",
            args.model.loss.criterion,
            args.model.seq_len,
            args.model.pred_len,
            args.model.n_patches,
            args.model.ref_points
        )
        return setting

    def forward(self, batch_x, x_mark, x_t): # (batch x input_len x var_num x 2+1) var_num x 2+1: [input,mark,time]
        
        # experts_out = [
        #     self.experts[i](batch_x, x_mark, x_t) for i in range(len(self.n_patches_list))
        # ]
        # out = torch.stack(experts_out, dim=-1).mean(dim=-1) # [batch x output_len x 1]
        # Project to output dimension
        # out = self.output_projection(out) # [batch x output_len]
        # out = out.unsqueeze(-1) # [batch x output_len x 1]
        main_x = batch_x[:, :, -3:]
        main_out = self.main_model(main_x) # [batch x seq x main_var x d]
        main_out = self.main_projection(main_out) # [batch x seq x main_var x 1]
        main_out = self.main_model.revin_layer(main_out.squeeze(-1), "denorm") # [batch x seq x main_var x 1]

        main_out = torch.cat([batch_x[:, :, :-3], main_out], dim=-1) # [batch x seq x main_var x 2+1]
        main_out = main_out.permute(0, 2, 1) # [batch x main_var x seq]
        main_out = main_out.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)  # [b x patch_num x nvar x dim x patch_len]
        cor_out = self.cor_model(batch_x, x_mark, x_t) # [batch x var x patch_num x patch_len]
        
        cor = self.embbeding(cor_out)
        main = self.embbeding(main_out)
        cor = rearrange(cor, 'b k n d -> (b k) n d')
        main = rearrange(main, 'b k n d -> (b k) n d')
        # out = torch.cat([main_out, cor_out[:, -3:, :]], dim=-1)
        
        
        # out = self.decoder(main, cor)
        
        out,_ = self.patch_attention(main, cor, cor)
        
        
        # out = torch.cat([main_out, cor_out[:, -3:, :]], dim=-1)
        out = rearrange(out, '(b k) n d -> b k (n d)', b=batch_x.shape[0], k=batch_x.shape[-1])
        out = self.output_projection(out).permute(0, 2, 1) # [batch x 1 x output_len]
        # out = torch.cat([main_out, cor_out[:, -3:, :]], dim=-1)
        return out[:, :, -3:]

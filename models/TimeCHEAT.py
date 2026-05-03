import math
import numpy as np
import torch
from torch import nn

from models.layers.graph_layer import Encoder
from models.layers.Transformer_EncDec import Encoder as FormerEncoder, EncoderLayer as FormerEncoderLayer
from models.layers.SelfAttention_Family import FullAttention, AttentionLayer


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
        self.config = args
        self.dim = args.model.channels
        self.ath = args.model.attn_head
        self.latent_dim = args.model.latent_dim
        self.n_layers = args.model.n_layers

        '''
        获取输入数据的每个patch的索引，以time为索引，而TimeCHEAT的time为0-1的浮点数
        args.config['obs'] 为当前时间点，因为TimeCHEAT是以观测历史多久的时间长度为输入，
        比如历史12个小时，则它的输入数据是12个小时内全部的时间点，长度不固定
        而这里输入长度固定已知，且最完整的直接是 seq_len/(seq_len+pred_len)，
        即没有数据点缺失的情况（大多数，除非和TimeCHEAT一样，将全NAN的删除，时间会不连续），
        但需要讨论似乎不如上述合理。
        '''
        self.obs = args.model.seq_len / (args.model.seq_len + args.model.pred_len)
        self.n_patches = args.model.n_patches
        self.register_buffer('patch_range', torch.linspace(0, 1 * self.obs, self.n_patches + 1)) # (n_patches + 1,)
        assert args.model.ref_points % self.n_patches == 0
        # 获取每个patch的参考点，也就是虚拟时间点吧，可能一个patch原始有m个时间点，而全部的ref_points有n个，那么每个patch的参考点就是m/n个
        self.register_buffer('ref_points', torch.linspace(0, 1 * self.obs, args.model.ref_points)) # (ref_points,)
        self.ref_points = self.ref_points.reshape(self.n_patches, -1) # (n_patches, ref_points/n_patches)

        # graph patch
        self.encoder = Encoder(dim=self.dim, attn_head=self.ath, n_patches=self.n_patches, nkernel=self.latent_dim, n_layers=self.n_layers)
        self.position_embedding = PositionalEmbedding(self.ref_points.size(-1))
        self.dropout = nn.Dropout(args.model.dropout)

        '''
        # transformer
        self.former = FormerEncoder(
            [
                FormerEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.model.former_factor, attention_dropout=args.model.dropout,
                                      output_attention=args.model.former_output_attention), self.ref_points.size(-1), args.model.former_heads),
                    self.ref_points.size(-1),
                    args.model.former_dff,
                    dropout=args.model.dropout,
                    activation=args.model.former_activation
                ) for _ in range(args.model.former_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.ref_points.size(-1))
        )
        '''
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(self.n_patches * self.ref_points.size(-1), args.model.pred_len)
        self.dropout = nn.Dropout(args.model.dropout)

        self.setting = self.get_setting(args)
        
    def get_setting(self, args):
        setting = 'TimeCHEAT_{}_{}_data{}_lr{}_loss{}_sl{}_pl{}_patch{}_rp{}'.format(
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

    def _split_patch(self, data, mask, time, i_patch):
        # data: (batch_size=32, seq_len=512, dim=7)
        # mask: (32, 512, 7)
        # time: (32, 512)
        # get boundaries of this patch and its ref points
        start = self.patch_range[i_patch]                                           # scalar
        end = self.patch_range[i_patch + 1]                                         # scalar
        ref_points = self.ref_points[i_patch].to(data.device)                      # ref_points: (n_ref_points,)
        # select time‐steps in [start, end] where at least one feature is observed
        time_mask = torch.logical_and(torch.logical_and(time >= start, time <= end),
                                      mask.sum(-1) > 0)                             # time_mask: (32, 512), bool
        num_observed = time_mask.sum(1).long()                                      # num_observed: (32,)
        n_ref_points = ref_points.size(0)                                           # scalar
        max_obs = num_observed.max().item()                                         # scalar
        # initialize patch arrays with room for observed + ref points
        patch = torch.zeros(data.size(0), max_obs + n_ref_points, self.dim,
                            device=data.device)                                     # patch: (32, max_obs+n_ref, 7)
        patch_mask = torch.zeros_like(patch)                                        # patch_mask: (32, max_obs+n_ref, 7)
        patch_time = torch.zeros(patch.size(0), patch.size(1), device=data.device)  # patch_time: (32, max_obs+n_ref)
        rp_mask = torch.zeros_like(patch_mask)                                      # rp_mask: (32, max_obs+n_ref, 7)
        indices = torch.arange(patch.size(1), device=data.device)                   # indices: (max_obs+n_ref,)
        # fill per‐sample observed data, then interleave ref points
        for i in range(data.size(0)):  # iterate batch (32)
            obs_i = num_observed[i].item()                                        # scalar for sample i
            # copy observed mask/data/time
            patch_mask[i, :obs_i, :] = mask[i, time_mask[i]]                     # (obs_i,7)
            patch[i, :obs_i, :]      = data[i, time_mask[i]]                     # (obs_i,7)
            patch_time[i, :obs_i]    = time[i, time_mask[i]]                     # (obs_i,)
            # append ref points in time
            patch_time[i, obs_i: obs_i + n_ref_points] = ref_points              # (n_ref_points,)
            # sort the combined observed+ref by time
            sorted_idx_obs = torch.argsort(patch_time[i, :obs_i + n_ref_points]) # (obs_i+n_ref_points,)
            sorted_index = torch.cat([sorted_idx_obs,
                                      indices[obs_i + n_ref_points:]])           # (max_obs+n_ref,)
            # reorder data, mask, time to merged order
            patch_mask[i] = patch_mask[i, sorted_index]                          # (max_obs+n_ref,7)
            patch[i]      = patch[i, sorted_index]                               # (max_obs+n_ref,7)
            patch_time[i] = patch_time[i, sorted_index]                          # (max_obs+n_ref,)
            # mark which positions are ref points
            rp_mask[i, sorted_index[obs_i: obs_i + n_ref_points]] = 1.           # (max_obs+n_ref,7)
        # outputs:
        #   patch:      (32, max_obs+n_ref, 7)
        #   patch_mask: (32, max_obs+n_ref, 7)
        #   patch_time: (32, max_obs+n_ref)
        #   rp_mask:    (32, max_obs+n_ref, 7)  
        return patch, patch_mask, patch_time, rp_mask
    # End of Selectio

    def embedding(self, vals, mask, time):
        # vals, mask, time = data[..., :self.dim], data[..., self.dim:-1], data[..., -1]

        # encoder
        repr_patch = []
        for i_patch in range(self.n_patches):
            v, m, t, rp_m = self._split_patch(vals, mask, time, i_patch)

            v = v * m
            context_mask = m + rp_m
            # out -> n_patch * B * {t_i} * channel * latent_dim
            repr, repr_mask, _, _ = self.encoder(t, v, context_mask, rp_m, i_patch)
            # repr_patch.append(repr[repr_mask.sum(-1) > 0, ...].reshape(repr.size(0), -1, self.dim).unsqueeze(1))
            repr_patch.append(repr[repr_mask == 1].reshape(repr.size(0), -1, self.dim).unsqueeze(1))
        
        # combined, [batch x patch_num x patch_len x dim] => 
        repr_patch = torch.cat(repr_patch, dim=1).contiguous().permute(0, 3, 1, 2) # [batch x dim x patch_num x patch_len]

        return repr_patch

        # positional embedding
        repr_patch = torch.reshape(repr_patch, (repr_patch.shape[0] * repr_patch.shape[1], repr_patch.shape[2], repr_patch.shape[3]))
        repr_patch += self.position_embedding(repr_patch)
        repr_patch = self.dropout(repr_patch)
        
        # transformer encode
        embedding, _ = self.former(repr_patch)
        
        embedding = torch.reshape(embedding, (-1, self.dim, embedding.shape[-2], embedding.shape[-1])).permute(0, 1, 3, 2)
        return embedding # [batch x dim x patch_len x patch_num]

    def forward(self, batch_x, x_mark, x_t): # (batch x input_len x var_num x 2+1) var_num x 2+1: [input,mark,time]
        
        embedding = self.embedding(batch_x, x_mark, x_t) # [batch x dim x patch_len x patch_num]
        return embedding
        # out = self.dropout(self.linear(self.flatten(embedding))) # [batch x dim x output_len]
        # out = out.permute(0, 2, 1) # [batch x output_len x dim]
        # if self.config.model_name == 'pathCHEAT':
        out = self.flatten(embedding)

        return out

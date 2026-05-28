import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math

from models.layers.Mask_RevIN import MaskedRevIN
from models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.layers.Transformer_EncDec import Encoder, EncoderLayer
from models.layers.Embed import PositionalEmbedding
from models.layers.PatchGAT import IntraPatchGATEncoder
from models.layers.mask_attention import MaskedVariableCrossAttention


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.threshold = args.model.threshold
        self.pred_len = args.model.pred_len
        self.seq_len = args.model.seq_len
        self.seq_len_max_irr = args.model.seq_len_max_irr
        self.pred_len_max_irr = args.model.pred_len_max_irr
        self.latent_dim = int(args.model.dim.latent)
        self.input_dim = int(args.model.enc_in)
        self.output_dim = int(args.model.enc_in)
        self.d_model = args.model.d_model
        self.min_patch_size = args.model.min_patch_size
        self.stride = args.model.min_patch_size
        
        self.similarity_func = args.model.similarity_func
        
        self.revin_layer = MaskedRevIN(args.model.enc_in, affine=True, subtract_last=False)
        
        self.hidden_layer = nn.Linear(1, self.d_model)
        
        self.var_atten = MaskedVariableCrossAttention(hidden_dim=self.d_model*1, num_heads=args.model.n_heads)

        self.position_embedding = PositionalEmbedding(self.d_model*1)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            args.model.factor,
                            attention_dropout=args.model.dropout,
                            output_attention=args.model.output_attention,
                        ),
                        args.model.d_model*1,
                        1,
                    ),
                    args.model.d_model*1,
                    args.model.d_ff,
                    dropout=args.model.dropout,
                    activation=args.model.activation,
                )
                for l in range(args.model.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.model.d_model*1),
        )
        
        self.intra_patch_graph = IntraPatchGATEncoder(in_dim=self.d_model * 1, hidden_dim=self.d_model * 1)
        
        # Prediction Head
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, self.d_model - 1)
        self.decoder = nn.Sequential(
                nn.Linear(self.d_model * 1, self.d_model*2),
                nn.ReLU(inplace=True),
                nn.Linear(self.d_model*2, self.d_model),
                nn.ReLU(inplace=True),
                nn.Linear(self.d_model, 1)
            )

        self.setting = self.get_setting(args)
        
    def get_setting(self, args):
        setting = 'TiWeaver-{}'.format(
            args.data.data_path,
        )
        return setting
    
    def LearnableTE(self, tt):
        # learnable continuous time embeddings
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)


    def merge_patches_parallel(self, patch_inputs, BOARDERS):
        B, N, D = patch_inputs.shape
        device = patch_inputs.device

        group_ids = torch.full((B, N), -1, dtype=torch.long, device=device)
        merge_boundaries = BOARDERS[:, 1::2] == 1  # [B, N-1]
        cumsum = torch.cumsum(~merge_boundaries, dim=1)  # [B, N-1]
        group_ids[:, 0] = 0 
        group_ids[:, 1:] = cumsum

        new_BOARDERS = torch.zeros_like(BOARDERS)
        new_BOARDERS[:, ::2] = -1
        new_BOARDERS[:, 1::2] = 0
        
        patch_weights = BOARDERS[:, ::2].clamp(min=1) 
        group_sizes = torch.zeros_like(group_ids, dtype=torch.int)

        group_sizes.scatter_add_(
            dim=1,
            index=group_ids,
            src=patch_weights
        )

        new_patch_inputs = torch.zeros_like(patch_inputs)
        
        for b in range(B):

            _, counts = torch.unique(group_ids[b], return_counts=True)
            split_sizes = counts*D
            splits = torch.split(patch_inputs[b].reshape(-1), split_sizes.tolist())
            
            downsampled_splits = []
            for split, factor in zip(splits, counts.tolist()):
                if factor == 1:
                    downsampled_splits.append(split)
                else:
                    new_length = split.size(0) // factor
                    indices = torch.arange(0, split.size(0), factor)[:new_length]
                    downsampled_splits.append(split[indices])
            
            merge_inputs = torch.cat(downsampled_splits).reshape(-1, D)
            new_patch_inputs[b, :len(merge_inputs), :] = merge_inputs
        
            num_complex = (BOARDERS[b, ::2] < 0).sum().item()
            id_groups = group_ids[b].unique() if num_complex == 0 else group_ids[b].unique()[:-num_complex] 
            new_BOARDERS[b, id_groups*2] = group_sizes[b, id_groups]
        
        return new_patch_inputs, new_BOARDERS

    def forward(self, x, x_mark, y_mark, x_mask=None, y_mask=None, batch=None):
        # x: [batch_size, seq_len, n_vars]
        B, T, K = x.shape

        # Normalization from Non-stationary Transformer
        x = self.revin_layer(x, x_mask, 'norm')
        
        if T < self.min_patch_size:
            padding = torch.zeros([B, self.min_patch_size - T, K]).to(x.device)
            x = torch.cat((x, padding), dim=1)
        x = rearrange(x, 'b t K -> (b K) t')
        x_mask = rearrange(x_mask, 'b t K -> (b K) t')

        patch_size = self.min_patch_size
        marge_patch_size = self.min_patch_size * 2
        max_patch_num = (T - self.min_patch_size) // self.stride + 1
        BOARDERS = torch.ones([B*K, 2*max_patch_num-1], dtype=torch.int).to(x.device)

        if x_mask is not None:
            
            sort_idx = torch.argsort(-x_mask, dim=1, stable=True)  # [B*K, T]
            x = torch.gather(x, 1, sort_idx)
            x_mask = torch.gather(x_mask, 1, sort_idx)
            
            # Expand x_mark to [B, K, T] to match inputs [B*K, T]
            x_mark_expanded = x_mark.unsqueeze(1).expand(-1, K, -1).reshape(B*K, T)
            x_mark = torch.gather(x_mark_expanded, 1, sort_idx)

            patch_masks = x_mask.unfold(1, self.min_patch_size, self.stride)  # [B*K, num_patches, min_patch_size]
            patch_masks = patch_masks.any(dim=-1).int()  # [B*K, num_patches]
            BOARDERS[:, ::2] = patch_masks
            
            
        x_mark_diff = torch.abs(x_mark[:, 1:] - x_mark[:, :-1])  # [B, T-1, K]
        if x_mask is not None:
            x_mask_bool = x_mask.bool()
            valid_pairs = x_mask_bool[:, 1:] & x_mask_bool[:, :-1]
            if valid_pairs.any():
                x_mark_max_diff = x_mark_diff[valid_pairs].max().item()
            else:
                x_mark_max_diff = x_mark_diff.max().item()
        else:
            x_mark_max_diff = x_mark_diff.max().item()
            
        x_emb = self.hidden_layer(x.unsqueeze(-1))  # [B*K, T, D]
        x_mark_emb = self.LearnableTE(x_mark.unsqueeze(-1))  # [B*K, T, D]
        inputs_emb = x_emb + x_mark_emb  # [B*K, T, D+D]

        marge_patch_inputs_emb = None
        while True:
            
            patch_inputs = inputs_emb.unfold(dimension=1, size=patch_size, step=self.stride).permute(0, 1, 3, 2)  # inputs: [B*K x num_patch x min_patch_size x 2D]
            patch_inputs_emb = self.intra_patch_graph(patch_inputs)  # [B*K x num_patch x 2D]

            patch_x_mark = x_mark.unfold(dimension=1, size=patch_size, step=self.stride)  # inputs: [bs x num_patch x n_vars x min_patch_size]
            patch_x_mark_diff = patch_x_mark[:,1:,0]-patch_x_mark[:,:-1,-1]
            patch_x_mark_exp = torch.exp(-patch_x_mark_diff / x_mark_max_diff + 1e-8)  # [B*K, num_patches-1]
            attns = F.softmax(torch.matmul(patch_inputs_emb, patch_inputs_emb.transpose(-2, -1)) / math.sqrt(patch_inputs_emb.shape[-1]), dim=-1)
            sim_mask = (attns[:,0,1:]*patch_x_mark_exp) < self.threshold  # [B*K, num_boarders] 判断相邻patch相似度是否小于阈值
            BOARDERS[:,1::2] = ~sim_mask
            
            new_patch_inputs = inputs_emb.unfold(dimension=1, size=marge_patch_size, step=self.stride).permute(0, 1, 3, 2)
            new_patch_inputs_emb = self.intra_patch_graph(new_patch_inputs)  # [B*K x num_patch x 2D]
            
            new_attns = F.softmax(torch.matmul(new_patch_inputs_emb, patch_inputs_emb.transpose(-2, -1)) / math.sqrt(patch_inputs_emb.shape[-1]), dim=-1)
            new_sim_mask = (new_attns[:, :, :-1].diagonal(dim1=-2, dim2=-1) < self.threshold) | (new_attns[:, :, 1:].diagonal(dim1=-2, dim2=-1) < self.threshold)  # [B*K, num_patches-1]
            BOARDERS[:,1::2] = BOARDERS[:,1::2] & ~new_sim_mask
            
            if not BOARDERS[:,1::2].any():
                if marge_patch_inputs_emb is None:
                    marge_patch_inputs_emb = patch_inputs
                break
            else:
                patch_inputs = patch_inputs.reshape(B*K, patch_inputs.shape[1], -1)
                marge_patch_inputs_emb, BOARDERS = self.merge_patches_parallel(patch_inputs, BOARDERS)
                inputs_emb = marge_patch_inputs_emb.reshape(B*K, -1, inputs_emb.shape[-1])
                
                
        marge_patch_inputs_emb = rearrange(patch_inputs_emb, '(b k) n d -> b k n d', b=B, k=K, n=patch_inputs_emb.shape[1])  # [bs x n_vars x num_patches x hidden_dim]
        merge_patch_inputs_mask = BOARDERS[:, ::2] > 0
        marge_patch_mask = rearrange(merge_patch_inputs_mask, '(b k) n -> b k n', b=B, k=K)  # [bs x n_vars x num_patches]
        
        var_attn_out = self.var_atten(marge_patch_inputs_emb, marge_patch_mask)
        
        input_emb = rearrange(var_attn_out, 'b k n d -> (b k) n d')
        
        # Encoder
        encoder_mask = marge_patch_mask.reshape(B*K, -1).unsqueeze(1).unsqueeze(1).repeat(1, 1, marge_patch_mask.shape[-1], 1)
        position_emb = self.position_embedding(input_emb)
        emb = input_emb + position_emb
        enc_out, _ = self.encoder(emb, attn_mask=encoder_mask)  # [bs*n_vars x num_patch x hidden_dim]
        enc_out = rearrange(enc_out, '(b k) n d -> b k n d', b=B, k=K, n=enc_out.shape[1])

        # Get the last valid patch index for each variable
        last_valid_idx = marge_patch_mask.sum(dim=-1) - 1  # [bs x n_vars]
        
        batch_idx = torch.arange(B, device=enc_out.device).view(-1, 1, 1).expand(-1, K, 1)  # [bs x n_vars x 1]
        var_idx = torch.arange(K, device=enc_out.device).view(1, -1, 1).expand(B, -1, 1)    # [bs x n_vars x 1]
        patch_idx = last_valid_idx.unsqueeze(-1)  # [bs x n_vars x 1]
        
        Z0 = enc_out[batch_idx, var_idx, patch_idx]  # [bs x n_vars x 1 x hidden_dim]
        Z0 = Z0.squeeze(2)  # [bs x n_vars x hidden_dim]
        L_pred = y_mark.shape[-1]
        y_mark = y_mark.view(B, 1, L_pred, 1).repeat(1, K, 1, 1)
        h = Z0.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)

        te_pred = self.LearnableTE(y_mark) # [bs x nvars x L_pred x D]
        h = h + te_pred
        
        pred_y = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1)
        pred_y = self.revin_layer(pred_y, y_mask, 'denorm')
        
        return pred_y
    
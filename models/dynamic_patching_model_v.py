import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import gc

from models.layers.RevIN import RevIN
from models.layers.Mask_RevIN import MaskedRevIN
from models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.layers.Transformer_EncDec import Encoder, EncoderLayer
from models.layers.Embed import PositionalEmbedding
# from models.decoder_Layer import Transformer_Decoder
from models.layers.Diffeq_solver import DiffeqSolver
from models.layers.Decoder import Linear_Decoder
from models.layers.Unk_Dynamics import Unk_odefunc
from models.layers.tools import EncoderAttrs
from models.dynamic_encoder_layer import DynamicEncoderLayer

from utils.utils import ConfigDict, load_config


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


def FFT_for_Period(x, k=16):
    # [bs x seq x var]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period

def temporal_to_frequency(patch, k=16):
    # [bs x patch_num x n_vars x patch_size]
    fft = torch.fft.rfft(patch, dim=-1)
    # Get the magnitude of FFT coefficients
    fft_magnitude = torch.abs(fft)
    # Find top-k frequencies for each patch
    _, topk_indices = torch.topk(fft_magnitude, k, dim=-1)
    # Gather the corresponding FFT coefficients
    fft_k = torch.gather(fft, -1, topk_indices)
    fft_vec = torch.cat([fft_k.real, fft_k.imag], dim=-1)  # [bs x n_vars x 2k]
    return fft_vec, topk_indices

def frequency_to_temporal(fft_vec, topk_indices, original_patch_size, k=16):
    # 分离实部和虚部
    real_part = fft_vec[..., :k]
    imag_part = fft_vec[..., k:]
    fft_k = torch.complex(real_part, imag_part)
    
    # 创建完整的FFT系数数组
    full_fft = torch.zeros(fft_vec.shape[:-1] + (original_patch_size // 2 + 1,),
                         dtype=torch.complex64, device=fft_vec.device)
    
    # 使用保存的索引将系数放回正确位置
    full_fft.scatter_(-1, topk_indices, fft_k)
    
    # 执行逆FFT
    reconstructed_patch = torch.fft.irfft(full_fft, n=original_patch_size, dim=-1)
    
    return reconstructed_patch
    
class MaskedVariableCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Projection layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask):
        """
        Args:
            x:    [B, V, P, D]  - 输入特征（变量维度）
            mask: [B, V, P]     - patch-level 有效性 mask（1=有效，0=无效）
        Returns:
            out:  [B, V, P, D]  - 经过 cross attention 后的输出
        """
        B, V, P, D = x.shape
        H = self.num_heads
        d = self.head_dim

        # Project to Q, K, V
        q = self.q_proj(x).view(B, V, P, H, d).transpose(2, 3)  # [B, V, H, P, d]
        k = self.k_proj(x).view(B, V, P, H, d).transpose(2, 3)
        v = self.v_proj(x).view(B, V, P, H, d).transpose(2, 3)

        # Cross: 对每个变量 i，与其他变量 j 的 k,v 交叉注意力
        q = q.unsqueeze(2)  # [B, Vq, 1, H, P, d]
        k = k.unsqueeze(1)  # [B, 1, Vk, H, P, d]
        v = v.unsqueeze(1)  # [B, 1, Vk, H, P, d]

        # Compute attention scores
        # attn_logits = torch.einsum("bvqhpd,bvkhqd->bvqkhpq", q, k)  # [B, Vq, Vk, H, Pq, Pk]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)

        # Patch-level key mask
        key_mask = mask.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [B, 1, Vk, 1, 1, Pk]
        attn_logits = attn_logits.masked_fill(~key_mask, float(0))

        # Exclude self-variable attention
        self_mask =  torch.eye(V, device=x.device).bool().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [1, Vq, Vk]
        # self_mask = self_mask.unsqueeze(3).unsqueeze(4).unsqueeze(5)               # [1, Vq, Vk, 1, 1, 1]
        attn_logits = attn_logits.masked_fill(self_mask, float(0))

        # Softmax
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, Vq, Vk, H, Pq, Pk]
        
        # del attn_logits
        # gc.collect()
        # torch.cuda.empty_cache()

        # Weighted sum over V
        # attn_out = torch.einsum("bvqkhij,bvkhjpd->bvqhipd", attn_weights, v)  # [B, Vq, Vk, H, Pq, d]
        attn_out = torch.matmul(attn_weights, v)
        
        # del attn_weights
        # gc.collect()
        # torch.cuda.empty_cache()
        
        # Sum over Vk (other variables)
        attn_out = attn_out.sum(dim=2)  # [B, Vq, H, Pq, Dh]

        # Restore shape
        attn_out = attn_out.permute(0, 1, 3, 2, 4).contiguous()  # [B, V, P, H, Dh]
        attn_out = attn_out.view(B, V, P, D)  # [B, V, P, D]
        
        out = self.out_proj(attn_out)  # [B, V, P, D]
        
        return out

class Model(nn.Module, EncoderAttrs):
    def __init__(self, args):
        super(Model, self).__init__()
        # EncoderAttrs.__init__(self, args)
        
        self.threshold = args.model.threshold
        self.task_name = args.exp_name
        
        self.revin = args.model.revin
        self.pred_len = args.model.pred_len
        self.seq_len = args.model.seq_len
        self.latent_dim = int(args.model.dim.latent)
        self.input_dim = int(args.model.enc_in)
        self.output_dim = int(args.model.enc_in)
        self.d_model = args.model.d_model
        self.min_patch_size = args.model.min_patch_size
        self.stride = args.model.min_patch_size
        # self.max_patch_num = math.ceil((self.seq_len - self.min_patch_size) / self.stride) + 1
        
        self.similarity_func = args.model.similarity_func
        
        if self.revin:
            # RevIn_mask
            if self.task_name == 'regular':
                self.revin_layer = RevIN(args.model.enc_in, affine=True, subtract_last=False)
            else:
                self.revin_layer = MaskedRevIN(args.model.enc_in, affine=True, subtract_last=False)
        
        # FFT编码
        self.fft_k = args.model.fft_k
        self.use_fft = args.model.use_fft
        if self.use_fft:
            self.hidden_layer = nn.Linear(self.fft_k*2, self.d_model)
            self.re_hidden_layer = nn.Linear(self.d_model, self.fft_k*2)
        else:
            self.emb_layer = nn.Linear(self.min_patch_size, self.d_model, bias=False)
        
        # self.patch_attention = nn.MultiheadAttention(self.d_model, 1, dropout=0.1, batch_first=True)
        self.var_atten = MaskedVariableCrossAttention(hidden_dim=self.d_model, num_heads=args.model.n_heads)

        self.position_embedding = PositionalEmbedding(self.d_model)
        

        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            # True,
                            False,
                            args.model.factor,
                            attention_dropout=args.model.dropout,
                            output_attention=args.model.output_attention,
                        ),
                        args.model.d_model,
                        1,
                    ),
                    args.model.d_model,
                    args.model.d_ff,
                    dropout=args.model.dropout,
                    activation=args.model.activation,
                )
                for l in range(args.model.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.model.d_model),
        )
        
        
        # Prediction Head
        if self.task_name == 'regular':
            self.time_dim = 0
            self.head_nf = self.d_model * int((self.seq_len - self.min_patch_size) / self.stride + 1)
            self.head = FlattenHead(
                    args.model.enc_in,
                    self.head_nf,
                    self.pred_len,
                    head_dropout=args.model.dropout,
                )
            
        elif self.task_name == 'irregular':
            self.time_dim = self.d_model
            self.te_scale = nn.Linear(1, 1)
            self.te_periodic = nn.Linear(1, self.d_model - 1)
            self.decoder = nn.Sequential(
                    nn.Linear(self.d_model, self.d_model),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.d_model, self.d_model),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.d_model, 1)
                )
        self.usetime = args.model.time
        self.setting = self.get_setting(args)
        
    def get_setting(self, args):
        setting = 'Dynamic_linear_{}_{}_lr{}_df{}_dm{}_{}_sl{}_pl{}_t{}_time{}_fft{}_layer{}_patch{}_revin{}'.format(
            args.data.data_path,
            args.data.freq,
            args.train.lr,
            args.model.d_ff,
            args.model.d_model,
            args.model.loss.criterion,
            args.model.seq_len,
            args.model.pred_len,
            args.model.threshold,
            args.model.time,
            args.model.use_fft,
            args.model.e_layers,
            args.model.min_patch_size,
            args.model.revin,
        )
        return setting
    
    def LearnableTE(self, tt):
        # learnable continuous time embeddings
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)
    
    def batch_pad_tensor_list(self, tensor_list, lengths, max_len):
        # 假设 tensor_list: List[Tensor], each [Nᵢ, D]
        B = len(tensor_list)
        D = tensor_list[0].shape[1]
        device = tensor_list[0].device
        lengths = torch.tensor([t.shape[0] for t in tensor_list], device=device)
        # max_len = lengths.max().item()

        # 提前构造 [B, max_len, D] 的全 0 张量
        padded = torch.zeros(B, max_len, D, device=device)

        # 将所有 tensor 填入 padded（构造索引 mask）
        idx = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)  # [B, max_len]
        mask = idx < lengths.unsqueeze(1)  # [B, max_len] 位置为 True 才填入数据

        # concat 所有 tensor 到 [sum(Nᵢ), D]
        concat = torch.cat(tensor_list, dim=0)  # [total_len, D]
        padded[mask] = concat  # 利用 mask 写入

        return padded  # [B, max_len, D]

    def merge_patches_parallel(self, patch_inputs, BOARDERS):
        B, N, D = patch_inputs.shape
        device = patch_inputs.device
        
        # 1. 计算每个patch的合并组
        # 创建组ID张量，初始化为-1
        group_ids = torch.full((B, N), -1, dtype=torch.long, device=device)
        
        # 使用布尔掩码找到所有需要合并的边界
        merge_boundaries = BOARDERS[:, 1::2] == 1  # [B, N-1]
        
        # 创建累积和来计算组ID
        cumsum = torch.cumsum(~merge_boundaries, dim=1)  # [B, N-1]
        group_ids[:, 0] = 0  # 第一个patch总是组0
        group_ids[:, 1:] = cumsum
        
        # 2. 计算每个组的平均值
        # 创建结果张量
        new_patch_inputs = torch.zeros_like(patch_inputs)
        
        # 获取每个样本的唯一组ID和有效组
        # unique_groups = [torch.unique(group_ids[b]) for b in range(B)]
        # valid_groups = [groups[groups != -1] for groups in unique_groups]
        # max_groups = max(len(groups) for groups in valid_groups)
        
        # 创建组ID扩展张量用于scatter_reduce
        group_ids_expanded = group_ids.unsqueeze(-1).expand(-1, -1, D)
        
        # 使用scatter_reduce进行并行平均池化
        new_patch_inputs.scatter_reduce_(
            1,
            group_ids_expanded,
            patch_inputs,
            reduce='mean',
            include_self=False
        )
        
        # 计算每个组的大小并更新BOARDERS
        new_BOARDERS = torch.zeros_like(BOARDERS)
        new_BOARDERS[:, ::2] = -1  # 奇数列填充-1
        new_BOARDERS[:, 1::2] = 0  # 偶数列初始填充0
        
        # 替换原有bincount部分
        patch_weights = BOARDERS[:, ::2].clamp(min=1)  # 获取原始patch/组大小，确保≥1
        group_sizes = torch.zeros_like(group_ids, dtype=torch.int)

        # 并行计算每个组的累计大小
        group_sizes.scatter_add_(
            dim=1,
            index=group_ids,
            src=patch_weights
        )
        # group_sizes[group_sizes == 0] = -1
        
        for b in range(B):
            # sizes = torch.bincount(group_ids[b], minlength=N)[:len(valid_groups[b])]
            # 更新BOARDERS的奇数列
            # new_BOARDERS[b, ::2][:len(sizes)] = sizes
            # id_groups = group_ids[b].unique()[:(BOARDERS[b, ::2].sum()-group_sizes[b].sum())]
            # 获取BOARDERS[b, ::2]中复数项的个数
            num_complex = (BOARDERS[b, ::2] < 0).sum().item()
            id_groups = group_ids[b].unique() if num_complex == 0 else group_ids[b].unique()[:-num_complex] 
            new_BOARDERS[b, id_groups*2] = group_sizes[b, id_groups]
        
        return new_patch_inputs, new_BOARDERS

    def var_cross_attention(self, x, mask):
        """
        Args:
            x: [bs x n_vars x num_patches x hidden_dim]
            mask: [bs x n_vars x num_patches]
        Returns:
            out: [bs x n_vars x num_patches x hidden_dim]
        """
        B, K, N, D = x.shape
        device = x.device
        
        # 创建变量间的attention mask
        # var_mask = torch.ones(K, K, device=device)
        # var_mask.fill_diagonal_(0)  # 对角线设为0，表示不能看到自己
        var_mask = (~torch.eye(K, dtype=torch.bool, device=x.device)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # 扩展x为[B, K, K, N, D]，其中第二个K维度用于复制query
        x_expanded = x.unsqueeze(1).expand(B, K, K, N, D)  # [B, K, K, N, D]
        
        # 创建key和value，使用var_mask来mask掉自身
        key = x_expanded.masked_fill(~var_mask, 0)  # [B, K, K, N, D]
        value = key.clone()
        
        # 将key和value reshape为[B, K, (K-1)*N, D]
        key = key.reshape(B, K, (K-1)*N, D)
        value = value.reshape(B, K, (K-1)*N, D)
        
        # 将query reshape为[B, K, N, D]
        query = x
        
        # 扩展mask为[B, K, N]
        mask_expanded = mask.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, K, N]
        mask_expanded = mask_expanded.reshape(B, K, (K-1)*N)  # [B, K, (K-1)*N]
        
        # 计算attention
        attn_output, _ = self.patch_attention(
            query.reshape(B*K, N, D),
            key.reshape(B*K, (K-1)*N, D),
            value.reshape(B*K, (K-1)*N, D),
            key_padding_mask=~mask_expanded.reshape(B*K, (K-1)*N).bool()
        )
        
        # 将输出reshape回原始形状
        out = attn_output.reshape(B, K, N, D)
        
        return out

    def forward(self, x, x_mark=None, y_mark=None, x_mask=None, y_mask=None, batch=None):
        # x: [batch_size, seq_len, n_vars]
        B, T, K = x.shape

        # Normalization from Non-stationary Transformer
        if self.revin:
            if self.task_name == 'regular':
                x = self.revin_layer(x, 'norm')
            else:
                x = self.revin_layer(x, x_mask, 'norm')
        
        # 按照 min_patch_size 进行 patching
        if T < self.min_patch_size:
            padding = torch.zeros([B, self.min_patch_size - T, K]).to(x.device)
            x = torch.cat((x, padding), dim=1)
        inputs = rearrange(x, 'b t K -> (b K) t')

        patch_size = self.min_patch_size
        marge_patch_size = self.min_patch_size * 2
        max_patch_num = (T - self.min_patch_size) // self.stride + 1 # +1 要考虑数据加载后本身input已经有插入值nan，且不一致，如何解决padding问题
        BOARDERS = torch.ones([B*K, 2*max_patch_num-1], dtype=torch.int).to(x.device)

        # 将无效的0值移到每行末尾，并同步调整x_mask
        if x_mask is not None:
            x_mask = rearrange(x_mask, 'b t K -> (b K) t')
            # x, x_mask: [B*K, T]
            # 获取有效元素的数量
            # valid_counts = x_mask.sum(dim=1)  # [B*K]
            # T = x.shape[1]
            # 对每一行进行排序，使得mask为1的元素排在前面
            sort_idx = torch.argsort(-x_mask, dim=1, stable=True)  # [B*K, T]
            inputs = torch.gather(inputs, 1, sort_idx)
            x_mask = torch.gather(x_mask, 1, sort_idx)
            
            # Expand x_mark to [B, K, T] to match inputs [B*K, T]
            x_mark_expanded = x_mark.unsqueeze(1).expand(-1, K, -1).reshape(B*K, T)
            # Adjust x_mark according to the same sort_idx as inputs/x_mask
            x_mark = torch.gather(x_mark_expanded, 1, sort_idx)
            # After this, x_mark_expanded is [B*K, T], aligned with inputs/x_mask
            # If you need to use x_mark later, use x_mark_expanded
            # x_mark = x_mark_expanded

            # 对x_mask按min_patch_size切分，并对每个patch内做或操作，得到patch级mask
            patch_masks = x_mask.unfold(1, self.min_patch_size, self.stride)  # [B*K, num_patches, min_patch_size]
            patch_masks = patch_masks.any(dim=-1).int()  # [B*K, num_patches]
            BOARDERS[:, ::2] = patch_masks
            
            # 计算相邻时间点的差值
            x_mark_diff = torch.abs(x_mark[:, 1:] - x_mark[:, :-1])  # [B, T-1, K]
        # if x_mask is not None:
            x_mask_bool = x_mask.bool()
            # 只计算mask为True的相邻对
            valid_pairs = x_mask_bool[:, 1:] & x_mask_bool[:, :-1]
            # 只保留有效对
            if valid_pairs.any():
                x_mark_max_diff = x_mark_diff[valid_pairs].max().item()
            else:
                x_mark_max_diff = 0.0


        marge_patch_inputs = None
        # patch_old_fft_emb = None
        while True:
            # patch_size += self.min_patch_size
            # 对每个patch进行FFT变换
            
            patch_inputs = inputs.unfold(dimension=1, size=patch_size, step=self.stride)  # inputs: [bs x num_patch x n_vars x min_patch_size]
            if self.use_fft:
                patch_inputs_fft, _ = temporal_to_frequency(patch_inputs, k=self.fft_k)  # [bs x num_patch x n_vars x 2k]
                patch_inputs_fft_emb = self.hidden_layer(patch_inputs_fft)  # [bs x num_patch x n_vars x hidden_dim]
            else:
                patch_inputs_fft_emb = self.emb_layer(patch_inputs)  # [bs x num_patch x n_vars x hidden_dim]

            # num_patches = patch_inputs.shape[1]
            
            # 根据attns合并不同的patch

            if x_mark is not None:
                patch_x_mark = x_mark.unfold(dimension=1, size=patch_size, step=self.stride)  # inputs: [bs x num_patch x n_vars x min_patch_size]
                patch_x_mark_diff = patch_x_mark[:,1:,0]-patch_x_mark[:,:-1,-1]
                # 计算归一化差值的e的负次方
                patch_x_mark_exp = torch.exp(-patch_x_mark_diff / x_mark_max_diff + 1e-8)  # [B*K, num_patches-1]
            else:
                patch_x_mark_exp = 1

            attns = F.softmax(torch.matmul(patch_inputs_fft_emb, patch_inputs_fft_emb.transpose(-2, -1)) / math.sqrt(patch_inputs_fft_emb.shape[-1]), dim=-1)
            sim_mask = (attns[:,0,1:] * patch_x_mark_exp) < self.threshold  # [B*K, num_boarders] 判断相邻patch相似度是否小于阈值
            BOARDERS[:,1::2] = ~sim_mask
            
            # 当相邻patch相似时，要判断是否是连续的patch，即merge后与merge前单独patch的相似度是否大于阈值
            # patch_size += self.min_patch_size
            new_patch_inputs = inputs.unfold(dimension=1, size=marge_patch_size, step=self.stride)  # inputs: [bs x num_patch x n_vars x min_patch_size]
            if self.use_fft:
                new_patch_inputs_fft, _ = temporal_to_frequency(new_patch_inputs, k=self.fft_k)  # [bs x num_patch x n_vars x 2k]
                new_patch_inputs_fft_emb = self.hidden_layer(new_patch_inputs_fft)  # [bs x num_patch x n_vars x hidden_dim]
            else:
                # Average pooling to reduce patch size by half
                new_patch_inputs = F.avg_pool1d(new_patch_inputs, kernel_size=2, stride=2)
                new_patch_inputs_fft_emb = self.emb_layer(new_patch_inputs)
            # 计算new_patch_inputs_fft_emb和patch_inputs_fft_emb的相似度 [B*K, (num_patches-1)*num_patches]
            
            # merge_patch 重合，此处无需计算时间差，但当使用GAN时需要重新考虑
            new_attns = F.softmax(torch.matmul(new_patch_inputs_fft_emb, patch_inputs_fft_emb.transpose(-2, -1)) / math.sqrt(patch_inputs_fft_emb.shape[-1]), dim=-1)
            new_sim_mask = (new_attns[:, :, :-1].diagonal(dim1=-2, dim2=-1) < self.threshold) | (new_attns[:, :, 1:].diagonal(dim1=-2, dim2=-1) < self.threshold)  # [B*K, num_patches-1]
            # 更新BOARDERS，只有当新patch与原始patch相似度大于阈值时才合并
            BOARDERS[:,1::2] = BOARDERS[:,1::2] & ~new_sim_mask
            
            ######### 根据BOARDERS合并patches #########
            if not BOARDERS[:,1::2].any():
                # 如果BOARDERS全为False，说明没有相似度大于阈值的patch，无需进行合并，直接返回
                if marge_patch_inputs is None:
                    marge_patch_inputs = patch_inputs
                break
            else:
                # 合并patches
                print("merge")
                marge_patch_inputs, BOARDERS = self.merge_patches_parallel(patch_inputs, BOARDERS)
                inputs = marge_patch_inputs.reshape(B*K, -1)
                
            # del patch_inputs, patch_inputs_fft_emb, attns, sim_mask, new_patch_inputs, new_patch_inputs_fft_emb, new_attns, new_sim_mask
            # gc.collect()
            # torch.cuda.empty_cache()

        '''
        # Get the first sample's data
        sample_data = marge_patch_inputs[0].cpu().detach().numpy()
        
        # Create save directory if it doesn't exist
        save_dir = './Run/plots'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Plot and save each variable separately
        n_vars = sample_data.shape[0]
        for i in range(n_vars):
            plt.figure(figsize=(12, 4))
            plt.plot(sample_data[i])
            plt.title(f'Variable {i+1}')
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            plt.grid(True)
            plt.tight_layout()
            
            # Save individual plot
            plt.savefig(os.path.join(save_dir, f'marge_patch_inputs_var_{i+1}.png'))
            plt.close()
        ''' 
        
        # dynamic_patch_emb n_vars * [bs x num_patches' x hidden_dim]
        # var_attention encoder
        if self.use_fft:
            marge_patch_inputs_fft, topk_indices = temporal_to_frequency(marge_patch_inputs, k=self.fft_k)  # [bs*n_vars x num_patch x 2k]
            marge_patch_inputs_fft_emb = self.hidden_layer(marge_patch_inputs_fft)  # [bs*n_vars x num_patch x hidden_dim]
        else:
            marge_patch_inputs_fft_emb = self.emb_layer(marge_patch_inputs)  # [bs*n_vars x num_patch x hidden_dim]
            
        marge_patch_inputs_fft_emb = rearrange(marge_patch_inputs_fft_emb, '(b k) n d -> b k n d', b=B, k=K, n=marge_patch_inputs_fft_emb.shape[1])  # [bs x n_vars x num_patches x hidden_dim]
        merge_patch_inputs_mask = BOARDERS[:, ::2] > 0
        marge_patch_mask = rearrange(merge_patch_inputs_mask, '(b k) n -> b k n', b=B, k=K)  # [bs x n_vars x num_patches]
        
        # 对变量间进行cross attention
        var_attn_out = self.var_atten(marge_patch_inputs_fft_emb, marge_patch_mask)
        
        # 将结果reshape为encoder输入格式
        input_emb = rearrange(var_attn_out, 'b k n d -> (b k) n d')
        
        # Encoder
        position_emb = self.position_embedding(input_emb)
        emb = input_emb + position_emb
        enc_out, _ = self.encoder(emb)  # [bs*n_vars x num_patch x hidden_dim]
        # encoder_mask = marge_patch_mask.reshape(B*K, -1).unsqueeze(1).unsqueeze(1)
        # enc_out, _ = self.encoder(emb, attn_mask=encoder_mask)  # [bs*n_vars x num_patch x hidden_dim]
        enc_out = rearrange(enc_out, '(b k) n d -> b k n d', b=B, k=K, n=enc_out.shape[1])

        # Get the last valid patch index for each variable
        last_valid_idx = marge_patch_mask.sum(dim=-1) - 1  # [bs x n_vars]
        
        # Create index tensor for gathering
        batch_idx = torch.arange(B, device=enc_out.device).view(-1, 1, 1).expand(-1, K, 1)  # [bs x n_vars x 1]
        var_idx = torch.arange(K, device=enc_out.device).view(1, -1, 1).expand(B, -1, 1)    # [bs x n_vars x 1]
        patch_idx = last_valid_idx.unsqueeze(-1)  # [bs x n_vars x 1]
        
        # Gather the last valid patch representation
        Z0 = enc_out[batch_idx, var_idx, patch_idx]  # [bs x n_vars x 1 x hidden_dim]
        Z0 = Z0.squeeze(2)  # [bs x n_vars x hidden_dim]
        # Z0 = rearrange(Z0, 'b k d -> (b k) d')  # [bs*n_vars x hidden_dim]

        if self.use_fft:
            enc_out_tem = self.re_hidden_layer(enc_out.reshape(B*K, enc_out.shape[2], enc_out.shape[3]))
            enc_out = frequency_to_temporal(enc_out_tem, topk_indices, self.min_patch_size, self.fft_k)
            enc_out = rearrange(enc_out, '(b k) n d -> b k n d', b=B, k=K, n=enc_out.shape[1])

        
        if self.task_name == 'regular':
            
            enc_out = enc_out.permute(0, 1, 3, 2)

            # Decoder
            dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
            dec_out = dec_out.permute(0, 2, 1)
            
            if self.revin:
                pred_y = self.revin_layer(dec_out, 'denorm')

        elif self.task_name == 'irregular':
            """ Decoder """
            L_pred = y_mark.shape[-1]
            y_mark = y_mark.view(B, 1, L_pred, 1).repeat(1, K, 1, 1)
            te_pred = self.LearnableTE(y_mark)
            
            # Z0: [bs x nvars x D]
            h = Z0.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)
            if self.usetime:
                h = h + te_pred
            
            pred_y = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1)
            
            if self.revin:
                pred_y = self.revin_layer(pred_y, None, 'denorm')
            
            return pred_y
    
        return pred_y[:, -self.pred_len :, :]
    
    def predict(self, Z0):
        time_steps_to_predict = torch.arange(start=0, end=self.pred_len + 1, step=1).float()  # horizon 1 + 24
        time_steps_to_predict = time_steps_to_predict / len(time_steps_to_predict)
        pred_z, fe = self.diffeq_solver.solve(self.unk_odefunc, Z0, time_steps_to_predict)  # T x B*var x D
        pred_z = pred_z[1:]

        return pred_z, fe

    def reconstruct(self, Z0):
        time_steps_to_recon = torch.arange(start=0, end=-self.seq_len, step=-1).float()  # seq_len 24
        time_steps_to_recon = time_steps_to_recon / len(time_steps_to_recon)
        recon_z, rev_fe = self.diffeq_solver.solve(self.unk_odefunc, Z0, time_steps_to_recon)  # seq_len x B*var x D
        recon_z = recon_z.flip(dims=[0])

        return recon_z, rev_fe
    
if __name__ == "__main__":
    config = load_config('../Model_Config/BAOWU/Dynamic.yaml')
    config = ConfigDict(config)
    config.exp_idx = 0
    config.des = 'test'
    model = Model(config)
    X = torch.randn(config.data.batch_size, config.model.seq_len, config.model.input_dim)
    Y = torch.randn(config.data.batch_size, config.model.pred_len, config.model.output_dim)
    print(model(X)[0].shape)
    print(model.get_loss(X, Y))

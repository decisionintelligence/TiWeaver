import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math

from models.layers.DWT import DWTClustering
from models.layers.RevIN import RevIN
from models.layers.Mask_RevIN import MaskedRevIN
from models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.layers.Transformer_EncDec import Encoder, EncoderLayer
from models.layers.Embed import PositionalEmbedding
from models.layers.tools import EncoderAttrs
from models.PatchGAT import IntraPatchGATEncoder

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
    return fft_vec
    

    
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
        self.seq_len_max_irr = args.model.seq_len_max_irr
        self.pred_len_max_irr = args.model.pred_len_max_irr
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
            # self.revin_layer = RevIN(args.model.enc_in, affine=True, subtract_last=False)
            self.revin_layer = MaskedRevIN(args.model.enc_in, affine=True, subtract_last=False)
        
        # FFT编码
        # self.fft_k = args.model.fft_k
        # self.use_fft = args.model.use_fft

        self.hidden_layer = nn.Linear(1, self.d_model)
        # self.emb_layer = nn.Linear(self.min_patch_size, self.d_model, bias=False)
        
        # self.patch_attention = nn.MultiheadAttention(self.d_model, 1, dropout=0.1, batch_first=True)
        self.var_atten = MaskedVariableCrossAttention(hidden_dim=self.d_model*1, num_heads=args.model.n_heads)

        self.position_embedding = PositionalEmbedding(self.d_model*1)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
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
        # self.merge_patch_graph = PatchClusterGAT(in_dim=self.d_model * 2, out_dim=self.d_model * 2, heads=4, min_patch_size=self.min_patch_size)
        
        # self.gat_patch_graph = GATConv(self.d_model, self.d_model, heads=1, concat=False)
        
        # Prediction Head
        if self.task_name == 'regular':
            
            self.head_nf = self.d_model*1 * int((self.seq_len - self.min_patch_size) / self.stride + 1)
            self.head = FlattenHead(
                    args.model.enc_in,
                    self.head_nf,
                    self.pred_len,
                    head_dropout=args.model.dropout,
                )
            
        elif self.task_name == 'irregular':
            self.te_scale = nn.Linear(1, 1)
            self.te_periodic = nn.Linear(1, self.d_model - 1)

            # '''
            self.decoder = nn.Sequential(
                    nn.Linear(self.d_model * 1, self.d_model*2),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.d_model*2, self.d_model),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.d_model, 1)
                )
            '''
            if self.seq_len_max_irr is None:
                self.seq_len_max_irr = self.seq_len
                self.pred_len_max_irr = self.pred_len

            # print("seq_len_max_irr", self.seq_len_max_irr)
            self.head_nf = self.d_model*1 * int((self.seq_len_max_irr - self.min_patch_size) / self.stride + 1)
            self.head = FlattenHead(
                    args.model.enc_in,
                    self.head_nf,
                    self.pred_len_max_irr,
                    head_dropout=args.model.dropout,
                )
            # '''
        self.usetime = args.model.time
        self.use_catt = args.model.catt
        self.usegap = args.model.gap
        self.group_type = args.model.group_type
        self.setting = self.get_setting(args)
        
    def get_setting(self, args):
        setting = 'Dynamic_group_{}_{}_time{}_lr{}_df{}_dm{}_{}_sl{}_pl{}_t{}_catt{}_fft{}_layer{}_patch{}_revin{}'.format(
            args.model.group_type,
            args.data.data_path,
            args.model.time,
            args.train.lr,
            args.model.d_ff,
            args.model.d_model,
            args.model.loss.criterion,
            args.model.seq_len,
            args.model.pred_len,
            args.model.threshold,
            args.model.catt,
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


    def merge_patches_parallels(self, patch_inputs, BOARDERS):
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
        
        # 创建结果张量
        new_patch_inputs = torch.zeros_like(patch_inputs)
        
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
            # 获取BOARDERS[b, ::2]中复数项的个数
            num_complex = (BOARDERS[b, ::2] < 0).sum().item()
            id_groups = group_ids[b].unique() if num_complex == 0 else group_ids[b].unique()[:-num_complex] 
            new_BOARDERS[b, id_groups*2] = group_sizes[b, id_groups]
        
        return new_patch_inputs, new_BOARDERS

    def merge_patches_parallel(self, patch_inputs, BOARDERS):
        B, N, D = patch_inputs.shape
        device = patch_inputs.device

        # 1. 计算每个patch的合并组
        group_ids = torch.full((B, N), -1, dtype=torch.long, device=device)
        merge_boundaries = BOARDERS[:, 1::2] == 1  # [B, N-1]
        cumsum = torch.cumsum(~merge_boundaries, dim=1)  # [B, N-1]
        group_ids[:, 0] = 0  # 第一个patch总是组0
        group_ids[:, 1:] = cumsum

        # 计算每个组的大小并更新BOARDERS
        new_BOARDERS = torch.zeros_like(BOARDERS)
        new_BOARDERS[:, ::2] = -1  # 奇数列填充-1
        new_BOARDERS[:, 1::2] = 0  # 偶数列初始填充0
        
        patch_weights = BOARDERS[:, ::2].clamp(min=1)  # 获取原始patch/组大小，确保≥1
        group_sizes = torch.zeros_like(group_ids, dtype=torch.int)

        # 并行计算每个组的累计大小
        group_sizes.scatter_add_(
            dim=1,
            index=group_ids,
            src=patch_weights
        )

        # group_ids_expanded = group_ids.unsqueeze(-1).expand(-1, -1, D)

        # 创建结果张量
        new_patch_inputs = torch.zeros_like(patch_inputs)
        
        for b in range(B):

            unique_elements, counts = torch.unique(group_ids[b], return_counts=True)
            # counts = counts.tolist()
            split_sizes = counts*D
            # 切分数据
            splits = torch.split(patch_inputs[b].reshape(-1), split_sizes.tolist())
            
            # 对每个切分子序列进行均匀降采样
            downsampled_splits = []
            for split, factor in zip(splits, counts.tolist()):
                if factor == 1:
                    # 如果降采样因子为1，则不进行降采样
                    downsampled_splits.append(split)
                else:
                    # 计算降采样后的长度
                    new_length = split.size(0) // factor
                    # 使用均匀间隔的索引进行降采样
                    indices = torch.arange(0, split.size(0), factor)[:new_length]
                    downsampled_splits.append(split[indices])
            
            # 将切分子序列重新组合成一个张量
            merge_inputs = torch.cat(downsampled_splits).reshape(-1, D)
            new_patch_inputs[b, :len(merge_inputs), :] = merge_inputs
        
            # 重构边界
            num_complex = (BOARDERS[b, ::2] < 0).sum().item()
            id_groups = group_ids[b].unique() if num_complex == 0 else group_ids[b].unique()[:-num_complex] 
            new_BOARDERS[b, id_groups*2] = group_sizes[b, id_groups]
        
        return new_patch_inputs, new_BOARDERS

    def forward(self, x, x_mark, y_mark, x_mask=None, y_mask=None):
        # x: [batch_size, seq_len, n_vars]
        B, T, K = x.shape

        # Normalization from Non-stationary Transformer
        if self.revin:
            x = self.revin_layer(x, x_mask, 'norm')
        
        # 按照 min_patch_size 进行 patching
        if T < self.min_patch_size:
            padding = torch.zeros([B, self.min_patch_size - T, K]).to(x.device)
            x = torch.cat((x, padding), dim=1)
        x = rearrange(x, 'b t K -> (b K) t')
        x_mask = rearrange(x_mask, 'b t K -> (b K) t')

        patch_size = self.min_patch_size
        marge_patch_size = self.min_patch_size * 2
        max_patch_num = (T - self.min_patch_size) // self.stride + 1 # +1 要考虑数据加载后本身input已经有插入值nan，且不一致，如何解决padding问题
        BOARDERS = torch.ones([B*K, 2*max_patch_num-1], dtype=torch.int).to(x.device)

        # 将无效的0值移到每行末尾，并同步调整x_mask
        if x_mask is not None:
            
            # 对每一行进行排序，使得mask为1的元素排在前面
            sort_idx = torch.argsort(-x_mask, dim=1, stable=True)  # [B*K, T]
            x = torch.gather(x, 1, sort_idx)
            x_mask = torch.gather(x_mask, 1, sort_idx)
            
            # Expand x_mark to [B, K, T] to match inputs [B*K, T]
            x_mark_expanded = x_mark.unsqueeze(1).expand(-1, K, -1).reshape(B*K, T)
            x_mark = torch.gather(x_mark_expanded, 1, sort_idx)

            # 对x_mask按min_patch_size切分，并对每个patch内做或操作，得到patch级mask
            patch_masks = x_mask.unfold(1, self.min_patch_size, self.stride)  # [B*K, num_patches, min_patch_size]
            patch_masks = patch_masks.any(dim=-1).int()  # [B*K, num_patches]
            BOARDERS[:, ::2] = patch_masks
            
            
        # 计算相邻时间点的差值
        x_mark_diff = torch.abs(x_mark[:, 1:] - x_mark[:, :-1])  # [B, T-1, K]
        if x_mask is not None:
            x_mask_bool = x_mask.bool()
            # 只计算mask为True的相邻对
            valid_pairs = x_mask_bool[:, 1:] & x_mask_bool[:, :-1]
            # 只保留有效对
            if valid_pairs.any():
                x_mark_max_diff = x_mark_diff[valid_pairs].max().item()
            else:
                x_mark_max_diff = x_mark_diff.max().item()
        else:
            x_mark_max_diff = x_mark_diff.max().item()
            
        x_mark_emb = self.LearnableTE(x_mark.unsqueeze(-1))  # [B*K, T, D]
        x_emb = self.hidden_layer(x.unsqueeze(-1))  # [B*K, T, D]
        if self.usetime:
            # print('usetime')
            inputs_emb = x_emb + x_mark_emb  # [B*K, T, D+D]
        else:
            # print('nousetime')
            inputs_emb = x_emb
        # inputs_emb = torch.cat([x_emb, x_mark_emb], dim=-1)  # [B*K, T, D+D]
        # inputs_emb = x_emb

        marge_patch_inputs_emb = None
        patch_inputs = inputs_emb.unfold(dimension=1, size=patch_size, step=self.stride).permute(0, 1, 3, 2)  # inputs: [B*K x num_patch x min_patch_size x 2D]
        patch_inputs_emb = self.intra_patch_graph(patch_inputs)  # [B*K x num_patch x 2D]
        
        total_samples, num_patches, feature_dim = patch_inputs_emb.shape
        
        if self.group_type == "kmean":
            kmeans = KMeans(n_clusters=16, random_state=42)
    
            # 重塑数据用于批量处理 [total_samples * num_patches, feature_dim]
            flattened_patches = rearrange(patch_inputs_emb, 'b t K -> (b t) K').detach().cpu().numpy()
            
            # 为每个样本创建索引
            sample_indices = torch.repeat_interleave(
                torch.arange(total_samples), num_patches
            ).numpy()
            
            # 批量K-Means聚类（对每个样本分别聚类）
            all_cluster_labels = np.zeros(total_samples * num_patches, dtype=np.int32)
            all_cluster_centers = np.zeros((total_samples, 16, feature_dim))
            
            for sample_idx in range(total_samples):
                sample_mask = (sample_indices == sample_idx)
                sample_patches = flattened_patches[sample_mask]
                
                kmeans = KMeans(n_clusters=16, random_state=42, n_init=16)
                cluster_labels = kmeans.fit_predict(sample_patches)
                
                all_cluster_labels[sample_mask] = cluster_labels
                all_cluster_centers[sample_idx] = kmeans.cluster_centers_
            
            # 重新组织结果
            cluster_labels_all = torch.tensor(all_cluster_labels.reshape(total_samples, num_patches))
            cluster_centers_all = torch.tensor(all_cluster_centers)
            
            # 批量计算池化表征
            pooled_representations = torch.zeros_like(cluster_centers_all)
            
            for sample_idx in range(total_samples):
                for cluster_id in range(16):
                    cluster_mask = (cluster_labels_all[sample_idx] == cluster_id)
                    if cluster_mask.sum() > 0:
                        cluster_patches = patch_inputs_emb[sample_idx, cluster_mask]
                        pooled_representations[sample_idx, cluster_id] = cluster_patches.mean(dim=0)
            
            # pooled_representations = pooled_representations.float().to(patch_inputs_emb.device)
            # pooled_representations = rearrange(pooled_representations, '(b k) n d -> b k n d', b=B, k=K)  # [bs x n_vars x num_patches]
            # merge_patch_inputs_mask = BOARDERS[:, ::2] > 0
            # marge_patch_mask = rearrange(merge_patch_inputs_mask, '(b k) n -> b k n', b=B, k=K)  # [bs x n_vars x num_patches]
            # new_mask = torch.ones_like(marge_patch_mask)[:,:,:6]
            # var_attn_out = self.var_atten(pooled_representations, new_mask)

        elif self.group_type == "dwt":
            # 初始化结果
            cluster_centers_all = torch.zeros(total_samples, 10, feature_dim)
            cluster_labels_all = torch.zeros(total_samples, num_patches, dtype=torch.long)
            pooled_representations = torch.zeros(total_samples, 10, feature_dim)
            
            # 对每个样本独立进行DTW聚类
            for sample_idx in range(total_samples):
                sample_patches = patch_inputs_emb[sample_idx]  # [num_patches, feature_dim]
                
                # 转换为numpy用于DTW聚类
                patches_np = sample_patches.detach().cpu().numpy()
                
                # 执行DTW聚类
                dwt_cluster = DWTClustering(n_clusters=10, method='kmedoids')
                cluster_labels = dwt_cluster.fit_predict(patches_np)
                
                cluster_labels_all[sample_idx] = torch.tensor(cluster_labels)
                
                # 对每个簇进行池化
                for cluster_id in range(10):
                    cluster_mask = (cluster_labels == cluster_id)
                    
                    if np.sum(cluster_mask) > 0:
                        cluster_patches = sample_patches[cluster_mask]
                        
                        # 简单平均池化
                        pooled_repr = cluster_patches.mean(dim=0)
                        pooled_representations[sample_idx, cluster_id] = pooled_repr
                        cluster_centers_all[sample_idx, cluster_id] = pooled_repr

        pooled_representations = pooled_representations.float().to(patch_inputs_emb.device)
        pooled_representations = rearrange(pooled_representations, '(b k) n d -> b k n d', b=B, k=K)  # [bs x n_vars x num_patches]
        merge_patch_inputs_mask = BOARDERS[:, ::2] > 0
        marge_patch_mask = rearrange(merge_patch_inputs_mask, '(b k) n -> b k n', b=B, k=K)  # [bs x n_vars x num_patches]
        new_mask = torch.ones_like(marge_patch_mask)[:,:,:16]
        var_attn_out = self.var_atten(pooled_representations, new_mask)

        input_emb = rearrange(var_attn_out, 'b k n d -> (b k) n d')
        
        # Encoder
        position_emb = self.position_embedding(input_emb)
        emb = input_emb + position_emb
        # encoder_mask = marge_patch_mask.reshape(B*K, -1).unsqueeze(1).unsqueeze(1).repeat(1, 1, marge_patch_mask.shape[-1], 1)
        enc_out, _ = self.encoder(emb)  # [bs*n_vars x num_patch x hidden_dim]
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
        
        if self.task_name == 'regular':
            
            enc_out = enc_out.permute(0, 1, 3, 2)

            # Decoder
            dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
            dec_out = dec_out.permute(0, 2, 1)
            
            if self.revin:
                pred_y = self.revin_layer(dec_out, 'denorm')
                
            return pred_y[:, -self.pred_len :, :]
            
        elif self.task_name == 'irregular':
            """ Decoder """
            L_pred = y_mark.shape[-1]
            y_mark = y_mark.view(B, 1, L_pred, 1).repeat(1, K, 1, 1)
            te_pred = self.LearnableTE(y_mark) # [bs x nvars x L_pred x D]
            
            # padding_size = (self.min_patch_size - te_pred.size(2) % self.min_patch_size) % self.min_patch_size
            # if padding_size > 0:
                # last_value = te_pred[:, : ,-1, :].unsqueeze(2).repeat(1,1,padding_size,1)
                # te_pred = torch.cat([te_pred, last_value], dim=2)
            
            # te_pred_patch = te_pred.unfold(dimension=2, size=self.min_patch_size, step=self.stride)  # [bs x nvars x num_patch x min_patch_size x D]
            # te_pred_patch = te_pred_patch.mean(dim=-1)  # [bs x nvars x num_patch x D]

            # Z0: [bs x nvars x D]
            h = Z0.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)
            if self.usetime:
                h = h + te_pred
            
            pred_y = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1)
            # pred_y = self.decoder(h).reshape(B, K, -1).permute(0, 2, 1) # [bs x nvars x L_pred]
            
            if self.revin:
                pred_y = self.revin_layer(pred_y, y_mask, 'denorm')
            
            return pred_y
    
if __name__ == "__main__":
    config = load_config('../Model_Config/BAOWU/Dynamic.yaml')
    config = ConfigDict(config)
    config.exp_idx = 0
    config.des = 'test'
    model = Model(config)
    # X = torch.randn(config.data.batch_size, config.model.seq_len, config.model.input_dim)
    # Y = torch.randn(config.data.batch_size, config.model.pred_len, config.model.output_dim)
    # print(model(X)[0].shape)
    # print(model.get_loss(X, Y))
    # 创建测试数据
    patch_inputs = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                                [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]])
    BOARDERS = torch.tensor([[1, 0, 1, 1, 1, 0, 1],
                            [1, 0, 1, 0, 1, 1, 1]])

    # # 创建类的实例并调用函数
    # example = ExampleClass()
    merged_patches, new_BOARDERS = model.merge_patches_parallel(patch_inputs, BOARDERS)


import time
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
from collections import Counter

class SinglePatchGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=1, concat=False):
        super().__init__()
        self.gat = GATConv(in_dim, hidden_dim, heads=heads, concat=concat)

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)

class PatchGAT(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=64, head=1, gat_num_layers=3, concat=False):
        super(PatchGAT, self).__init__()
        self.gat_num_layers = gat_num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.head = head
        self.concat = concat
        self.residual = nn.Identity()
        self.fusion = self.get_fusion()

    def forward(self, x, edge_index):
        
        out = self.fusion[0](x, edge_index)
        out = F.relu(out)

        for gnn in self.fusion[1:-1]:
            residual = self.residual(out)
            out = gnn(out, edge_index)
            out = F.relu(out) + residual

        x = self.fusion[-1](out, edge_index)

        return x

    def get_fusion(self):
        fusion = nn.ModuleList()

        fusion.append(
                GATConv(in_channels=self.input_dim, out_channels=self.hid_dim,
                        heads=self.head, concat=self.concat)
            )

        for _ in range(self.gat_num_layers - 2):
                fusion.append(
                    GATConv(in_channels=self.hid_dim, out_channels=self.hid_dim,
                            heads=self.head, concat=self.concat)
                )

        fusion.append(
                GATConv(in_channels=self.hid_dim, out_channels=self.output_dim,
                        heads=self.head, concat=self.concat)
            )

        return fusion


class IntraPatchGATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=1, gat_num_layer=3):
        super(IntraPatchGATEncoder, self).__init__()
        self.gat_layer = SinglePatchGAT(in_dim, hidden_dim, heads)
        # self.gat_layer = PatchGAT(in_dim, in_dim, hidden_dim, heads, gat_num_layer)
        
    def build_batch_graph(self, patch_inputs):
        """
        patch_inputs: Tensor of shape (B*N, P, d)
        return: torch_geometric.data.Data with (B*N*(P+1), d)
        """
        BN, P, d = patch_inputs.shape
        device = patch_inputs.device
        
        # 为每个patch添加全局节点
        # global_nodes = patch_inputs.mean(dim=1, keepdim=True)  # (BN, 1, d)
        global_nodes = torch.zeros([BN, 1, d], device=patch_inputs.device)
        x = torch.cat([patch_inputs, global_nodes], dim=1)     # (BN, P+1, d)
        x = x.view(BN*(P+1), d)                               # (BN*(P+1), d)
        
        # 构建边索引
        patch_offsets = (torch.arange(BN, device=device) * (P+1)).view(-1, 1, 1)
        
        # 1. Patch内部全连接边
        base_idx = torch.arange(P, device=device)
        grid_i, grid_j = torch.meshgrid(base_idx, base_idx, indexing='ij')
        patch_edges = torch.stack([grid_i.flatten(), grid_j.flatten()], dim=1)  # [P*P, 2]
        
        # 2. Patch到全局节点的边
        global_node_idxs = torch.full((P,), P, dtype=torch.long, device=device)
        patch_to_global = torch.stack([base_idx, global_node_idxs], dim=1)  # [P, 2]
        
        # 合并边并扩展到所有patch
        intra_edges = torch.cat([patch_edges, patch_to_global], dim=0)  # [P*(P+1), 2]
        intra_edges = intra_edges.unsqueeze(0).expand(BN, -1, 2)        # [BN, P*(P+1), 2]
        intra_edges = intra_edges + patch_offsets
        edge_index = intra_edges.reshape(-1, 2).t().contiguous()        # [2, BN*P*(P+1)]
        
        return Data(x=x, edge_index=edge_index)

    def forward(self, patch_inputs):
        """
        patch_inputs: (B, N, P, d)
        return: patch_reprs: (B, N, hidden_dim)
        """
        B, N, P, d = patch_inputs.shape
        
        # 一次性构建整个batch的图
        patch_inputs = patch_inputs.reshape(B*N, P, d)
        graph = self.build_batch_graph(patch_inputs)
        
        # GAT处理
        out = self.gat_layer(graph.x, graph.edge_index)  # (B*N*(P+1), hidden_dim)
        
        # 提取每个patch的全局节点表示
        global_node_indices = torch.arange(P, (B*N+1)*(P+1)-1, P+1, device=out.device)
        patch_reprs = out[global_node_indices].view(B, N, -1)  # (B, N, hidden_dim)
        
        return patch_reprs
    

class PatchClusterGATEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim, heads=heads, concat=False)

    def merge_flags_to_group_ids(self, merge_flags):
        """
        merge_flags: (B, N-1), 0/1
        return: (B, N), group IDs
        """
        B, N1 = merge_flags.shape
        group_ids = torch.zeros((B, N1 + 1), dtype=torch.long, device=merge_flags.device)
        group_ids[:, 1:] = torch.cumsum(1 - merge_flags, dim=1)
        return group_ids

    def build_batch_cluster_graph(self, patch_emb_batch, group_ids_batch):
        """
        patch_emb_batch: (B, N, D)
        group_ids_batch: (B, N)
        return: (Data, list) - graph data and cluster counts per batch
        """
        B, N, D = patch_emb_batch.shape
        device = patch_emb_batch.device
        
        # Calculate offsets for each batch
        max_clusters = (group_ids_batch.max(dim=1)[0] + 1).max().item()
        offsets = (N + max_clusters) * torch.arange(B, device=device)
        
        # Prepare node features (patches + global nodes)
        global_nodes = torch.zeros((B, max_clusters, D), device=device)
        x = torch.cat([
            patch_emb_batch.reshape(B*N, D),
            global_nodes.reshape(B*max_clusters, D)
        ], dim=0)
        
        # Build all edges in batch
        edge_list = []
        for b in range(B):
            group_ids = group_ids_batch[b]
            patch_offset = offsets[b]
            
            # Patch to global node edges
            patch_indices = torch.arange(N, device=device)
            global_indices = N + group_ids
            patch2global = torch.stack([
                patch_indices + patch_offset,
                global_indices + patch_offset
            ], dim=1)
            
            # Intra-cluster edges (adjacent patches in same cluster)
            same_cluster = (group_ids[:-1] == group_ids[1:])
            idx = torch.arange(N-1, device=device)[same_cluster]
            adj_edges = torch.cat([
                torch.stack([idx+patch_offset, idx+1+patch_offset], dim=1),
                torch.stack([idx+1+patch_offset, idx+patch_offset], dim=1)
            ], dim=0) if N > 1 else torch.zeros((0, 2), device=device, dtype=torch.long)
            
            edges = torch.cat([patch2global, adj_edges], dim=0)
            edge_list.append(edges)
        
        edge_index = torch.cat(edge_list, dim=0).t().contiguous()
        cluster_counts = (group_ids_batch.max(dim=1)[0] + 1).tolist()
        
        return Data(x=x, edge_index=edge_index), cluster_counts

    def forward(self, patch_inputs_emb, BOARDERS):
        """
        patch_inputs_emb: (B, N, D)
        BOARDERS: (B, 2N - 1)
        return: 
            - cluster_reprs: (B, max_clusters, out_dim)
            - updated_BOARDERS: (B, 2N - 1)
        """
        B, N, D = patch_inputs_emb.shape
        merge_flags = BOARDERS[:, 1::2]  # (B, N-1)
        group_ids_batch = self.merge_flags_to_group_ids(merge_flags)
        
        # Build single batched graph
        graph, cluster_counts = self.build_batch_cluster_graph(patch_inputs_emb, group_ids_batch)
        x_out = self.gat(graph.x, graph.edge_index)
        
        # Extract global node representations
        max_clusters = max(cluster_counts)
        cluster_reprs = torch.zeros(B, max_clusters, x_out.shape[-1], device=x_out.device)
        
        # Calculate offsets for global nodes
        offsets = (N + max_clusters) * torch.arange(B, device=x_out.device)
        global_node_starts = offsets + N
        
        # Vectorized update of cluster_reprs and BOARDERS
        # Create mask for valid cluster positions
        cluster_mask = torch.arange(max_clusters, device=x_out.device).expand(B, -1) < torch.tensor(cluster_counts, device=x_out.device).unsqueeze(1)
        
        # Vectorized assignment of cluster representations
        global_node_indices = global_node_starts.unsqueeze(1) + torch.arange(max_clusters, device=x_out.device)
        global_node_indices = torch.minimum(global_node_indices, torch.tensor(x_out.shape[0]-1, device=x_out.device))
        cluster_reprs[cluster_mask] = x_out[global_node_indices[cluster_mask]]
        
        # Vectorized BOARDERS update
        # updated_BOARDERS = torch.ones_like(BOARDERS, dtype=BOARDERS.dtype, device=BOARDERS.device)
        # updated_BOARDERS[:, ::2] = -1
        # valid_counts = torch.stack([torch.bincount(g) for g in group_ids_batch])
        # count_mask = torch.arange(valid_counts.shape[1], device=BOARDERS.device).unsqueeze(0) < (valid_counts > 0).sum(dim=1).unsqueeze(1)
        # updated_BOARDERS[:, ::2][count_mask] = valid_counts[valid_counts > 0]

         # Update BOARDERS (vectorized)
        updated_BOARDERS = torch.ones_like(BOARDERS, dtype=BOARDERS.dtype, device=BOARDERS.device)
        updated_BOARDERS[:, ::2] = -1

        for b in range(B):
            start = global_node_starts[b]
            end = start + cluster_counts[b]
            cluster_reprs[b, :cluster_counts[b]] = x_out[start:end]

            valid_counts = torch.bincount(group_ids_batch[b])
            updated_BOARDERS[b, ::2][:len(valid_counts)] = valid_counts
            
        return cluster_reprs, updated_BOARDERS


class PatchClusterGAT(nn.Module):
    def __init__(self, in_dim, out_dim, min_patch_size, heads=1):
        super().__init__()
        self.min_patch_size = min_patch_size
        self.gat = GATConv(in_dim, out_dim, heads=heads, concat=False)

    def merge_flags_to_group_ids(self, merge_flags):
        """
        merge_flags: (B, N-1), 0/1
        return: (B, N), group IDs
        """
        B, N1 = merge_flags.shape
        group_ids = torch.zeros((B, N1 + 1), dtype=torch.long, device=merge_flags.device)
        group_ids[:, 1:] = torch.cumsum(1 - merge_flags, dim=1)
        return group_ids

    def build_batch_cluster_graph(self, patch_emb_batch, group_ids_batch):
        """
        patch_emb_batch: (B, N, P, D)
        group_ids_batch: (B, N)
        return: (Data, list) - graph data and cluster counts per batch
        """
        B, N, P, D = patch_emb_batch.shape
        device = patch_emb_batch.device
        
        # Calculate offsets for each batch
        max_clusters = (group_ids_batch.max(dim=1)[0] + 1).max().item()
        offsets = (N + max_clusters)*P * torch.arange(B, device=device)
        
        # Prepare node features (patches + global nodes)
        global_nodes = torch.zeros((B, max_clusters, self.min_patch_size, D), device=device)
        x = torch.cat([
            patch_emb_batch,
            global_nodes
        ], dim=1).reshape(-1,D)
        
        # Build all edges in batch
        edge_list = []
        intra_patch_ids = torch.tile(torch.arange(P).unsqueeze(0), (max_clusters, 1)).view(-1).to(device)

        for b in range(B):
            group_ids = group_ids_batch[b]
            # group_ids = group_ids.unsqueeze(1).expand(-1, max_clusters).reshape(-1)
            unique_elements, counts = torch.unique(group_ids_batch[b], return_counts=True)
            group_num_ids = torch.arange(len(unique_elements)*P,device=device).reshape(-1,P).repeat_interleave(counts, dim=0)
            patch_offset = offsets[b]
            
            # Patch to global node edges
            patch_indices = torch.arange(N*P, device=device)
            global_indices = (N*P + group_num_ids).reshape(-1)
            patch2global = torch.stack([
                patch_indices + patch_offset,
                global_indices + patch_offset
            ], dim=1)
            
            # Intra-cluster edges (adjacent patches in same cluster)
            group_ids = group_ids.unsqueeze(1).expand(-1, P).reshape(-1)
            same_cluster = (group_ids[:-1] == group_ids[1:])
            idx = torch.arange(N*P-1, device=device)[same_cluster]
            adj_edges = torch.cat([
                torch.stack([idx+patch_offset, idx+1+patch_offset], dim=1),
                torch.stack([idx+1+patch_offset, idx+patch_offset], dim=1)
            ], dim=0) if N > 1 else torch.zeros((0, 2), device=device, dtype=torch.long)
            
            edges = torch.cat([patch2global, adj_edges], dim=0)
            edge_list.append(edges)
        
        edge_index = torch.cat(edge_list, dim=0).t().contiguous()
        cluster_counts = (group_ids_batch.max(dim=1)[0] + 1).tolist()
        
        return Data(x=x, edge_index=edge_index), cluster_counts
    
    

    def forward(self, patch_inputs_emb, BOARDERS):
        """
        patch_inputs_emb: (B, N, D)
        BOARDERS: (B, 2N - 1)
        return: 
            - cluster_reprs: (B, max_clusters, out_dim)
            - updated_BOARDERS: (B, 2N - 1)
        """
        B, N, P, D = patch_inputs_emb.shape

        Boarder_end = torch.zeros_like(BOARDERS[0],device=BOARDERS.device, dtype=BOARDERS.dtype)
        BOARDERS = torch.cat([BOARDERS, Boarder_end.unsqueeze(0)], dim=0)
        merge_flags = BOARDERS[:, 1::2]  # (B, N-1)
        group_ids_batch = self.merge_flags_to_group_ids(merge_flags)

        inputs_end = torch.zeros_like(patch_inputs_emb[0],device=patch_inputs_emb.device, dtype=patch_inputs_emb.dtype)
        patch_inputs_emb = torch.cat([patch_inputs_emb, inputs_end.unsqueeze(0)], dim=0)

        # Build single batched graph
        graph, cluster_counts = self.build_batch_cluster_graph(patch_inputs_emb, group_ids_batch)
        x_out = self.gat(graph.x, graph.edge_index)
        
        # Extract global node representations
        # Update BOARDERS
        updated_BOARDERS = torch.ones_like(BOARDERS, dtype=BOARDERS.dtype, device=BOARDERS.device)
        updated_BOARDERS[:, ::2] = -1

        for b in range(B):
            valid_counts = torch.bincount(group_ids_batch[b])
            updated_BOARDERS[b, ::2][:len(valid_counts)] = valid_counts

        # 提取 global_nodes
        global_nodes = x_out.reshape(B+1, N*2, P, -1)[:-1, N:]
            
        return global_nodes, updated_BOARDERS[:-1,:]
    
if __name__ == "__main__":

    # '''
    B, N, P, d = 2, 3, 3, 4
    x = torch.randn(B, N, P, d)
    model = IntraPatchGATEncoder(in_dim=d, hidden_dim=32)
    start_time = time.time()
    out = model(x)  # 应该得到(B, N, 32)的输出
    end_time = time.time()
    print('Total time:', end_time - start_time)

    '''  

    B, N, D = 3, 6, 4  # batch_size, num_patches, feature_dim
    patch_inputs_emb = torch.randn(B, N, D)
    BOARDERS = torch.randint(0, 2, (B, 2 * N - 1))  # dummy BOARDERS

    model = PatchClusterGATEncoder(in_dim=D, out_dim=128, heads=4)
    s_time = time.time()
    cluster_reprs, updated_BOARDERS = model(patch_inputs_emb, BOARDERS)
    e_time = time.time()

 
    
    B, N, P, D = 384, 24, 4, 64 # batch_size, num_patches, feature_dim
    patch_inputs_emb = torch.randn(B, N, P, D)
    BOARDERS = torch.randint(0, 2, (B, 2 * N - 1))  # dummy BOARDERS

    model = PatchClusterGAT(in_dim=D, out_dim=64, heads=4, min_patch_size=P)
    s_time = time.time()
    cluster_reprs, updated_BOARDERS = model(patch_inputs_emb, BOARDERS)
    e_time = time.time()
    print('Execution time:', e_time - s_time)

    print(cluster_reprs.shape)  # (B, max_clusters, 128)
    print(updated_BOARDERS.shape)  # (B, 2*N - 1)
    # '''

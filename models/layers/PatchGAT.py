import time
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

class SinglePatchGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=1, concat=False):
        super().__init__()
        self.gat = GATConv(in_dim, hidden_dim, heads=heads, concat=concat)

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)


class IntraPatchGATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=1, gat_num_layer=3):
        super(IntraPatchGATEncoder, self).__init__()
        self.gat_layer = SinglePatchGAT(in_dim, hidden_dim, heads)
        
    def build_batch_graph(self, patch_inputs):

        BN, P, d = patch_inputs.shape
        device = patch_inputs.device
        
        global_nodes = torch.zeros([BN, 1, d], device=patch_inputs.device)
        x = torch.cat([patch_inputs, global_nodes], dim=1)     # (BN, P+1, d)
        x = x.view(BN*(P+1), d)                               # (BN*(P+1), d)
        
        patch_offsets = (torch.arange(BN, device=device) * (P+1)).view(-1, 1, 1)
        
        base_idx = torch.arange(P, device=device)
        grid_i, grid_j = torch.meshgrid(base_idx, base_idx, indexing='ij')
        patch_edges = torch.stack([grid_i.flatten(), grid_j.flatten()], dim=1)  # [P*P, 2]
        
        global_node_idxs = torch.full((P,), P, dtype=torch.long, device=device)
        patch_to_global = torch.stack([base_idx, global_node_idxs], dim=1)  # [P, 2]
        
        intra_edges = torch.cat([patch_edges, patch_to_global], dim=0)  # [P*(P+1), 2]
        intra_edges = intra_edges.unsqueeze(0).expand(BN, -1, 2)        # [BN, P*(P+1), 2]
        intra_edges = intra_edges + patch_offsets
        edge_index = intra_edges.reshape(-1, 2).t().contiguous()        # [2, BN*P*(P+1)]
        
        return Data(x=x, edge_index=edge_index)

    def forward(self, patch_inputs):

        B, N, P, d = patch_inputs.shape
        
        patch_inputs = patch_inputs.reshape(B*N, P, d)
        graph = self.build_batch_graph(patch_inputs)
        
        out = self.gat_layer(graph.x, graph.edge_index)  # (B*N*(P+1), hidden_dim)
        
        global_node_indices = torch.arange(P, (B*N+1)*(P+1)-1, P+1, device=out.device)
        patch_reprs = out[global_node_indices].view(B, N, -1)  # (B, N, hidden_dim)
        
        return patch_reprs
   
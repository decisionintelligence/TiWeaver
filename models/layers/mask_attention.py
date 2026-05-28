import torch
from torch import nn
import torch.nn.functional as F


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

        B, V, P, D = x.shape
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(x).view(B, V, P, H, d).transpose(2, 3)  # [B, V, H, P, d]
        k = self.k_proj(x).view(B, V, P, H, d).transpose(2, 3)
        v = self.v_proj(x).view(B, V, P, H, d).transpose(2, 3)

        q = q.unsqueeze(2)  # [B, Vq, 1, H, P, d]
        k = k.unsqueeze(1)  # [B, 1, Vk, H, P, d]
        v = v.unsqueeze(1)  # [B, 1, Vk, H, P, d]

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)

        key_mask = mask.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [B, 1, Vk, 1, 1, Pk]
        attn_logits = attn_logits.masked_fill(~key_mask, float(0))

        self_mask =  torch.eye(V, device=x.device).bool().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [1, Vq, Vk]
        attn_logits = attn_logits.masked_fill(self_mask, float(0))

        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, Vq, Vk, H, Pq, Pk]
        
        attn_out = torch.matmul(attn_weights, v)
        
        attn_out = attn_out.sum(dim=2)  # [B, Vq, H, Pq, Dh]

        attn_out = attn_out.permute(0, 1, 3, 2, 4).contiguous()  # [B, V, P, H, Dh]
        attn_out = attn_out.view(B, V, P, D)  # [B, V, P, D]
        
        out = self.out_proj(attn_out)  # [B, V, P, D]
        
        return out
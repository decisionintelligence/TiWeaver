import torch
from torch import nn
from torch.nn import functional as F

class Unk_odefunc(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(Unk_odefunc, self).__init__()
        self.nfe = 0
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, input_dim)
        )

    def forward(self, t, z):
        self.nfe += 1
        return self.net(z)


class Unk_odefunc_DC(nn.Module):
    # DC: Dependent-channel
    def __init__(self, latent_dim, num_nodes, d_f=32):
        super(Unk_odefunc_DC, self).__init__()
        self.nfe = 0
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.inter_var_MLP = nn.Sequential(
            nn.Linear(latent_dim, d_f),
            nn.Tanh(),
            nn.Linear(d_f, latent_dim)
        )

    def forward(self, t, z):
        """
        仅通过MLP考虑变量间的时序关系
        :param t:
        :param z: B x N*latent_dim
        :return:
        """
        self.nfe += 1
        B = z.shape[0]
        z = z.view(B, self.num_nodes, self.latent_dim)
        z = self.inter_var_MLP(z)
        return z.view(B, self.num_nodes * self.latent_dim)


class Unk_odefunc_ATT(nn.Module):
    def __init__(self, latent_dim, num_nodes, n_heads, device, adj_mask=None, d_f=32):
        super(Unk_odefunc_ATT, self).__init__()
        self.nfe = 0
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        if adj_mask is None:
            self.adj_mask = None
        else:
            self.adj_mask = torch.tensor(adj_mask, dtype=torch.int8,
                            device=device) + torch.eye(self.num_nodes, device=device)

        self.fc = nn.Linear(latent_dim, latent_dim)
        self.spatial_att = nn.MultiheadAttention(latent_dim, num_heads=n_heads, batch_first=True)
        # LayerNorm 模块，用于在注意力输出后进行归一化
        self.layer_norm_1 = nn.LayerNorm(latent_dim)
        self.layer_norm_2 = nn.LayerNorm(latent_dim)
        # residual
        self.residual_1 = nn.Identity()
        self.residual_2 = nn.Identity()

    def forward(self, t, z):
        """
        利用空间Attention考虑变量间的时空关系
        :param t:
        :param z: B x N*latent_dim
        :return:
        """
        self.nfe += 1
        B = z.shape[0]
        z = z.reshape(B, self.num_nodes, self.latent_dim)  # B x N x latent_dim
        # att-add&norm
        # Masked self-attention
        z = self.residual_1(z) + self.spatial_att(z, z, z, attn_mask=self.adj_mask)[0]
        z = self.residual_1(z) + self.spatial_att(z, z, z)[0]
        z = self.layer_norm_1(z)
        # ffd-add&norm
        z = self.residual_2(z) + F.relu(self.fc(z))
        z = self.layer_norm_2(z)

        return z.reshape(B, self.num_nodes * self.latent_dim)


if __name__ == "__main__":
    latent_dim = 128
    n_heads = 4
    Z0 = torch.rand(64, latent_dim)

    unk_odefunc = Unk_odefunc(latent_dim)
    print(unk_odefunc(0, Z0).shape)

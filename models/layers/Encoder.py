from utils.utils import init_network_weights, ConfigDict, split_last_dim
from torch import nn
import torch
from typing import Optional
from torch.nn.modules.rnn import GRU

class Encoder_unk_z(nn.Module):
    def __init__(self, input_dim, latent_dim, rnn_dim, n_layers):
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.rnn_dim = rnn_dim
        self.gru_rnn = GRU(self.input_dim, rnn_dim, num_layers=n_layers)

        # hidden to z0 settings
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(self.rnn_dim, 50),
            nn.Tanh(),
            nn.Linear(50, self.latent_dim))

        init_network_weights(self.hiddens_to_z0)

    def forward(self, X):
        """
        encoder forward pass on t time steps
        :param X: seq_len x B x D
        :return: Z0: B x latent_dim
        """
        outputs, _ = self.gru_rnn(X)  # seq_len x B x rnn_dim

        last_output = outputs[-1]   # B x rnn_dim
        Z0 = self.hiddens_to_z0(last_output)

        return Z0


if __name__ == '__main__':
    T, B, input_dim = 24, 64, 6
    latent_dim = 128
    rnn_dim = 64
    n_layers = 1
    inp = torch.randn(T, B, input_dim)
    encoder = Encoder_unk_z(input_dim, latent_dim, rnn_dim, n_layers)
    out = encoder(inp)
    print(out.shape)
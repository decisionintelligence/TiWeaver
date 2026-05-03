import torch
import torch.nn as nn
from torch.nn import functional as F


class Old_Gated_Fusion(nn.Module):
    def __init__(self, num_nodes, var_dim):
        super(Old_Gated_Fusion, self).__init__()
        self.hid_dim = num_nodes * var_dim
        self.fc = nn.Linear(self.hid_dim, self.hid_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, grad_diff, grad_adv):
        X_diff = self.fc(grad_diff)
        X_adv = self.fc(grad_adv)
        z = self.sigmoid(torch.add(X_diff, X_adv))
        H = torch.add((z * X_diff), ((1 - z) * X_adv))
        return H


class New_Gated_Fusion(nn.Module):
    def __init__(self, num_nodes, var_dim, hid_dim=64):
        super(New_Gated_Fusion, self).__init__()
        self.phy_dim = num_nodes * var_dim

        self.diff_fc = nn.Linear(self.phy_dim, hid_dim)
        self.adv_fc = nn.Linear(self.phy_dim, hid_dim)
        self.gate_fc = nn.Linear(hid_dim * 2, hid_dim)
        self.output_fc = nn.Linear(hid_dim, self.phy_dim)

        self.sigmoid = torch.nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, grad_diff, grad_adv):
        """

        :param grad_diff: B x NDout
        :param grad_adv: B x NDout
        :return: B x NDout
        """
        diff_hidden = self.diff_fc(grad_diff)
        adv_hidden = self.adv_fc(grad_adv)

        combined_hidden = torch.cat((diff_hidden, adv_hidden), dim=1)
        gate_values = self.sigmoid(self.gate_fc(combined_hidden))

        gated_hidden = gate_values * diff_hidden + (1 - gate_values) * adv_hidden

        output = self.output_fc(gated_hidden)
        return output


class Simple_Gated_Fusion(nn.Module):
    def __init__(self, num_nodes, var_dim):
        super(Simple_Gated_Fusion, self).__init__()

        self.num_nodes = num_nodes
        self.var_dim = var_dim
        self.gated_fc = nn.Linear(2, 1)

    def forward(self, grad_diff, grad_adv):
        """

        :param grad_diff: B x NDout
        :param grad_adv: B x NDout
        :return: B x NDout
        """
        B = grad_diff.shape[0]
        grad_diff = grad_diff.reshape(B, self.num_nodes, self.var_dim)
        grad_adv = grad_adv.reshape(B, self.num_nodes, self.var_dim)
        concat = torch.cat((grad_diff, grad_adv), dim=-1)  # B x N x 2
        g = torch.sigmoid(self.gated_fc(concat))

        grad_diff_adv = g * grad_diff + (1 - g) * grad_adv  # B x N x 1

        return grad_diff_adv.reshape(B, self.num_nodes*self.var_dim)
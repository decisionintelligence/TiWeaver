from torch import nn
import torch
from models.layers.Diffeq_solver import DiffeqSolver
from typing import Optional
from models.layers.tools import EncoderAttrs
from models.layers.Encoder import Encoder_unk_z
from models.layers.Decoder import Linear_Decoder
from models.layers.Unk_Dynamics import Unk_odefunc

from utils.utils import ConfigDict, load_config


class Model(nn.Module, EncoderAttrs):
    def __init__(self, args: Optional[ConfigDict] = None):
        nn.Module.__init__(self)
        EncoderAttrs.__init__(self, args)
        self.setting = self.get_setting(args)

        # gpu device
        self.device = torch.device("cuda:{}".format(args.GPU.gpu))

        # ODE solver
        self.ode_method = args.model.odefunc.ode_method
        self.adjoint = args.model.odefunc.adjoint
        self.atol = float(args.model.odefunc.odeint_atol)
        self.rtol = float(args.model.odefunc.odeint_rtol)
        # 损失函数定义
        self.ode_pred = self.args.model.loss.pred_loss
        self.ode_recon = self.args.model.loss.recon_loss

        # 网络结构
        self.encoder = Encoder_unk_z(self.input_dim, self.latent_dim,
                                     self.rnn_dim, self.num_rnn_layers)
        self.unk_odefunc = Unk_odefunc(input_dim=self.latent_dim)  # 数据驱动的未知ODE
        
        self.pred_decoder = Linear_Decoder(input_dim=self.latent_dim,
                                         output_dim=self.output_dim)
        self.recon_decoder = Linear_Decoder(input_dim=self.latent_dim,
                                          output_dim=self.output_dim)

        # 求解器
        self.diffeq_solver = DiffeqSolver(self.ode_method,
                                          odeint_rtol=self.rtol, odeint_atol=self.atol,
                                          adjoint=self.adjoint)

    def forward(self, inputs, labels=None):
        """
        seq2seq forward (数据驱动的动态系统)
        :param inputs: seq_len x B x D
        :param labels: pred_len x B x D
        :return: pred_len x B x D
        """
        inputs = inputs.permute(1, 0, 2)  # B x seq_len x D -> seq_len x B x D
        Z0 = self.encoder(inputs)  # B x D
        unk_z_pred, unk_fe = self.predict(Z0)
        self.pred_y = self.pred_decoder(unk_z_pred)  # T x B x D

        if self.ode_recon:
            unk_z_recon, unk_fe_rev = self.reconstruct(Z0)
            self.recon_x = self.recon_decoder(unk_z_recon)  # pred_len x B x D
        else:
            unk_fe_rev = (0, 0)

        self.total_nfe = unk_fe_rev + unk_fe

        self.pred_y = self.pred_y.permute(1, 0, 2)  # pred_len x B x output_dim -> B x pred_len x output_dim
        return self.pred_y

    def get_loss(self, inputs, labels):
        """
        :param labels: 真实值
        """
        loss = 0
        loss_func = nn.L1Loss()
        if self.args.model.loss.recon_loss:
            loss += self.args.loss.recon_coeff * loss_func(inputs, self.recon_x)
        if self.args.model.loss.pred_loss:
            loss += loss_func(labels, self.pred_y)
        return loss

    def predict(self, Z0):
        time_steps_to_predict = torch.arange(start=0, end=self.pred_len + 1, step=1).float()  # horizon 1 + 24
        time_steps_to_predict = time_steps_to_predict / len(time_steps_to_predict)
        pred_z, fe = self.diffeq_solver.solve(self.unk_odefunc, Z0, time_steps_to_predict)  # T x B x N*D
        pred_z = pred_z[1:]

        return pred_z, fe

    def reconstruct(self, Z0):
        time_steps_to_recon = torch.arange(start=0, end=-self.seq_len, step=-1).float()  # seq_len 24
        time_steps_to_recon = time_steps_to_recon / len(time_steps_to_recon)
        recon_z, rev_fe = self.diffeq_solver.solve(self.unk_odefunc, Z0, time_steps_to_recon)  # seq_len x B x N*D
        recon_z = recon_z.flip(dims=[0])

        return recon_z, rev_fe

    def get_setting(self, args):
        setting = 'NODE_{}_lr{}_loss{}-{}_recon_{}_bs{}_sl{}_pl{}_lat{}_rl{}_rd{}'.format(
            args.data.data_path + "_" + args.data.freq,
            args.train.lr,
            f"{int(args.model.loss.recon_loss)}{int(args.model.loss.pred_loss)}",
            args.model.loss.criterion,
            args.model.loss.recon_coeff,
            args.data.batch_size,
            args.model.seq_len,
            args.model.pred_len,
            args.model.dim.latent,
            args.model.num_rnn_layers,
            args.model.rnn_dim,
        )
        return setting


if __name__ == "__main__":
    config = load_config('../Model_Config/BAOWU/NODE_config.yaml')
    config = ConfigDict(config)
    config.exp_idx = 0
    config.des = 'test'
    model = Model(config)
    X = torch.randn(config.model.seq_len, config.data.batch_size, config.model.input_dim)
    Y = torch.randn(config.model.pred_len, config.data.batch_size, config.model.output_dim)
    print(model(X)[0].shape)
    print(model.get_loss(X, Y))


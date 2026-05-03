import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from models.layers.RevIN import RevIN
# from models.layers.SelfAttention_Family import FullAttention, AttentionLayer
# from models.layers.Transformer_EncDec import Encoder, EncoderLayer
from models.layers.Embed import PositionalEmbedding
# from models.decoder_Layer import Transformer_Decoder
from models.layers.Diffeq_solver import DiffeqSolver
from models.layers.Decoder import Linear_Decoder
from models.layers.Unk_Dynamics import Unk_odefunc
from models.layers.tools import EncoderAttrs
from models.dynamic_encoder_layer import DynamicEncoderLayer

from utils.utils import ConfigDict, load_config


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
    

class Model(nn.Module, EncoderAttrs):
    def __init__(self, args):
        super(Model, self).__init__()
        # EncoderAttrs.__init__(self, args)
        
        self.min_patch_size = args.model.min_patch_size
        self.stride = args.model.min_patch_size
        
        self.threshold = args.model.threshold
        
        self.pred_len = args.model.pred_len
        self.seq_len = args.model.seq_len
        self.latent_dim = int(args.model.dim.latent)
        self.input_dim = int(args.model.enc_in)
        self.output_dim = int(args.model.enc_in)
        self.d_model = args.model.d_model
        
        # RevIn
        self.revin_layer = RevIN(args.model.enc_in, affine=True, subtract_last=False)
        
        # FFT编码
        self.fft_k = args.model.fft_k
        self.hidden_layer = nn.Linear(self.fft_k*2, self.d_model)
        
        self.patch_attention = nn.MultiheadAttention(self.d_model, 1, dropout=0.1, batch_first=True)

        self.position_embedding = PositionalEmbedding(self.d_model)
        
        self.encoder = nn.ModuleList()
        for num in range(args.model.e_layers):
            self.encoder.append(
                DynamicEncoderLayer(
                        d_model=args.model.d_model,
                        d_ff=args.model.d_ff,
                    )
            )
        
        # ODE solver
        self.ode_method = args.model.odefunc.ode_method
        self.adjoint = args.model.odefunc.adjoint
        self.atol = float(args.model.odefunc.odeint_atol)
        self.rtol = float(args.model.odefunc.odeint_rtol)
        # 损失函数定义
        self.ode_pred = args.model.loss.pred_loss
        self.ode_recon = args.model.loss.recon_loss

        # 网络结构
        # self.encoder = Encoder_unk_z(self.input_dim, self.latent_dim,
        #                              self.rnn_dim, self.num_rnn_layers)
        self.unk_odefunc = Unk_odefunc(input_dim=self.latent_dim)  # 数据驱动的未知ODE
        
        self.pred_decoder = Linear_Decoder(input_dim=self.latent_dim,
                                         output_dim=1)
        self.recon_decoder = Linear_Decoder(input_dim=self.latent_dim,
                                          output_dim=1)

        # 求解器
        self.diffeq_solver = DiffeqSolver(self.ode_method,
                                          odeint_rtol=self.rtol, odeint_atol=self.atol,
                                          adjoint=self.adjoint)
        
        # self.decoder = Transformer_Decoder(
        #     d_model=args.model.d_model,
        #     d_ff=args.model.d_ff,
        #     # factorized=True,
        #     layer_number=args.model.layer_number,
        # )
        
        # self.output_layer = nn.Linear(self.d_model, self.pred_len)

        self.setting = self.get_setting(args)
        
    def get_setting(self, args):
        setting = 'DynamicNODE_{}_{}_lr{}_df{}_dm{}_{}_sl{}_pl{}_t{}'.format(
            args.data.data_path,
            args.data.freq,
            args.train.lr,
            args.model.d_ff,
            args.model.d_model,
            args.model.loss.criterion,
            args.model.seq_len,
            args.model.pred_len,
            args.model.threshold,
        )
        return setting
        
    def forward(self, x, x_mark=None):
        # x: [batch_size, seq_len, n_vars]
        B, T, K = x.shape
        
        # Normalization from Non-stationary Transformer
        x = self.revin_layer(x, 'norm')
        
        # 按照 min_patch_size 进行 patching
        if T < self.min_patch_size:
            padding = torch.zeros([B, self.min_patch_size - T, K]).to(x.device)
            x = torch.cat((x, padding), dim=1)
        all_patch_inputs = x.unfold(dimension=1, size=self.min_patch_size, step=self.min_patch_size)  # inputs: [bs x num_patch x n_vars x min_patch_size]
        
        # 对每个patch进行FFT变换
        all_patch_inputs_fft = temporal_to_frequency(all_patch_inputs, k=self.fft_k)  # [bs x num_patch x n_vars x 2k]
        all_patch_inputs_fft_embs = self.hidden_layer(all_patch_inputs_fft)  # [bs x num_patch x n_vars x hidden_dim]
        
        # Patching
        enc_outs = []
        for b in range(B):
            patch_inputs = all_patch_inputs_fft[b, :, :, :].unsqueeze(0)  # [num_patch x n_vars x hidden_dim]
            patch_inputs_fft_embs = all_patch_inputs_fft_embs[b, :, :, :].unsqueeze(0)  # [num_patch x n_vars x hidden_dim]
            dynamic_patch_embs = []
            for k in range(K):
                # 不同变量的时序特征不一致，因此需要分别进行动态patching
                patch_inputs_fft_emb = patch_inputs_fft_embs[:, :, k, :] # [num_patch x hidden_dim]
                
                # dynamic patching
                num_patches = patch_inputs.shape[1]
                ATTNS = torch.full((num_patches,), self.threshold).to(x.device)
                while True:
                    BOARDERS = torch.ones([2*num_patches-1])
                    num_boarders = num_patches - 1
                    # patch_pos_emb = self.position_embedding(patch_inputs)
                    # 使用自注意力计算每个patch之间的关系
                    # dynamic_patch_emb, attns = self.encoder(patch_inputs_fft_emb+patch_pos_emb)
                    # dynamic_patch_emb, attns = self.encoder(patch_inputs_fft_emb)
                    dynamic_patch_emb, attns = self.patch_attention(patch_inputs_fft_emb, patch_inputs_fft_emb, patch_inputs_fft_emb)
                    
                    attns = attns[0].squeeze(1).squeeze(0)  # [bs x num_boarders x num_boarders]
                    # 根据attns合并不同的patch
                    i = 0
                    while i < num_boarders:
                        if attns[i][i+1] < self.threshold:
                            BOARDERS[2*i+1] = 0
                        else:
                            if attns[i][i+1] < ATTNS[i] or attns[i][i+1] < ATTNS[i+1]:
                                BOARDERS[2*i+1] = 0
                            elif i < num_boarders-1:
                                if attns[i+1][i+2] < self.threshold:
                                    BOARDERS[2*(i+1)+1] = 0
                                else: # 考虑三个相邻patch如何合并？
                                    idx = torch.argmax(torch.tensor([ATTNS[i], ATTNS[i+1], attns[i][i+1]]))
                                    idx = 2*(i+idx)+1
                                    if idx < len(BOARDERS):
                                        BOARDERS[idx] = 0
                                i+=1
                        i+=1
                    
                    # 根据BOARDERS合并patches
                    merged_patches = []
                    merged_patches_fft = []
                    merged_ATTNS = []
                    i = 0
                    while i < num_patches:
                        # 找到连续需要合并的patches
                        merge_count = 0
                        while i + merge_count < num_patches - 1 and BOARDERS[2*(i+merge_count)+1] == 1:
                            merge_count += 1
                        
                        if merge_count > 0:
                            # 合并连续的patches
                            patches_to_merge = [all_patch_inputs[b,i+j,k,:].unsqueeze(0) for j in range(merge_count + 1)]
                            merged_patch = torch.cat(patches_to_merge, dim=1)
                            # merged_patches.append(merged_patch)
                            merge_patch_fft = temporal_to_frequency(merged_patch, k=self.fft_k)  # [bs x 2k]
                            merge_patch_fft_emb = self.hidden_layer(merge_patch_fft)  # [bs x hidden_dim]
                            merged_patches_fft.append(merge_patch_fft_emb)
                            merged_ATTNS.append(attns[i, i:i+merge_count+1].sum().mean())
                            i += merge_count + 1
                        else:
                            # 保持当前patch不变
                            # merged_patches.append(all_patch_inputs[b,i,k,:].unsqueeze(0))
                            merged_patches_fft.append(patch_inputs_fft_emb[:,i,:])
                            merged_ATTNS.append(attns[i, i])
                            i += 1
                    
                    # 更新patch_inputs_fft_emb为合并后的结果
                    patch_inputs_fft_emb = torch.stack(merged_patches_fft, dim=1)
                    # patch_inputs = torch.stack(merged_patches, dim=1)
                    ATTNS = torch.stack(merged_ATTNS, dim=0)
                    
                    num_boarders = BOARDERS.sum().item() - num_patches
                    num_patches = patch_inputs_fft_emb.shape[1]
                    # 当只剩下一个patch或者所有patch间都存在空隙时，停止patching；很容易出现merge为1个patch的问题，需要解决
                    
                    if num_patches == 1 or num_boarders == 0:
                    # if num_boarders == 1:
                        break
                if b == 0:
                    print('num_patches', num_patches)
                    print(BOARDERS)
                    print(patch_inputs_fft_emb.shape)
                    print("--------------")
                dynamic_patch_embs.append(dynamic_patch_emb)
                
            # dynamic_patch_emb n_vars * [bs x num_patches' x hidden_dim]
            # var_attention encoder
            out = dynamic_patch_embs
            for layer in self.encoder:
                out = layer(out) # [bs x patch_num' x hidden_dim]

            out_tensor = torch.stack([out[i][:, -1] for i in range(len(out))], dim=0)  # [K, B, hidden_dim]
            enc_outs.append(out_tensor.permute(1, 0, 2))  # [B, K, hidden_dim]
            
        # 合并所有变量的输出
        enc_out = torch.stack(enc_outs, dim=0).squeeze(1)  # [bs x n_vars x hidden_dim]
        # dec_out = self.output_layer(dec_out)  # [bs x n_vars x seq_len]
        # dec_out = dec_out.permute(0, 2, 1)  # [bs x pred_len x n_vars]
            
        # NODE
        # inputs = inputs.permute(1, 0, 2)  # B*var x seq_len x D -> seq_len x B*var x D
        # Z0 = self.encoder(inputs)  # B*var x D
        Z0 = rearrange(enc_out, 'b k d -> (b k) d')
        
        unk_z_pred, unk_fe = self.predict(Z0) # pred_len x B*var x D
        unk_z_pred = rearrange(unk_z_pred, 'p (b k) d -> b p k d', b=B, k=K, p=self.pred_len) # B x pred_len x n_vars x D
        # unk_z_pred = unk_z_pred.permute(1, 0, 2)  # B*var x pred_len x D
        pred_y = self.pred_decoder(unk_z_pred).squeeze(-1)  # # B x pred_len x n_vars
        # pred_y = pred_y.reshape(B, K, -1)  # B x pred_len x n_vars

        if self.ode_recon:
            unk_z_recon, unk_fe_rev = self.reconstruct(Z0)
            recon_x = rearrange(unk_z_recon, 'p (b k) d -> b p k d', b=B, k=K, p=self.seq_len)  # B x seq_len x n_vars x D
            recon_x = self.recon_decoder(recon_x).squeeze(-1)  # seq_len x B*var x D
            
        # else:
        #     unk_fe_rev = (0, 0)

        # self.total_nfe = unk_fe_rev + unk_fe

        # self.pred_y = self.pred_y.permute(1, 0, 2)  # pred_len x B x output_dim -> B x pred_len x output_dim
            
        pred_y = self.revin_layer(pred_y, 'denorm')
        recon_x = self.revin_layer(recon_x, 'denorm')
        
        return pred_y[:, -self.pred_len :, :], recon_x
    
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

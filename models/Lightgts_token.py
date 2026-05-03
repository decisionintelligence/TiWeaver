import torch
from torch import nn
import torch.nn.functional as F

from models.layers.Embed import PatchEmbedding
from models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.layers.Transformer_EncDec import Encoder, EncoderLayer
from models.layers.RevIN import RevIN
from models.layers.resample_emb import resample_patchemb

class decoder_PredictHead(nn.Module):
    def __init__(self, d_model, target_patch_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, target_patch_len)

    def forward(self, x, patch_len, finetune=True):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """
        # if finetune:
        linear = nn.Linear(self.d_model, patch_len, bias=False)
        linear.weight.data = resample_patchemb(old=self.linear.weight.data.T, new_patch_len=patch_len).T

        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        # x = resize(x, target_patch_len=patch_len)
        x = x.permute(0,2,3,1)                  # [bs x num_patch x  x patch_len x nvars]
        return x.reshape(x.shape[0],-1,x.shape[3])


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    # def __init__(self, config, patch_len=16, stride=8):
    def __init__(self, args):

        super(Model, self).__init__()
        config = args.model
        self.task_name = "long_term_forecast"
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        # self.patch_len = config.patch_len
        self.d_model = config.d_model
        self.stride = config.stride
        # padding = self.stride
        
        self.target_patch_len = args.model.target_patch_size
        self.embedding = nn.Linear(self.target_patch_len, config.d_model)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, 1, config.d_model),requires_grad=True)
        self.fft_k = config.k
        # self.finetune = ~config.finetune
        

        # RevIn
        self.revin_layer = RevIN(config.enc_in, affine=True, subtract_last=False)

        # patching and embedding
        # self.patch_embedding = PatchEmbedding(
        #     config.d_model, self.patch_len, self.stride, padding, config.dropout
        # )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )

        # Prediction Head
        # self.head_nf = config.d_model * int((config.seq_len - self.patch_len) / self.stride + 2)
        # if (
        #     self.task_name == "long_term_forecast"
        #     or self.task_name == "short_term_forecast"
        # ):
        #     self.head = FlattenHead(
        #         config.enc_in,
        #         self.head_nf,
        #         config.pred_len,
        #         head_dropout=config.dropout,
        #     )
        self.head = decoder_PredictHead(config.d_model,
                                        self.target_patch_len, 
                                        config.dropout)
        
        self.setting = self.get_setting(args)
        
    def get_setting(self, args):
        setting = 'tst_token_test_{}_{}_fttk{}_lr{}_tp{}_loss{}_sl{}_pl{}'.format(
            args.data.data_path,
            args.data.freq,
            args.model.k,
            args.train.lr,
            args.model.target_patch_size,
            args.model.loss.criterion,
            args.model.seq_len,
            args.model.pred_len,
        )
        return setting
            
    def forward(self, x_enc):
        B, seq_len, n_vars = x_enc.shape
        
        patch_len = FFT_for_Period(x_enc, self.fft_k)[0]
        num_patch = (max(self.seq_len, patch_len)-patch_len) // patch_len + 1    
        tgt_len = patch_len  + patch_len*(num_patch-1)
        s_begin = self.seq_len - tgt_len
        # Normalization from Non-stationary Transformer
        x_enc = self.revin_layer(x_enc, 'norm') # 可逆正则化

        # do patching

        if seq_len < patch_len:
            padding = torch.zeros([B, patch_len - seq_len, n_vars]).to(x_enc.device)
            x_enc = torch.cat((x_enc, padding), dim=1)
        patch_inputs = x_enc[:, s_begin:, :]  # inputs: [bs x tgt_len x nvars]
        z = patch_inputs.unfold(dimension=1, size=patch_len, step=patch_len)  # inputs: [bs x num_patch x n_vars x patch_len]

        # z = resize(z, target_patch_len=self.target_patch_len)
        
        # tokenizer
        cls_tokens = self.cls_embedding.expand(B, n_vars, -1, -1)
        
        embedding = nn.Linear(patch_len, self.d_model, bias=False)
        embedding.weight.data = resample_patchemb(old=self.embedding.weight.data, new_patch_len=patch_len)
        z = embedding(z).permute(0,2,1,3) # [bs x n_vars x num_patch x d_model]
        z = torch.cat((cls_tokens, z), dim=2)  # [bs x n_vars x (1 + num_patch) x d_model]
        # z = self.drop_out(z + self.pos[:1 + self.num_patch, :])

        # encoder 
        enc_out = torch.reshape(z, (-1, 1 + num_patch, self.d_model)) # [bs*n_vars x num_patch x d_model]
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out, patch_len)  # z: [bs x nvars x target_window]
        # dec_out = dec_out.permute(0, 2, 1)

        dec_out = self.revin_layer(dec_out, 'denorm')
        # dec_out = dec_out.permute(0,2,1)
        return dec_out[:, :self.pred_len, :] 


def resize(x, target_patch_len):
    '''
    x: tensor [bs x num_patch x n_vars x patch_len]]
    '''
    bs, num_patch, n_vars, patch_len = x.shape
    x = x.reshape(bs*num_patch, n_vars, patch_len)
    x = F.interpolate(x, size=target_patch_len, mode='linear', align_corners=False)
    return x.reshape(bs, num_patch, n_vars, target_patch_len)

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    # 计算周期: 序列长度除以频率索引
    # 例如: 如果序列长度为100, top_list=[2,4], 则period=[50,25]
    # 这表示序列中每50个时间步和每25个时间步分别有一个主要周期
    period = x.shape[1] // top_list
    return period
    return period, abs(xf).mean(-1)[:, top_list]
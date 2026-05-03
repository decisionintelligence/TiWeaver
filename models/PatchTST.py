import torch
from torch import nn

from models.layers.Embed import PatchEmbedding
from models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.layers.Transformer_EncDec import Encoder, EncoderLayer
from models.layers.RevIN import RevIN

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
        self.patch_len = config.patch_len
        self.stride = config.stride
        padding = self.stride
        # print(config.enc_in)

        # RevIn
        self.revin_layer = RevIN(config.enc_in, affine=True, subtract_last=False)

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            config.d_model, self.patch_len, self.stride, padding, config.dropout
        )

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
        self.head_nf = config.d_model * int((config.seq_len - self.patch_len) / self.stride + 2)
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.head = FlattenHead(
                config.enc_in,
                self.head_nf,
                config.pred_len,
                head_dropout=config.dropout,
            )
        
        self.setting = self.get_setting(args)
        
    def get_setting(self, args):
        setting = 'PatchTST_{}_{}_data{}_lr{}_loss{}_sl{}_pl{}'.format(
            args.data.data_path,
            args.data.freq,
            args.train.lr,
            f"{int(args.model.loss.pred_loss)}",
            args.model.loss.criterion,
            args.model.seq_len,
            args.model.pred_len,
        )
        return setting
            
    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        x_enc = self.revin_layer(x_enc, 'norm') # 可逆正则化

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = self.revin_layer(dec_out, 'denorm')
        # dec_out = dec_out.permute(0,2,1)
        return dec_out

    # def forward(self, x_enc):
    def forward(self, x_enc, x_mark=None, y_mark=None, x_mask=None, y_mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]

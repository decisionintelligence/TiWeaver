__all__ = ["PDF"]

# Cell
from typing import Optional

from torch import nn
from torch import Tensor
import torch.nn.functional as F
from models.layers.PDF_backbone import PDF_backbone


class Model(nn.Module):
    def __init__(
        self,
        args,
        max_seq_len: Optional[int] = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type="flatten",
        verbose: bool = False,
        **kwargs
    ):

        super().__init__()
        configs = args
        # load parameters
        c_in = configs.model.enc_in
        context_window = configs.model.seq_len
        target_window = configs.model.pred_len

        n_layers = configs.model.e_layers
        n_heads = configs.model.n_heads
        d_model = configs.model.d_model
        d_ff = configs.model.d_ff
        dropout = configs.model.dropout
        fc_dropout = configs.model.fc_dropout
        head_dropout = configs.model.head_dropout

        individual = configs.model.individual

        add = configs.model.add
        wo_conv = configs.model.wo_conv
        serial_conv = configs.model.serial_conv

        kernel_list = configs.model.kernel_list
        patch_len = configs.model.patch_len
        period = configs.model.period
        stride = configs.model.stride

        padding_patch = configs.model.padding_patch

        revin = configs.model.revin
        affine = configs.model.affine
        subtract_last = configs.model.subtract_last

        # models
        self.model = PDF_backbone(
            c_in=c_in,
            context_window=context_window,
            target_window=target_window,
            wo_conv=wo_conv,
            serial_conv=serial_conv,
            add=add,
            patch_len=patch_len,
            kernel_list=kernel_list,
            period=period,
            stride=stride,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            pretrain_head=pretrain_head,
            head_type=head_type,
            individual=individual,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            verbose=verbose,
            **kwargs
        )
        self.setting = self.get_setting(args)
        
    def get_setting(self, args):
        setting = 'PDF_{}_{}_data{}_lr{}_loss{}_sl{}_pl{}_{}'.format(
            args.data.data_path,
            args.data.freq,
            args.train.lr,
            f"{int(args.model.loss.pred_loss)}",
            args.model.loss.criterion,
            args.model.seq_len,
            args.model.pred_len,
            args.des
        )
        return setting

    # def forward(self, x):  # x: [Batch, Input length, Channel]
    def forward(self, x, x_mark=None, y_mark=None, x_mask=None, y_mask=None):
        x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x
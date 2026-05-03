from einops import rearrange
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args):
        
        super().__init__()
        self.model = nn.Linear(in_features=args.model.seq_len, out_features=args.model.pred_len)
        self.setting = self.get_setting(args)
        
    def get_setting(self, args):
        setting = 'LR{}_data{}_lr{}_loss{}_sl{}_pl{}'.format(
            args.data.data_path,
            args.train.lr,
            f"{int(args.model.loss.pred_loss)}",
            args.model.loss.criterion,
            args.model.seq_len,
            args.model.pred_len,
        )
        return setting


    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        x = rearrange(x, 'b l f -> b f l') 
        outputs = self.model(x)
        outputs = outputs.permute(0,2,1)   
        # outputs = rearrange(outputs, '(b f) l -> b l f', b=128)
        return outputs


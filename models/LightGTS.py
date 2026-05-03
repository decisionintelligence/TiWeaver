from einops import rearrange
import torch.nn as nn
import torch
from models.layers.RevIN import RevIN
from models.layers.LightGTS import LightGTS as LightGTS_predict
from models.layers.LightGTS_resample import LightGTS as LightGTS_finetune

class Model(nn.Module):
    def __init__(self, args):
        
        super().__init__()

        # self.prediction_length = args.model.pred_len
        self.patch_len = args.model.patch_len
        self.stride = args.model.stride
        # self.n_embedding = args.model.n_embedding
        
        num_patch = (max(args.model.seq_len, args.model.patch_len)-args.model.patch_len) // args.model.stride + 1    
        tgt_len = self.patch_len  + self.stride*(num_patch-1)
        self.s_begin = args.model.seq_len - tgt_len
        
        self.revin = RevIN(num_features=args.model.c_in)
        assert args.model.head_type in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'
        
        ckpt_path = f"models/lightgts_checkpoints/LightGTS_{args.model.size}.pth"
        LightGTS = LightGTS_finetune if not args.model.zero_shot else LightGTS_predict
        print('finetuning' if not args.model.zero_shot else 'zero-shot')
        self.model = LightGTS(c_in=args.model.c_in,
                target_dim=args.model.pred_len,
                patch_len=args.model.patch_len,
                stride=args.model.stride,
                num_patch=num_patch,
                e_layers=args.model.e_layers,
                d_layers=args.model.d_layers,
                n_heads=args.model.n_heads,
                d_model=args.model.d_model,
                shared_embedding=True,
                d_ff=args.model.d_ff,                        
                dropout=args.model.dropout,
                attn_dropout=args.model.attn_dropout,
                head_dropout=args.model.head_dropout,
                act='relu',
                head_type=args.model.head_type,
                res_attention=False,
                learn_pe=False
                )
        self.model = self.transfer_weights(ckpt_path, self.model, exclude_head=False)

        self.setting = self.get_setting(args)
        
    def get_setting(self, args):
        setting = 'LightGTS_zero{}_{}_{}_lr{}_ps{}_loss{}_sl{}_pl{}'.format(
            args.model.zero_shot,
            args.data.data_path,
            args.data.freq,
            args.train.lr,
            args.model.patch_len,
            args.model.loss.criterion,
            args.model.seq_len,
            args.model.pred_len,
        )
        return setting

    def transfer_weights(self, weights_path, model, exclude_head=True, device='cuda:0'):
        # state_dict = model.state_dict()
        # new_state_dict = torch.load(ckpt_path, map_location=device)
        new_state_dict = torch.load(weights_path, map_location=device)
        if 'model' in new_state_dict.keys():
            new_state_dict=new_state_dict['model']
        #print('new_state_dict',new_state_dict)
        matched_layers = 0
        unmatched_layers = []
        for name, param in model.state_dict().items():        
            if exclude_head and 'head' in name: continue
            if name in new_state_dict:            
                matched_layers += 1
                input_param = new_state_dict[name]
                if input_param.shape == param.shape: param.copy_(input_param)
                else: unmatched_layers.append(name)
            else:
                unmatched_layers.append(name)
                pass # these are weights that weren't in the original model, such as a new head
        if matched_layers == 0: raise Exception("No shared weight names were found between the models")
        else:
            if len(unmatched_layers) > 0:
                print(f'check unmatched_layers: {unmatched_layers}')
            else:
                print(f"weights from {weights_path} successfully transferred!\n")
        model = model.to(device)
        return model

    def freeze(self):
        """ 
        freeze the model head
        require the model to have head attribute
        """
        if hasattr(self.model, 'head'): 
            for param in self.model.parameters(): param.requires_grad = True         
            
    def unfreeze(self):
        for param in self.model.parameters(): param.requires_grad = True
        for param in self.model.encoder.parameters(): param.requires_grad = True


    
    def forward(self, inputs):        
        """
        inputs: [bs x seq_len x n_vars]
        """
        B, seq_len, K = inputs.shape
        
        revin_inputs = self.revin(inputs, 'norm')
        
        # num_patch = (max(seq_len, self.patch_len)-self.patch_len) // self.stride + 1
        
        if seq_len < self.patch_len:
            padding = torch.zeros([B, self.patch_len - seq_len, K]).to(inputs.device)
            inputs = torch.cat((inputs, padding), dim=1)
        patch_inputs = revin_inputs[:, self.s_begin:, :]  # inputs: [bs x tgt_len x nvars]
        patch_inputs = patch_inputs.unfold(dimension=1, size=self.patch_len, step=self.stride)  # inputs: [bs x num_patch x n_vars x patch_len]

        pred = self.model(patch_inputs)
            
        output = self.revin(pred, 'denorm')
        return output


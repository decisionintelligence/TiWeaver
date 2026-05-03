from einops import rearrange
import torch.nn as nn
import torch
from models.layers.RevIN import RevIN
from models.layers.ROSE_lowrank import ROSE as ROSE_finetune
from models.layers.ROSE_predict import ROSE as ROSE_predict

class Model(nn.Module):
    def __init__(self, args):
        
        super().__init__()
        config = args.model
        # self.prediction_length = config.pred_len
        self.prediction_length = 96
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.n_embedding = config.n_embedding
        num_patch = (max(512, config.patch_len)-config.patch_len) // config.stride + 1    
        # num_patch = (max(config.seq_len, config.patch_len)-config.patch_len) // config.stride + 1    
        self.revin = RevIN(num_features=config.enc_in)
        
        if not config.zero_shot:
            print('finetuning')
            self.model = ROSE_finetune(c_in=config.enc_in,
                        target_dim=96,
                        patch_len=config.patch_len,
                        stride=config.stride,
                        n_embedding=config.n_embedding,
                        num_patch=num_patch,
                        n_layers=config.n_layers,
                        n_heads=config.n_heads,
                        d_model=config.d_model,
                        shared_embedding=True,
                        d_ff=config.d_ff,                        
                        dropout=config.dropout,
                        head_dropout=config.head_dropout,
                        # norm ='LayerNorm',
                        act='relu',
                        head_type=config.head_type,
                        res_attention=False
                        )    
            ckpt_path = "models/rose_checkpoints/full-shot.pth"
        else:
            print('zero-shot')
            self.model = ROSE_predict(c_in=config.enc_in,
                target_dim=96,
                patch_len=config.patch_len,
                stride=config.stride,
                n_embedding=config.n_embedding,
                num_slots=config.num_slots,
                num_patch=num_patch,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                d_model=config.d_model,
                shared_embedding=True,
                d_ff=config.d_ff,                        
                dropout=config.dropout,
                head_dropout=config.head_dropout,
                norm='BatchNorm',
                act='relu',
                head_type=config.head_type,
                res_attention=False
                )    
            ckpt_path = "models/rose_checkpoints/zero-shot.pth"
        self.model = self.transfer_weights(ckpt_path, self.model, exclude_head=False)

        
        self.setting = self.get_setting(args)
        
    def get_setting(self, args):
        setting = 'ROSE_{}_{}_{}_data{}_lr{}_loss{}_sl{}_pl{}'.format(
            args.model.zero_shot,
            args.data.data_path,
            args.data.freq,
            args.train.lr,
            f"{int(args.model.loss.pred_loss)}",
            args.model.loss.criterion,
            args.model.seq_len,
            args.model.pred_len,
        )
        return setting

    def transfer_weights(self, ckpt_path, model, exclude_head=True):
        # state_dict = model.state_dict()
        # new_state_dict = torch.load(ckpt_path, map_location=device)
        new_state_dict = torch.load(ckpt_path)
        new_state_dict=new_state_dict['model']
        #print('new_state_dict',new_state_dict)
        matched_layers = 0
        m_layers = []
        unmatched_layers = []
        for name, param in model.state_dict().items():        
            if exclude_head and 'head' in name: continue
            # if 'layers.2' in name or 'layers.1' in name: 
            #     print(name)
            #     continue          
            if name in new_state_dict:            
                matched_layers += 1          
                input_param = new_state_dict[name]
                if input_param.shape == param.shape: 
                    param.copy_(input_param)
                    m_layers.append(name)
                else: unmatched_layers.append(name)
            else:
                unmatched_layers.append(name)
                pass # these are weights that weren't in the original model, such as a new head
        if matched_layers == 0:
            print(f'matched)layers:{m_layers}')
            print(f'check unmatched_layers: {unmatched_layers}') 
            raise Exception("No shared weight names were found between the models")
            
        return model

    def freeze(self):
        """ 
        freeze the model head
        require the model to have head attribute
        """
        if hasattr(self.model, 'head_720'): 
            # print('model head is available')
            for param in self.model.parameters(): param.requires_grad = False
            for param in self.model.head_720.parameters(): param.requires_grad = True
            # self.model.task_token_prompt.requires_grad = True
            self.model.u.requires_grad = True
            self.model.v.requires_grad = True
        if hasattr(self.model, 'head_336'): 
            # print('model head is available')
            for param in self.model.parameters(): param.requires_grad = False
            for param in self.model.head_336.parameters(): param.requires_grad = True
            # self.model.task_token_prompt.requires_grad = True
            self.model.u.requires_grad = True
            self.model.v.requires_grad = True
        if hasattr(self.model, 'head_192'): 
            # print('model head is available')
            for param in self.model.parameters(): param.requires_grad = False
            for param in self.model.head_192.parameters(): param.requires_grad = True
            # self.model.task_token_prompt.requires_grad = True
            self.model.u.requires_grad = True
            self.model.v.requires_grad = True
        if hasattr(self.model, 'head_96'): 
            # print('model head is available')
            for param in self.model.parameters(): param.requires_grad = False
            for param in self.model.head_96.parameters(): param.requires_grad = True
            # self.model.task_token_prompt.requires_grad = True
            self.model.u.requires_grad = True
            self.model.v.requires_grad = True
            print('model is frozen except the head')        
            
    def unfreeze(self):
        for param in self.model.parameters(): param.requires_grad = True
        for param in self.model.vq_embedding.parameters(): param.requires_grad = False


    def forward(self, inputs, x_mark=None, y_mark=None, x_mask=None, y_mask=None, zero_shot=True):
    # def forward(self, inputs, zero_shot=True):        
        """
        inputs: [bs x seq_len x n_vars]
        """
        first_step = inputs[:, 0:1, :] 

        # 2. 将其复制 10 次
        # expand 不会复制内存，repeat 会复制。这里用 repeat 确保物理存储连续，适合后续计算
        # 结果形状：[B, 10, K]
        first_padding = first_step.repeat(1, 414, 1)

        # 3. 在第二个维度 (dim=1) 拼接
        # 结果形状：[B, 10 + seq_len, K]
        inputs = torch.cat([first_padding, inputs], dim=1)

        B, seq_len, K = inputs.shape
        
        revin_inputs = self.revin(inputs, 'norm')
        
        num_patch = (max(seq_len, self.patch_len)-self.patch_len) // self.stride + 1
        tgt_len = self.patch_len  + self.stride*(num_patch-1)
        s_begin = seq_len - tgt_len
            
        patch_inputs = revin_inputs[:, s_begin:, :]  # inputs: [bs x tgt_len x nvars]
        patch_inputs = patch_inputs.unfold(dimension=1, size=self.patch_len, step=self.stride)  # inputs: [bs x num_patch x n_vars x patch_len]

        output, xe, xq = 0, 0, 0
        if self.n_embedding!=0:
            # pred_96, pred_192, pred_336, pred_720, xe, xq = self.model(patch_inputs)
            
            if zero_shot:
                pred_96, pred_192, pred_336, pred_720, res, xe, xq = self.model(patch_inputs)
            else:
                pred_96, pred_192, pred_336, pred_720, xe, xq = self.model(patch_inputs)
                
            if self.prediction_length ==96:
                pred = pred_96
            elif self.prediction_length ==192:
                pred = pred_192
            elif self.prediction_length ==336:
                pred = pred_336
            elif self.prediction_length ==720:
                pred = pred_720
            # return output, xe, xq
        else:
            pred = self.model(patch_inputs)
            
        output = self.revin(pred, 'denorm')

        return output[:, :33, :], xe, xq


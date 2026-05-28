import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from Data_Provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader


data_dict = {
    'weather': Dataset_Custom,
    'zafnoo': Dataset_Custom,
    'exchange': Dataset_Custom,
}


def data_provider(args, flag):
    data_args = args.data
    model_args = args.model
    Data = data_dict[data_args.data_name]
    root_path = data_args.root_path
    data_path = data_args.data_path
    split_dataset=data_args.split_dataset
    seq_len = model_args.seq_len
    pred_len = model_args.pred_len
    
    if flag == 'train':
        shuffle_flag = True
        drop_last = True
    elif flag == 'val':
        shuffle_flag = True
        drop_last = False
    else:
        shuffle_flag = False
        drop_last = False

    batch_size = data_args.batch_size
    data_set = Data(
        root_path=root_path,
        data_path=data_path,
        seq_len=seq_len,
        pred_len=pred_len,
        split_dataset=split_dataset,
        flag=flag,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=data_args.num_workers,
        drop_last=drop_last)
    
    return data_set, data_loader

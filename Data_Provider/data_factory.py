import os
import sys
# 获取当前文件所在目录的上一级目录，即项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from Data_Provider.data_loader import Dataset_BAOWU, Dataset_BAOWU_irregular, Dataset_regular
from Data_Provider.data_loader_regular import Dataset_ETTm1_ir, Dataset_Custom, Dataset_ETTh1_ir, Dataset_Custom_ir
from torch.utils.data import DataLoader
import torch
from types import SimpleNamespace


data_dict = {
    'BAOWU': Dataset_BAOWU,
    'BAOWU_irregular': Dataset_BAOWU_irregular,
    'regular': Dataset_regular,
    'weather': Dataset_Custom,
    'solar': Dataset_Custom,
    'zafnoo': Dataset_Custom,
    'exchange': Dataset_Custom,
    'ETTh1': Dataset_ETTh1_ir,
    'weather_ir': Dataset_Custom_ir,
    'solar_ir': Dataset_Custom_ir,
    'zafnoo_ir': Dataset_Custom_ir,
    'exchange_ir': Dataset_Custom_ir,
    'ETTm1': Dataset_ETTm1_ir,
    'electricity': Dataset_Custom_ir,
}



def data_provider(args, flag):
    data_args = args.data
    model_args = args.model
    Data = data_dict[data_args.data_name]
    root_path = data_args.root_path
    data_path = data_args.data_path
    freq = data_args.freq
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

    if data_args.data_name == 'BAOWU_irregular':
        data_set = Data(
            root_path=root_path,
            data_path=data_path,
            seq_len=seq_len,
            pred_len=pred_len,
            freq=freq,
            # split_times=split_dataset,
            flag=flag,
            target_num=data_args.target_num,
        )
    else:
        data_set = Data(
            root_path=root_path,
            data_path=data_path,
            seq_len=seq_len,
            pred_len=pred_len,
            freq=freq,
            split_dataset=split_dataset,
            flag=flag,
            target_num=data_args.target_num,
        )

    # print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=data_args.num_workers,
        drop_last=drop_last)
    
    # if data_args.data_name == 'BAOWU':
    #     data_loader = DataLoader(
    #         data_set,
    #         batch_size=batch_size,
    #         shuffle=shuffle_flag,
    #         num_workers=data_args.num_workers,
    #         drop_last=drop_last)
    # else:
    #     data_loader = DataLoader(
    #         data_set,
    #         batch_size=batch_size,
    #         shuffle=shuffle_flag,
    #         num_workers=data_args.num_workers,
    #         collate_fn=patch_variable_time_collate_fn,
    #         drop_last=drop_last)
    return data_set, data_loader


if __name__ == "__main__":

    args = SimpleNamespace(
        data=SimpleNamespace(
            data_name="BAOWU",
            root_path="/home/BAOWU_TS/dataset/align_dataset",
            data_path="/home/BAOWU_TS/dataset/align_dataset/merged_10s.csv",
            freq="30s",
            split_dataset=[0.7, 0.1, 0.2],
            batch_size=32,
            num_workers=4
        ),
        model=SimpleNamespace(
            seq_len=96,
            pred_len=96
        )
    )
    data_set, data_loader = data_provider(args,flag='train')
 
    # 测试 data_loader：取出第一个 batch 并打印张量形状
    for batch in data_loader:
        b_seq_x, b_seq_y, b_seq_x_mark, b_seq_y_mark = batch
        print("Batch seq_x shape:", torch.tensor(b_seq_x).shape)
        print("Batch seq_y shape:", torch.tensor(b_seq_y).shape)
        print("Batch seq_x_mark shape:", torch.tensor(b_seq_x_mark).shape)
        print("Batch seq_y_mark shape:", torch.tensor(b_seq_y_mark).shape)
        break  # 只打印第一个 batch


























def patch_variable_time_collate_fn(batch, args, device=torch.device("cpu"), data_type="train", 
                                     data_min=None, data_max=None, time_max=None):
   

    D = batch[0][2].shape[1]  
    
    combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    n_observed_tp = torch.lt(combined_tt, args.history).sum()
    observed_tp = combined_tt[:n_observed_tp]  

   
    patch_indices = []     
    st, ed = 0, args.patch_size   
    for i in range(args.npatch):
       
        if i == args.npatch - 1:
            inds = torch.where((observed_tp >= st) & (observed_tp <= ed))[0]
        else:
          
            inds = torch.where((observed_tp >= st) & (observed_tp < ed))[0]
        patch_indices.append(inds)  
        st += args.stride         
        ed += args.stride         


    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
  
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)

    
    predicted_tp = []       
    predicted_data = []    
    predicted_mask = []     

    offset = 0 

    for b, (record_id, tt, vals, mask) in enumerate(batch):

        indices = inverse_indices[offset:offset+len(tt)]
        offset += len(tt)
 
        combined_vals[b, indices] = vals   
        combined_mask[b, indices] = mask


        tmp_n_observed_tp = torch.lt(tt, args.history).sum()
        predicted_tp.append(tt[tmp_n_observed_tp:])
        predicted_data.append(vals[tmp_n_observed_tp:])
        predicted_mask.append(mask[tmp_n_observed_tp:])


    combined_tt = combined_tt[:n_observed_tp]
    combined_vals = combined_vals[:, :n_observed_tp]
    combined_mask = combined_mask[:, :n_observed_tp]

    predicted_tp = pad_sequence(predicted_tp, batch_first=True)
    predicted_data = pad_sequence(predicted_data, batch_first=True)
    predicted_mask = pad_sequence(predicted_mask, batch_first=True)
    print(predicted_tp.shape)  
    print(predicted_data.shape)  
    print(predicted_mask.shape)

    if args.dataset != 'ushcn':
        combined_vals = utils.normalize_masked_data(combined_vals, combined_mask, 
                                                      att_min=data_min, att_max=data_max)
        predicted_data = utils.normalize_masked_data(predicted_data, predicted_mask, 
                                                       att_min=data_min, att_max=data_max)

    combined_tt = utils.normalize_masked_tp(combined_tt, att_min=0, att_max=time_max)
    predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min=0, att_max=time_max)

    # 构造初步的 data_dict 字典，包含归一化后的数据和时间信息
    data_dict = {
        "data": combined_vals,                # 历史观测数据，形状 (B, T_o, D)，其中 T_o = n_observed_tp
        "time_steps": combined_tt,            # 历史时间戳，形状 (T_o, )
        "mask": combined_mask,                # 历史数据的 mask，形状 (B, T_o, D)
        "data_to_predict": predicted_data,    # 预测目标数据，形状 (B, L_out, D)
        "tp_to_predict": predicted_tp,        # 预测目标时间戳，形状 (B, L_out)
        "mask_predicted_data": predicted_mask,# 预测数据的 mask，形状 (B, L_out, D)
    }

    data_dict = utils.split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices)

    return data_dict


def split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices):

	device = get_device(data_dict["data"])

	split_dict = {"tp_to_predict": data_dict["tp_to_predict"].clone(),
			"data_to_predict": data_dict["data_to_predict"].clone(),
			"mask_predicted_data": data_dict["mask_predicted_data"].clone()
			}
	
	observed_tp = data_dict["time_steps"].clone() # (n_observed_tp, ) 
	observed_data = data_dict["data"].clone() # (bs, n_observed_tp, D)
	observed_mask = data_dict["mask"].clone() # (bs, n_observed_tp, D)

	n_batch, n_tp, n_dim = observed_data.shape 
	observed_tp_patches = observed_tp.view(1, 1, -1, 1).repeat(n_batch, args.npatch, 1, n_dim) 
	observed_data_patches = observed_data.view(n_batch, 1, n_tp, n_dim).repeat(1, args.npatch, 1, 1) 
	observed_mask_patches = observed_mask.view(n_batch, 1, n_tp, n_dim).repeat(1, args.npatch, 1, 1) 
	

	max_patch_len = 0
	#对每个 patch，确定在该时间范围内批次中最多有多少个有效的时间点，得到的最大值 max_patch_len 用于将所有 patch 填充成统一的长度
	for i in range(args.npatch):
		indices = patch_indices[i]
		if(len(indices) == 0): continue
		st_ind, ed_ind = indices[0], indices[-1]
		n_data_points = observed_mask[:, st_ind:ed_ind+1].sum(dim=1).max().item()
		max_patch_len = max(max_patch_len, int(n_data_points))

	observed_mask_patches_fill = torch.zeros_like(observed_mask_patches, dtype=observed_mask.dtype) # n_batch, npacth, n_tp, n_dim
	patch_indices_fianl = torch.full((n_batch, args.npatch, max_patch_len, n_dim), n_tp).to(device) # n_batch, npacth, max_patch_len, n_dim
	observed_mask_patches_fill_reindex = torch.zeros_like(patch_indices_fianl, dtype=observed_mask.dtype)
	aux_tensor = torch.arange(max_patch_len).view(1, max_patch_len, 1).repeat(n_batch, 1, n_dim).to(device)
	for i in range(args.npatch):
		indices = patch_indices[i]
		if(len(indices) == 0): continue
		st_ind, ed_ind = indices[0], indices[-1]
		observed_mask_patches_fill[:, i, st_ind:ed_ind+1] = observed_mask[:, st_ind:ed_ind+1, :]
		L = observed_mask[:, st_ind:ed_ind+1, :].sum(dim=1, keepdim=True) # (bs, 1, D)
		observed_mask_patches_fill_reindex[:, i] = (aux_tensor < L)  # let first L[i] to be True
	
	### return a indices tuple like ([...], [...], [...], [...])
	mask_inds = torch.nonzero(observed_mask_patches_fill_reindex.permute(0,1,3,2), as_tuple=True) # reset indices
	ind_values = torch.nonzero(observed_mask_patches_fill.permute(0,1,3,2), as_tuple=True)[-1] # original indices of dimension 2

	### fill n_tp if the number of observed points are less than max_patch_len
	patch_indices_fianl.index_put_((mask_inds[0], mask_inds[1], mask_inds[3], mask_inds[2]), ind_values)

	pad_zeros_data = torch.zeros([n_batch, args.npatch, 1, n_dim]).to(device)
	observed_tp_patches = torch.cat([observed_tp_patches, pad_zeros_data], dim=2).gather(2, patch_indices_fianl) # (n_batch, npatch, max_patch_len, n_dim)
	observed_data_patches = torch.cat([observed_data_patches, pad_zeros_data], dim=2).gather(2, patch_indices_fianl)
	observed_mask_patches = torch.cat([observed_mask_patches, pad_zeros_data], dim=2).gather(2, patch_indices_fianl)
	
	split_dict["observed_tp"] = observed_tp_patches
	split_dict["observed_data"] = observed_data_patches
	split_dict["observed_mask"] = observed_mask_patches 

	return split_dict
"""
TODO list:
1. 数据集仅保留raw_data; merge.csv(主变量与协变量合并的版本，主变量为规则时序， 协变量为非规则时序); 10s_align.csv(对齐后的规则数据集)
2. 写个dataloader，可以从10s_align.csv中读取30s, 60s, 2minus等不同粒度的数据集（用freq='10s' / '30s' / '60s' / '120s'的方式控制）
3. 写完dataloader要在本页中的if __name__ == '__main__'中测试通过（注：必须要有，养成好习惯）
"""
import time
from collections import deque
import os
import sys
# 获取当前文件所在目录的上一级目录，即项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import random
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# print(sys.path)
# print(os.getcwd())
import torch
from datetime import datetime

from utils.timefeatures import time_features


def get_interval(freq):
    
    freq = int(freq.rstrip('s'))
    
    max_interval = 10
    sample_freq = 's'
    
    if 600 < freq < 3600:
        sample_freq = 'min'
        max_interval += freq//60
    elif freq >= 3600:
        sample_freq = 'h'
        max_interval += freq//60
    elif freq >= 86400:
        sample_freq = 'd'
        max_interval += freq//60
    
    return sample_freq, max_interval



class Dataset_ETTh1(Dataset):
    def __init__(self, root_path, data_path, seq_len, pred_len, freq, split_dataset=[0.525, 0.175, 0.3], flag='train', target_num=1):
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = int(seq_len)
        self.label_len = int(seq_len) // 2
        self.pred_len = int(pred_len)
        self.freq = freq
        
        # self.sample_freq = FREQ_MAP[freq]
        self.target_num = target_num
        # self.features = args.msg

        # Initialize the dataset type (train, val, test)
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.split_dataset = split_dataset

        self.sample_freq, self.max_interval = get_interval(freq)


        self.__read_data__()

        self.dim = self.data_x.shape[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        # 选择需要的列
        cols = list(df_raw.columns)
        self.target_name = cols[-self.target_num:]  # 目标变量列

        # 计算数据集划分边界
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 提取数据
        cols_data = df_raw.columns[1:]  # 除时间列外的所有列
        df_data = df_raw[cols_data]

        # 时间特征处理
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.sample_freq)
        data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp

        # 训练集进行归一化
        # train_data = df_data[border1s[0]:border2s[0]]
        # self.scaler.fit(train_data.values)
        
        # 应用归一化
        # data = self.scaler.transform(df_data)
        # 将转换后的 narray 数据转换为 DataFrame
        
        # scaled_df = pd.DataFrame(data, columns=df_data.columns)
        # date_column = df_raw[['date']]
        # scaled_df = pd.concat([date_column, scaled_df], axis=1)
        # # 将 DataFrame 保存为 CSV 文件
        # scaled_df.to_csv('/root/BAOWU_TS/dataset/ETT-small/ETTh1_norm.csv')
        # exit()
        data = df_data.values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]


    def __getitem__(self, index): 
        s_begin = index # continue window 
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # return seq_x[:,:self.dim], seq_y[:,-1], seq_x_mark[:,:self.dim], seq_y_mark[:,-1]
        return seq_x, seq_y, seq_x_mark, seq_y_mark          # x+y 的长度是一个样本

    def __len__(self):
        # return len(self.valid_indices) # skip window 
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_ETTh1_ir(Dataset):
    def __init__(self, root_path, data_path, seq_len, pred_len, freq, split_dataset=[0.525, 0.175, 0.3], flag='train', target_num=1):
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = int(seq_len)
        self.label_len = int(seq_len) // 2
        self.pred_len = int(pred_len)
        self.freq = freq
        
        # self.sample_freq = FREQ_MAP[freq]
        self.target_num = target_num
        # self.features = args.msg

        # Initialize the dataset type (train, val, test)
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.split_dataset = split_dataset

        self.sample_freq, self.max_interval = get_interval(freq)


        self.__read_data__()

        self.dim = self.data_x.shape[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        # 选择需要的列
        cols = list(df_raw.columns)
        self.target_name = cols[-self.target_num:]  # 目标变量列

        # 计算数据集划分边界
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 提取数据
        cols_data = df_raw.columns[1:]  # 除时间列外的所有列
        df_data = df_raw[cols_data]

        # 时间特征处理，将时间戳转换为数值，为了方便后续在单个样本中进行min-max归一化
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        data_stamp = df_stamp['date'].astype(np.int64) // 10**9
        data_stamp = data_stamp.values.reshape(-1, 1)
        self.data_stamp = data_stamp

        # 训练集进行归一化
        # train_data = df_data[border1s[0]:border2s[0]]
        # self.scaler.fit(train_data.values)
        
        # 应用归一化
        data = df_data.values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]


    def __getitem__(self, index): 
        s_begin = index # continue window 
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # 生成 NaN 掩码
        seq_x_mask = ~np.isnan(seq_x)
        seq_y_mask = ~np.isnan(seq_y)

        # 将 NaN 替换为 0
        seq_x = np.nan_to_num(seq_x, nan=0.0)
        seq_y = np.nan_to_num(seq_y, nan=0.0)

        # Apply min-max scaling using the scaler
        seq_x_t = self.data_stamp[s_begin:s_end]
        seq_y_t = self.data_stamp[s_end:r_end]
        min_t = min(seq_x_t.min(), seq_y_t.min())
        max_t = max(seq_x_t.max(), seq_y_t.max())
        seq_x_t = (seq_x_t - min_t) / (max_t - min_t)
        seq_y_t = (seq_y_t - min_t) / (max_t - min_t)

        # return seq_x[:,:self.dim], seq_y[:,-1], seq_x_mark[:,:self.dim], seq_y_mark[:,-1]
        return {
            "x": seq_x,
            "x_mark": seq_x_t,
            "x_mask": seq_x_mask,
            "y": seq_y,
            "y_mark": seq_y_t,
            "y_mask": seq_y_mask,
            "sample_ID": index
        }

    def __len__(self):
        # return len(self.valid_indices) # skip window 
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, seq_len, pred_len, target_num, flag='train', 
                 data_path='ETTh1.csv', split_dataset=[0.7, 0.1, 0.2],
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # self.args = args
        
        self.seq_len = int(seq_len)
        self.label_len = int(seq_len) // 2
        self.pred_len = int(pred_len)
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.split_dataset = split_dataset
        # self.features = features
        # self.target = target
        # self.scale = scale
        # self.timeenc = timeenc
        self.freq = freq
        
        self.sample_freq, self.max_interval = get_interval(freq)

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        # cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]
        num_train = int(len(df_raw) * self.split_dataset[0])
        num_test = int(len(df_raw) * self.split_dataset[-1])
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # if self.features == 'M' or self.features == 'MS':
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        # elif self.features == 'S':
        #     df_data = df_raw[[self.target]]

        # if self.scale:
            
        # else:
        #     data = df_data.values
        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        scaled_df = pd.DataFrame(data, columns=df_data.columns)
        date_column = df_raw[['date']]
        scaled_df = pd.concat([date_column, scaled_df], axis=1)
        # 将 DataFrame 保存为 CSV 文件
        scaled_df.to_csv(f'{self.root_path}/norm.csv', index=False)
        exit()

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # if self.timeenc == 0:
        #     df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        #     df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        #     df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #     df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        #     data_stamp = df_stamp.drop(['date'], 1).values
        # elif self.timeenc == 1:
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.sample_freq)
        data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #     self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom_ir(Dataset):
    def __init__(self, root_path, seq_len, pred_len, target_num, flag='train', 
                 data_path='ETTh1.csv', split_dataset=[0.7, 0.1, 0.2],
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # self.args = args
        
        self.seq_len = int(seq_len)
        self.label_len = int(seq_len) // 2
        self.pred_len = int(pred_len)
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.split_dataset = split_dataset
        # self.features = features
        # self.target = target
        # self.scale = scale
        # self.timeenc = timeenc
        self.freq = freq

        self.sample_freq, self.max_interval = get_interval(freq)

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        # cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]
        num_train = int(len(df_raw) * self.split_dataset[0])
        num_test = int(len(df_raw) * self.split_dataset[-1])
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # if self.features == 'M' or self.features == 'MS':
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        # elif self.features == 'S':
            # df_data = df_raw[[self.target]]

        # # 训练集进行归一化
        # train_data = df_data[border1s[0]:border2s[0]]
        # self.scaler.fit(train_data.values)
        
        # # 应用归一化
        # data = self.scaler.transform(df_data)
        # # 将转换后的 narray 数据转换为 DataFrame
        # scaled_df = pd.DataFrame(data, columns=df_data.columns)
        # date_column = df_raw[['date']]
        # scaled_df = pd.concat([date_column, scaled_df], axis=1)
        # # # 将 DataFrame 保存为 CSV 文件
        # scaled_df.to_csv('/root/BAOWU_TS/dataset/electricity/electricity_norm.csv', index=False)
        # exit()

        data = df_data.values


        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #     self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        # 时间特征处理，将时间戳转换为数值，为了方便后续在单个样本中进行min-max归一化
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        data_stamp = df_stamp['date'].astype(np.int64) // 10**9
        data_stamp = data_stamp.values.reshape(-1, 1)
        self.data_stamp = data_stamp

    def __getitem__(self, index): 
        s_begin = index # continue window 
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # 生成 NaN 掩码
        seq_x_mask = ~np.isnan(seq_x)
        seq_y_mask = ~np.isnan(seq_y)

        # 将 NaN 替换为 0
        seq_x = np.nan_to_num(seq_x, nan=0.0)
        seq_y = np.nan_to_num(seq_y, nan=0.0)

        # Apply min-max scaling using the scaler
        seq_x_t = self.data_stamp[s_begin:s_end]
        seq_y_t = self.data_stamp[s_end:r_end]
        min_t = min(seq_x_t.min(), seq_y_t.min())
        max_t = max(seq_x_t.max(), seq_y_t.max())
        seq_x_t = (seq_x_t - min_t) / (max_t - min_t)
        seq_y_t = (seq_y_t - min_t) / (max_t - min_t)

        # return seq_x[:,:self.dim], seq_y[:,-1], seq_x_mark[:,:self.dim], seq_y_mark[:,-1]
        return {
            "x": seq_x,
            "x_mark": seq_x_t,
            "x_mask": seq_x_mask,
            "y": seq_y,
            "y_mark": seq_y_t,
            "y_mask": seq_y_mask,
            "sample_ID": index
        }

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETTm1_ir(Dataset):
    def __init__(self, root_path, data_path, seq_len, pred_len, freq, split_dataset=[0.525, 0.175, 0.3], flag='train', target_num=1):
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = int(seq_len)
        self.label_len = int(seq_len) // 2
        self.pred_len = int(pred_len)
        self.freq = freq
        
        # self.sample_freq = FREQ_MAP[freq]
        self.target_num = target_num
        # self.features = args.msg

        # Initialize the dataset type (train, val, test)
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.split_dataset = split_dataset

        self.sample_freq, self.max_interval = get_interval(freq)


        self.__read_data__()

        self.dim = self.data_x.shape[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        border1s = [
            0, 
            12 * 30 * 24 * 4 - self.seq_len, 
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
            0
        ]
        border2s = [
            12 * 30 * 24 * 4, 
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        # 时间特征处理，将时间戳转换为数值，为了方便后续在单个样本中进行min-max归一化
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        data_stamp = df_stamp['date'].astype(np.int64) // 10**9
        data_stamp = data_stamp.values.reshape(-1, 1)
        self.data_stamp = data_stamp

        # # 训练集进行归一化
        # train_data = df_data[border1s[0]:border2s[0]]
        # self.scaler.fit(train_data.values)
        
        # # 应用归一化
        # data = self.scaler.transform(df_data)
        # # 将转换后的 narray 数据转换为 DataFrame
        # scaled_df = pd.DataFrame(data, columns=df_data.columns)
        # date_column = df_raw[['date']]
        # scaled_df = pd.concat([date_column, scaled_df], axis=1)
        # # 将 DataFrame 保存为 CSV 文件
        # scaled_df.to_csv('/root/BAOWU_TS/dataset/ETT-small/ETTm1_norm.csv', index=False)
        # exit()
        
        # 应用归一化
        data = df_data.values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        # 生成 NaN 掩码
        seq_x_mask = ~np.isnan(seq_x)
        seq_y_mask = ~np.isnan(seq_y)

        # 将 NaN 替换为 0
        seq_x = np.nan_to_num(seq_x, nan=0.0)
        seq_y = np.nan_to_num(seq_y, nan=0.0)

        # Apply min-max scaling using the scaler
        seq_x_t = self.data_stamp[s_begin:s_end]
        seq_y_t = self.data_stamp[s_end:r_end]
        min_t = min(seq_x_t.min(), seq_y_t.min())
        max_t = max(seq_x_t.max(), seq_y_t.max())
        seq_x_t = (seq_x_t - min_t) / (max_t - min_t)
        seq_y_t = (seq_y_t - min_t) / (max_t - min_t)

        # return seq_x[:,:self.dim], seq_y[:,-1], seq_x_mark[:,:self.dim], seq_y_mark[:,-1]
        return {
            "x": seq_x,
            "x_mark": seq_x_t,
            "x_mask": seq_x_mask,
            "y": seq_y,
            "y_mark": seq_y_t,
            "y_mask": seq_y_mask,
            "sample_ID": index
        }

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

if __name__ == "__main__":
    # 代码中所有组件需要在这里测试
    # dataset在这里测试，dataloader去data_factory中测试
    for freq in ['120s', '60s', '30s']:
        dataset = Dataset_BAOWU_irregular(root_path='dataset',
                                data_path='merged_irregular.csv',
                                seq_len=512,
                                pred_len=512,
                                freq=freq,
                                )
    # 120s 160; 60s 308; 30s 495
    # dataset = Dataset_BAOWU_irregular(root_path='dataset/raw_dataset',
    #                             data_path='merged_tolerance10s.csv',
    #                             seq_len=512,
    #                             pred_len=512,
    #                             freq='3600s',
    #                             )

    x, y, mark_x, mark_y, mark = dataset[0]
    print("Mark:", mark)
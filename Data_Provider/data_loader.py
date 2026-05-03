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

FREQ_MAP = {
            '10s': "s",
            '30s': "s",
            '60s': "min",
            '120s': "min",
        }


def double_exponential_smoothing(data, alpha=0.5, beta=0.5):
    smoothed = np.zeros_like(data)
    trend = np.zeros_like(data)
    
    # 初始化
    smoothed[0] = data[0]
    trend[0] = data[1] - data[0]
    
    # 从第二个数据点开始计算
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * (smoothed[t - 1] + trend[t - 1])
        trend[t] = beta * (smoothed[t] - smoothed[t - 1]) + (1 - beta) * trend[t - 1]
        if np.isnan(smoothed[t]) or np.isnan(trend[t]):
            # print(f"t={t}, data[t]={data[t-1]}, smoothed[t-1]={smoothed[t-1]}, trend[t-1]={trend[t-1]}, smoothed[t]={smoothed[t]}, trend[t]={trend[t]}")
            smoothed[t] = 0
    
    return smoothed

def check_outliers(df: pd.DataFrame):
    df = df.reset_index()
    value_col = df.columns[-1]  # 最后一列为需要检查的数值

    # 计算四分位数和 IQR
    Q1 = df[value_col].quantile(0.25)
    Q3 = df[value_col].quantile(0.75)
    IQR = Q3 - Q1

    # 计算异常值的上下限
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 遍历数据，检查异常值
    for i in range(1, len(df) - 1):  # 忽略第一行和最后一行，因为我们需要前后数据
        value = df.loc[i, value_col]
        
        # 如果值是异常的，替换为前后两个时间戳对应的数值
        if value < lower_bound or value > upper_bound:
            df.loc[i, value_col] = df.loc[i - 1, value_col]

    df.set_index(df.columns[0], inplace=True)
    return df

def find_valid_indices(max_interval, data_x, seq_len, pred_len):
    valid_indices = []
    data = data_x.reset_index()
    timestamps = pd.to_datetime(data["dtime"])
    time_diffs = timestamps.diff().fillna(pd.Timedelta(seconds=0))
    
    # Find all gaps larger than max_interval
    large_gaps = time_diffs > pd.Timedelta(minutes=max_interval)
    gap_indices = large_gaps[large_gaps].index.tolist()
    
    # Add start and end indices to process all segments
    gap_indices = [-1] + gap_indices + [len(data_x)]
    
    # Process each continuous segment between gaps
    for seg_start, seg_end in zip(gap_indices[:-1], gap_indices[1:]):
        start_idx = seg_start + 1
        end_idx = seg_end
        
        # Only process segments that are long enough
        if end_idx - start_idx >= seq_len + pred_len:
            # Add all valid indices in this continuous segment
            valid_indices.extend(range(start_idx, end_idx - seq_len - pred_len + 1))
    return valid_indices

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

class Dataset_BAOWU(Dataset):
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
        df_raw['dtime'] = pd.to_datetime(df_raw['dtime'])

        if self.data_path == "merged.csv":
            # 进行降采样
            base_freq = 10  # 原始数据采样间隔为 10 秒
            downsample_rate = max(1, int(self.freq.rstrip('s')) // base_freq)  # 确保间隔至少为 1
            df_raw = df_raw.iloc[::downsample_rate]  # 根据 freq 进行降采样

            # 重新索引，确保时间连续
            df_raw = df_raw.reset_index(drop=True)

        # 选择需要的列
        cols = list(df_raw.columns)
        self.target_name = cols[-self.target_num:]  # 目标变量列

        # 计算数据集划分边界
        num_train = int(len(df_raw) * self.split_dataset[0])
        num_test = int(len(df_raw) * self.split_dataset[-1])
        num_vali = len(df_raw) - num_train - num_test
        start_border = [0, num_train - self.seq_len, num_train + num_vali - self.seq_len]
        end_border = [num_train, num_train + num_vali, len(df_raw)]
        border1 = start_border[self.set_type]
        border2 = end_border[self.set_type]

        # 提取数据
        cols_data = df_raw.columns[1:]  # 除时间列外的所有列
        df_data = df_raw[cols_data]

        # 时间特征处理
        df_stamp = df_raw[['dtime']][border1:border2]
        df_stamp['dtime'] = pd.to_datetime(df_stamp['dtime'])
        data_stamp = time_features(pd.to_datetime(df_stamp['dtime'].values), freq=self.sample_freq)
        data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp

        # 训练集进行归一化
        train_data = df_data[start_border[0]:end_border[0]]
        self.scaler.fit(train_data.values)

        if self.set_type == 0: # skip window 
            smoothed_data = train_data.copy()
            for column in train_data.columns[1:]:
                smoothed_data[column] = double_exponential_smoothing(train_data[column].values, alpha=0.5, beta=0.5)

            train_data = check_outliers(smoothed_data)[cols_data]
            data = self.scaler.transform(train_data.values)
        else:
            # 应用归一化
            data = self.scaler.transform(df_data.values)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # 计算有效索引
        self.valid_indices = find_valid_indices(self.max_interval, df_raw[border1:border2], self.seq_len, self.pred_len)



    def __getitem__(self, index): 
        
        true_index = self.valid_indices[index] # skip window
        s_begin = true_index
        # s_begin = index # continue window 
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
        return len(self.valid_indices) # skip window 
        # return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_regular(Dataset):
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
        df_stamp = df_raw[['dtime']][border1:border2]
        df_stamp['dtime'] = pd.to_datetime(df_stamp['dtime'])
        data_stamp = time_features(pd.to_datetime(df_stamp['dtime'].values), freq=self.sample_freq)
        data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp

        # 训练集进行归一化
        # train_data = df_data[border1s[0]:border2s[0]]
        # self.scaler.fit(train_data.values)
        
        # 应用归一化
        # data = self.scaler.transform(df_data.values)
        self.data_x = df_data[border1:border2]
        self.data_y = df_data[border1:border2]


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


class Dataset_BAOWU_irregular(Dataset):
    def __init__(self, root_path, data_path, seq_len, pred_len, freq, 
                 split_times={'train': ['2024-01-01 00:00:03', '2024-03-15 08:00:08'], 
                              'val': ['2024-03-15 08:00:08', '2024-04-15 00:00:02'], 
                            #   'test': ['2024-04-15 00:00:02', '2024-05-22 16:24:17']}, 
                            #   'test': ['2024-04-15 00:00:02', '2024-04-15 4:24:17']}, 
                              'test': ['2024-04-15 00:00:02', '2024-05-22 10:56:06']}, 
                 flag='train', target_num=1):   
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = int(seq_len)
        self.label_len = int(seq_len) // 2
        self.pred_len = int(pred_len)
        # self.freq = freq


        self.sample_freq, self.max_interval = get_interval(freq)
        self.downsample_rate = max(1, int(freq.rstrip('s')) // 10)  # 获取降采样率
            
        self.target_num = target_num

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # 计算数据集划分边界
        # 将时间戳字符串转换为datetime对象
        self.split_times = {
            'train': [pd.to_datetime(t) for t in split_times['train']],
            'val': [pd.to_datetime(t) for t in split_times['val']], 
            'test': [pd.to_datetime(t) for t in split_times['test']]
        }

        self.__read_data__()

        self.dim = self.data.shape[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['dtime'] = pd.to_datetime(df_raw['dtime'])

        # 重新索引，确保时间连续
        df_raw = df_raw.reset_index(drop=True)

        # 选择需要的列
        cols = list(df_raw.columns)
        self.target_name = cols[-self.target_num:]  # 目标变量列
        data_cols = cols[1:]

        # 降采样
        s = time.time()
        if self.downsample_rate > 1:
            df_raw = self.fill_downsample_gaps(df_raw, self.downsample_rate, self.downsample_rate*5)
            # print('downsample time:', time.time() - s)
        
        # 按照时间戳切分数据集
        if self.set_type == 0:  # train
            start_time, end_time = self.split_times['train']
            border1 = df_raw[df_raw['dtime'] >= start_time].index[0]
            border2 = df_raw[df_raw['dtime'] < end_time].index[-1]
        elif self.set_type == 1:  # val
            start_time, end_time = self.split_times['val']
            border1 = df_raw[df_raw['dtime'] >= start_time].index[0] - self.seq_len
            border2 = df_raw[df_raw['dtime'] < end_time].index[-1]
        else:  # test
            start_time, end_time = self.split_times['test']
            border1 = df_raw[df_raw['dtime'] >= start_time].index[0] - self.seq_len
            border2 = df_raw[df_raw['dtime'] < end_time].index[-1]
            
        # 训练集进行归一化
        start_border = df_raw[df_raw['dtime'] >= start_time].index[0]
        end_border = df_raw[df_raw['dtime'] < end_time].index[-1]
        train_data = df_raw[start_border:end_border]
        self.scaler.fit(train_data[data_cols].values)

        # 提取数据
        data = df_raw[border1:border2]
        self.valid_indices = find_valid_indices(self.max_interval, data, self.seq_len, self.pred_len)
        
        # 应用归一化
        self.data = self.scaler.transform(data[data_cols].values)
        # self.data = data[data_cols].values

        # 时间特征处理，将时间戳转换为数值，为了方便后续在单个样本中进行min-max归一化
        df_stamp = df_raw[['dtime']][border1:border2]
        df_stamp['dtime'] = pd.to_datetime(df_stamp['dtime'])
        data_stamp = df_stamp['dtime'].astype(np.int64) // 10**9
        data_stamp = data_stamp.values.reshape(-1, 1)
        self.data_stamp = data_stamp


    def fill_downsample_gaps(self, df_raw_orig, downsample_rate, max_interval):
        """
        Args:
            df: 原始DataFrame
            downsample_rate: 降采样率
            max_interval: 最大允许填充时间间隔(秒)
        
        Returns:
            处理后的DataFrame
        """
        # df_raw = df.copy()
        # df_raw_orig = df.copy()
        
        # 计算降采样前后的索引映射
        # orig_indices = df_raw_orig.index.values
        
        df_raw = df_raw_orig.iloc[::downsample_rate]
        downsample_indices = df_raw.index.values
        
        curr_row, fill_data = None, None
        # 对每个被保留的数据点进行处理
        for i in range(len(downsample_indices)-1):
            
            curr_idx = downsample_indices[i]
            curr_row = df_raw.loc[curr_idx]
            
            # 计算与最近保留点的时间差
            curr_time = pd.to_datetime(curr_row['dtime'])
            MAX_INTERVAL = float('inf')  # 将 MAX_INTERVAL 定义为无穷大
            diff_time = MAX_INTERVAL
            # fill_data = None
            
            if pd.isna(curr_row[1:5]).any():
            
                # 获取当前时间点前max_interval秒的数据
                start_idx = max(0, curr_idx-max_interval)
                before_data = df_raw_orig.iloc[start_idx:curr_idx]
                
                # 获取当前时间点后max_interval秒的数据
                end_idx = min(len(df_raw_orig), curr_idx+max_interval+1)
                after_data = df_raw_orig.iloc[curr_idx+1:end_idx]
            
                # 获取before_data中非空值
                if not before_data.empty:
                    valid_vals = before_data.iloc[:, :5].dropna()
                    if not valid_vals.empty:
                        past_valid = valid_vals.loc[valid_vals.axes[0][-1]]
                        past_diff_time = abs((past_valid['dtime'] - curr_time).total_seconds())
                        if past_diff_time < diff_time:
                            diff_time = past_diff_time
                            fill_data = past_valid
                if not after_data.empty:
                    valid_vals = after_data.iloc[:, :5].dropna()
                    if not valid_vals.empty:
                        next_valid = valid_vals.iloc[0]
                        next_diff_time = abs((next_valid['dtime'] - curr_time).total_seconds())
                        if next_diff_time < diff_time:
                            diff_time = next_diff_time
                            fill_data = next_valid
                # 如果时间差在阈值内,填充到最近的保留点
                if diff_time <= max_interval:
                    if fill_data is not None:
                        df_raw.loc[curr_idx, df_raw.columns[1:5]] = fill_data.values[1:]
                        
        return df_raw.reset_index(drop=True)


    def __getitem__(self, index): 
        
        true_index = self.valid_indices[index] # skip window
        
        s_begin = true_index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[s_end:r_end]
        
        # 标记当前样本的协变量是否全部为 NaN，以方便后续对没有协变量数据的处理
        mark = 0 if np.isnan(seq_x[:, :-self.target_num]).all() else 1

        # 生成 NaN 掩码
        seq_x_mask = ~np.isnan(seq_x)
        seq_y_mask = ~np.isnan(seq_y)

        # 将 NaN 替换为 0
        seq_x = np.nan_to_num(seq_x, nan=0.0)
        seq_y = np.nan_to_num(seq_y, nan=0.0)
        
        '''
        时间归一化，按照TimeCHEAT的实现方式，将时间戳归一化到[0,1]之间
        '''
        # Apply min-max scaling using the scaler
        seq_x_t = self.data_stamp[s_begin:s_end]
        seq_y_t = self.data_stamp[s_end:r_end]
        min_t = min(seq_x_t.min(), seq_y_t.min())
        max_t = max(seq_x_t.max(), seq_y_t.max())
        seq_x_t = (seq_x_t - min_t) / (max_t - min_t)
        seq_y_t = (seq_y_t - min_t) / (max_t - min_t)
        
        
        return seq_x, seq_y, seq_x_mask, seq_y_mask, seq_x_t, seq_y_t, mark

    def __len__(self):
        return len(self.valid_indices) # skip window 

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
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class Dataset_Custom(Dataset):
    def __init__(self, root_path, seq_len, pred_len, flag='train', 
                 data_path='exchange.csv', split_dataset=[0.7, 0.1, 0.2]):

        self.seq_len = int(seq_len)
        self.label_len = int(seq_len) // 2
        self.pred_len = int(pred_len)
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.split_dataset = split_dataset

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))


        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]
        num_train = int(len(df_raw) * self.split_dataset[0])
        num_test = int(len(df_raw) * self.split_dataset[-1])
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        data = df_data.values


        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        data_stamp = df_stamp['date'].astype(np.int64) // 10**9
        data_stamp = data_stamp.values.reshape(-1, 1)
        self.data_stamp = data_stamp

    def __getitem__(self, index): 
        s_begin = index 
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mask = ~np.isnan(seq_x)
        seq_y_mask = ~np.isnan(seq_y)

        seq_x = np.nan_to_num(seq_x, nan=0.0)
        seq_y = np.nan_to_num(seq_y, nan=0.0)

        seq_x_t = self.data_stamp[s_begin:s_end]
        seq_y_t = self.data_stamp[s_end:r_end]
        min_t = min(seq_x_t.min(), seq_y_t.min())
        max_t = max(seq_x_t.max(), seq_y_t.max())
        seq_x_t = (seq_x_t - min_t) / (max_t - min_t)
        seq_y_t = (seq_y_t - min_t) / (max_t - min_t)

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

import os
import pandas as pd
import torch


class MIMIC_III(object):

    def __init__(self, root, n_samples = None, device = torch.device("cpu")):

        self.root = root
        self.device = device

        self.process()

        if device == torch.device("cpu"):
            self.data = torch.load(os.path.join(self.processed_folder, 'mimic.pt'), map_location='cpu')
        else:
            self.data = torch.load(os.path.join(self.processed_folder, 'mimic.pt'))

        if n_samples is not None:
            print('Total records:', len(self.data))
            self.data = self.data[:n_samples]

    def process(self):
        if self._check_exists():
            return
        
        filename = os.path.join(self.raw_folder, 'full_dataset.csv')
        
        os.makedirs(self.processed_folder, exist_ok=True)

        print('Processing {}...'.format(filename))

        full_data = pd.read_csv(filename, index_col=0)

        patients = []
        value_cols = [c.startswith('Value') for c in full_data.columns]
        value_cols = list(full_data.columns[value_cols])
        mask_cols = [('Mask' + x[5:]) for x in value_cols]
        # print(value_cols)
        # print(mask_cols)
        data_gp = full_data.groupby(level=0) # group by index
        for record_id, data in data_gp:
            tt = torch.tensor(data['Time'].values).to(self.device).float() / 60.
            vals = torch.tensor(data[value_cols].values).to(self.device).float()
            mask = torch.tensor(data[mask_cols].values).to(self.device).float()
            patients.append((record_id, tt, vals, mask))

        torch.save(
            patients,
            os.path.join(self.processed_folder, 'mimic.pt')
        )

        print('Total records:', len(patients))

        print('Done!')

    def _check_exists(self):

        if not os.path.exists(os.path.join(self.processed_folder, 'mimic.pt')):
            return False
        
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
def Activity_time_chunk(data, configs):

    chunk_data = []
    history = configs.model.seq_len # ms
    pred_window = configs.model.pred_len # ms
    sample_ID = 0
    for record_id, tt, vals, mask in data:
        t_max = int(tt.max())
        for st in range(0, t_max - history, 48):
            et_x = st + history
            et_y = st + history + pred_window
            if(et_x >= t_max):
                idx_x = torch.where((tt >= st) & (tt <= et_x))[0]
            else:
                idx_x = torch.where((tt >= st) & (tt < et_x))[0]
            if(et_y >= t_max):
                idx_y = torch.where((tt >= et_x) & (tt <= et_y))[0]
            else:
                idx_y = torch.where((tt >= et_x) & (tt < et_y))[0]
            new_id = f"{record_id}_{st//pred_window}"
            # chunk_data.append((new_id, tt[idx_x] - st, vals[idx], mask[idx]))
            t_start = tt[idx_x][0]
            t_end = tt[idx_y][-1] + 1 if len(tt[idx_y]) > 0 else tt[idx_x][-1] + 1
            chunk_data.append({
                "sample_ID": sample_ID,
                "x_mark": (tt[idx_x] - t_start) / (t_end - t_start),
                "y_mark": (tt[idx_y] - t_start) / (t_end - t_start),
                "x": vals[idx_x],
                "y": vals[idx_y],
                "x_mask": mask[idx_x],
                "y_mask": mask[idx_y],
            })
            sample_ID += 1

    return chunk_data

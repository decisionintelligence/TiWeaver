import math
import warnings

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, ConcatDataset
from sklearn import model_selection

# from utils.globals import logger
# from utils.ExpConfigs import ExpConfigs
# from utils.configs import configs
from Data_Provider.dependencies.USHCN.USHCN import USHCN, USHCN_time_chunk

warnings.filterwarnings('ignore')   

class Data(Dataset):
    def __init__(
        self, 
        configs,
        flag: str = 'train', 
        **kwargs
    ):
        '''
        warpper for USHCN DeBrouwer2019 dataset implemented in tsdm
        tsdm: https://openreview.net/forum?id=a-bD9-0ycs0

        this version of USHCN does not align the timesteps among samples (but do align within sample), which means:
        - It use custom collate_fn to pad trailing 0s in each batch
        - Tensor length along time dimension is not fixed in different batches, which depends on the max number of timesteps in each batch
        - time steps does not spread evenly, and the start and end time is also not fixed

        - max time length: 200
        - number of variables: 5
        - number of samples: 1114
        - actual time length: 4 year
        '''
        # logger.debug(f"getting {flag} set of USHCN in tsdm format")
        print(f"getting {flag} set of USHCN in tsdm format")
        self.configs = configs
        assert flag in ['train', 'test', 'val', 'test_all']
        self.flag = flag

        self.cache = True

        self.seq_len = configs.model.seq_len
        # self.label_len = configs.model.label_len
        self.pred_len = configs.model.pred_len

        self.dataset_root_path = configs.data.root_path + '/' + configs.data.data_path

        self.preprocess()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def preprocess(self):
        r"""
        preprocess without time alignment
        """

        task = USHCN(
            root=self.dataset_root_path
        )
        
        seen_data, test_data = model_selection.train_test_split(task, train_size= 0.8, random_state = 42, shuffle = False)
        train_data, val_data = model_selection.train_test_split(seen_data, train_size= 0.75, random_state = 42, shuffle = False)
        
        train_data = USHCN_time_chunk(train_data, self.configs.model)
        val_data = USHCN_time_chunk(val_data, self.configs.model)
        test_data = USHCN_time_chunk(test_data, self.configs.model)

        if self.flag != "val":
            # val set will follow the setting of train set
            # determine the max number of observations along time, among all samples
            test_all_data = train_data + val_data + test_data
            self.seq_len_max_irr = 0
            self.pred_len_max_irr = 0
            self.patch_len_max_irr = 0
            seq_residual_len = 0

            SEQ_LEN = self.configs.model.seq_len
            PRED_LEN = self.configs.model.pred_len

            # PATCH_LEN = self.configs.model.patch_len

            for sample in test_all_data:
                if sample["x"].shape[0] > self.seq_len_max_irr:
                    self.seq_len_max_irr = sample["x"].shape[0]
                if sample["y"].shape[0] > self.pred_len_max_irr:
                    self.pred_len_max_irr = sample["y"].shape[0]

            # create a new field in global configs to pass information to models
            self.configs.model.seq_len_max_irr = self.seq_len_max_irr
            self.configs.model.pred_len_max_irr = self.pred_len_max_irr

            print(f"{self.configs.model.seq_len_max_irr=}")
            print(f"{self.configs.model.pred_len_max_irr=}")

        if self.flag == "test_all":
            # merge the 3 datasets
            self.data = train_data + val_data + test_data
        elif self.flag == "train":
            self.data = train_data
        elif self.flag == "val":
            self.data = val_data
        elif self.flag == "test":
            self.data = test_data

def fix_nan_x_mark(x_mark, seq_len):
    L_TOTAL = 200
    # Create a tensor of indices
    BATCH_SIZE, SEQ_LEN_MAX_IRR, _ = x_mark.shape
    indices = torch.linspace(start=seq_len / L_TOTAL - 2 * 0.01, end=seq_len / L_TOTAL - 0.001, steps=SEQ_LEN_MAX_IRR).to(x_mark.device).view(1, -1, 1).repeat(BATCH_SIZE, 1, 1)

    # Create a mask for NaN values
    nan_mask = torch.isnan(x_mark)

    # Fill NaN values using the mask
    x_mark[nan_mask] = indices[nan_mask]

    return x_mark

def fix_nan_y_mark(y_mark):
    # Create a tensor of indices
    BATCH_SIZE, PRED_LEN, _ = y_mark.shape
    indices = torch.linspace(start=1 - 2 * 0.01, end=1 - 0.001, steps=PRED_LEN).to(y_mark.device).view(1, -1, 1).repeat(BATCH_SIZE, 1, 1)

    # Create a mask for NaN values
    nan_mask = torch.isnan(y_mark)

    # Fill NaN values using the mask
    y_mark[nan_mask] = indices[nan_mask]

    return y_mark

def collate_fn(
    batch: list[dict[str,Tensor]],
    configs
) -> dict[Tensor]:
    '''
    rewrite the collate_fn to return dictionary of Tensors, aligning with api
    '''
    # global configs
    seq_len_max_irr: int = configs.model.seq_len_max_irr
    pred_len_max_irr: int = configs.model.pred_len_max_irr

    xs: list[Tensor] = []
    ys: list[Tensor] = []
    x_marks: list[Tensor] = []
    y_marks: list[Tensor] = []
    x_masks: list[Tensor] = []
    y_masks: list[Tensor] = []
    sample_IDs: list[int] = []

    for sample in batch:
        x_mark = sample["x_mark"]
        x = sample["x"]
        y_mark = sample["y_mark"]
        y = sample["y"]

        x_mask = sample["x_mask"]
        y_mask = sample["y_mask"]
        sample_ID = sample["sample_ID"]

        xs.append(x)
        x_marks.append(x_mark)
        x_masks.append(x_mask)

        ys.append(y)
        y_marks.append(y_mark)
        y_masks.append(y_mask)

        sample_IDs.append(sample_ID)

    ENC_IN = xs[0].shape[-1]

    # to ensure padding to n_observations_max, we manually append a sample with desired shape then removed.
    xs.append(torch.zeros(seq_len_max_irr, ENC_IN))
    x_marks.append(torch.zeros(seq_len_max_irr))
    x_masks.append(torch.zeros(seq_len_max_irr, ENC_IN))
    ys.append(torch.zeros(pred_len_max_irr, ENC_IN))
    y_marks.append(torch.zeros(pred_len_max_irr))
    y_masks.append(torch.zeros(pred_len_max_irr, ENC_IN))

    xs=pad_sequence(xs, batch_first=True, padding_value=float("nan"))
    x_marks=pad_sequence(x_marks, batch_first=True, padding_value=float("nan"))
    x_masks=pad_sequence(x_masks, batch_first=True)
    ys=pad_sequence(ys, batch_first=True, padding_value=float("nan"))
    y_marks=pad_sequence(y_marks, batch_first=True, padding_value=float("nan"))
    y_masks=pad_sequence(y_masks, batch_first=True)

    xs = xs[:-1]
    x_marks = x_marks[:-1]
    x_masks = x_masks[:-1]
    ys = ys[:-1]
    y_marks = y_marks[:-1]
    y_masks = y_masks[:-1]

    sample_IDs = torch.tensor(sample_IDs).float()

    if configs.data.missing_rate > 0:
        # manually mask out some observations in input
        # Flatten the mask and data tensor
        flat_mask = x_masks.view(-1)
        flat_x = xs.view(-1)

        # Find indices of available data (where mask is 1)
        available_flat_indices = torch.where(flat_mask == 1)[0]
        num_available = available_flat_indices.size(0)
        num_to_mask = int(configs.data.missing_rate * num_available)

        if num_to_mask > 0:
            # Generate random permutation on the same device
            perm = torch.randperm(num_available, device=available_flat_indices.device)
            selected_flat = available_flat_indices[perm[:num_to_mask]]
            
            # Apply masking to x and x_mask. In-place operation
            flat_x[selected_flat] = torch.nan
            flat_mask[selected_flat] = 0
        else:
            # logger.warning(f"Number of observations {num_available} * missing rate {configs.missing_rate} = {num_to_mask} observations to be masked. Tips: either observations are too sparse, or --missing_rate is too small. Consider increase --missing_rate.")
            print(f"Number of observations {num_available} * missing rate {configs.data.missing_rate} = {num_to_mask} observations to be masked. Tips: either observations are too sparse, or --missing_rate is too small. Consider increase --missing_rate.")


    return {
        "x": torch.nan_to_num(xs),
        "x_mark": fix_nan_x_mark(x_marks.unsqueeze(-1), seq_len=configs.model.seq_len).float(),
        "x_mask": x_masks.float(),
        "y": torch.nan_to_num(ys),
        "y_mark": fix_nan_y_mark(y_marks.unsqueeze(-1)).float(),
        "y_mask": y_masks.float(),
        "sample_ID": sample_IDs
    }

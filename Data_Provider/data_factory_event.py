import importlib
from functools import partial
from torch.utils.data import Dataset, DataLoader

import random
import numpy as np
import torch


def data_provider(configs, flag: str, shuffle_flag: bool = None, drop_last: bool = None) -> tuple[Dataset, DataLoader]:

    dataset_module = importlib.import_module(f"Data_Provider.{configs.data.data_path}")
    
    Data = dataset_module.Data

    try:
        collate_fn = getattr(dataset_module, configs.data.collate_fn)
    except:
        collate_fn = None

    if flag in ["test", "test_all"]:
        shuffle_flag = shuffle_flag or False
        drop_last = drop_last or False
        batch_size = configs.data.batch_size
    else:
        shuffle_flag = shuffle_flag or True
        drop_last = drop_last or True
        batch_size = configs.data.batch_size

    data_set: Dataset = Data(
        configs=configs,
        flag=flag,
    )
    g = torch.Generator()
    g.manual_seed(2024)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=configs.data.num_workers,
        drop_last=drop_last,
        collate_fn=partial(collate_fn, configs=configs),
        generator=g
    ) 

    return data_set, data_loader

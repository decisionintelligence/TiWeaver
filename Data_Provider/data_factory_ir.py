import importlib
from functools import partial
from torch.utils.data import Dataset, DataLoader
# from utils.ExpConfigs import ExpConfigs

import random
import numpy as np
import torch

def check_rng_state(tag):
    print(f"\n--- RNG State @ {tag} ---")

    # Python random state
    print("Python random:", random.getstate()[1][:5])

    # Numpy state
    print("Numpy random:", np.random.get_state()[1][:5])

    # Torch state
    print("Torch seed:", torch.initial_seed())  # global RNG seed
    print("Torch randperm check:", torch.rand(5))  # show sample values (to confirm deterministicness)


def data_provider(configs, flag: str, shuffle_flag: bool = None, drop_last: bool = None) -> tuple[Dataset, DataLoader]:
    '''
    - flag: "train", "val", "test", "test_all"
    - shuffle_flag: In rare cases, it can be manually overwrite.
    - drop_last: In rare cases, it can be manually overwrite.
    '''
    # dynamically import the desired dataset class
    dataset_module = importlib.import_module(f"Data_Provider.{configs.data.data_path}")
    
    Data = dataset_module.Data

    # try to load custom collate_fn for the dataset, if present
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
        # DEBUG: temporal change
        # **configs._asdict()
    )
    g = torch.Generator()
    g.manual_seed(2024)
    # check_rng_state("before DataLoader")
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=configs.data.num_workers,
        drop_last=drop_last,
        collate_fn=partial(collate_fn, configs=configs),
        generator=g
    ) 
    # check_rng_state("after DataLoader")

    return data_set, data_loader

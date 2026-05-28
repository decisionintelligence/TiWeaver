import sys
import yaml
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

gpu_list = "0,1,2,3"  # GPU lst
device_map = {gpu: i for i, gpu in enumerate(gpu_list.split(','))}
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from utils.utils import parsing_syntax, ConfigDict, load_config, update_config, fix_seed
from exp.exp_forecasting import Exp_Forecasting

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dynamical System')

    parser.add_argument('--config_filename', type=str, default='./Model_Config/TiWeawer.yaml', help='Configuration yaml file')
    parser.add_argument('--random_seed', type=int, default=2021, help='Random seed.')

    args, unknown = parser.parse_known_args()
    unknown = parsing_syntax(unknown)

    config = load_config(args.config_filename)
    config = ConfigDict(config)
    config = update_config(config, unknown)
    for attr, value in config.items():
        setattr(args, attr, value)

    # random seed
    fix_seed(args.random_seed)

    args.GPU.use_gpu = True if torch.cuda.is_available() and args.GPU.use_gpu else False

    if args.GPU.use_gpu and not args.GPU.use_multi_gpu:
        try:
            args.GPU.gpu = device_map[str(args.GPU.gpu)]
        except KeyError:
            raise KeyError("This GPU isn't available.")

    if args.GPU.use_gpu and args.GPU.use_multi_gpu:
        args.GPU.devices = args.GPU.devices.replace(' ', '')
        device_ids = args.GPU.devices.split(',')
        args.GPU.device_ids = [int(id_) for id_ in device_ids]
        args.GPU.gpu = args.GPU.device_ids[0]


    exp = Exp_Forecasting(args)
    
    exp.train()
    
    torch.cuda.empty_cache()

    exp.test()


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
import random
from utils.utils import parsing_syntax, ConfigDict, load_config, update_config, fix_seed
from exp.exp_align_forecasting import Exp_Ailgn_Forecasting
from exp.exp_align_rose import Exp_Ailgn_ROSE
from exp.exp_irregular_rose import Exp_Irregular_ROSE
from exp.exp_irregular_timecheat import Exp_Irregular_TimeCHEAT
from exp.exp_irregular_forecasting import Exp_Irregular_Forecasting
from exp.exp_regular_forecasting import Exp_Regular_Forecasting
from exp.exp_align_lightgts import Exp_Ailgn_LightGTS

EXP_MAP = {
    'baowu': Exp_Ailgn_Forecasting,
    'ROSE': Exp_Ailgn_ROSE,
    'irregular_ROSE': Exp_Irregular_ROSE,
    'TimeCHEAT': Exp_Irregular_TimeCHEAT,
    'irregular': Exp_Irregular_Forecasting,
    'regular': Exp_Regular_Forecasting,
    'LightGTS': Exp_Ailgn_LightGTS
}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dynamical System')

    parser.add_argument('--config_filename', type=str, default='./Model_Config/BAOWU/Pathformer.yaml', help='Configuration yaml file')
    parser.add_argument('--itr', type=int, default=1, help='Number of experiments.')
    parser.add_argument('--random_seed', type=int, default=2021, help='Random seed.')
    parser.add_argument('--des', type=str, help="description of experiment.")
    parser.add_argument("--exp_name", type=str, default="baowu", help="exp name")
    parser.add_argument("--time_mark", type=bool, default=False, help="time_mark")
    parser.add_argument("--is_debug", type=bool, default=False, help="debug")
    parser.add_argument("--rolling", type=bool, default=False, help="rolling")

    # parser.add_argument('--loss', type=str, default='huber', help='debug mode.')
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

    rmse_list, mae_list, mape_list = [], [], []
    for exp_idx in range(args.itr):
        args.exp_idx = exp_idx
        if args.to_stdout:
            print('\nNo%d experiment ~~~' % exp_idx)

        exp = EXP_MAP[args.exp_name](args)
        
        exp.train()
        torch.cuda.empty_cache()
        exp.test()


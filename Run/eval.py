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
import numpy as np
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
def get_mean_std(data_list):
    return data_list.mean(), data_list.std()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent AirPhyNet')

    parser.add_argument('--config_filename', type=str, default='Model_Config/BAOWU/ROSE.yaml', help='Configuration yaml file')
    parser.add_argument('--itr', type=int, default=1, help='Number of experiments.')
    parser.add_argument('--random_seed', type=int, default=2024, help='Random seed.')
    parser.add_argument('--des', type=str, help="description of experiment.")
    parser.add_argument("--report_filepath", type=str, default=None, help="evaluation report output")
    parser.add_argument("--save_results", type=bool, default=False, help="whether to save results")
    parser.add_argument("--save_plots", type=bool, default=False, help="whether to save plots")
    parser.add_argument("--exp_name", type=str, default="baowu", help="exp name")
    parser.add_argument("--time_mark", type=bool, default=False, help="time_mark")
    parser.add_argument("--is_debug", type=bool, default=False, help="debug")
    parser.add_argument("--rolling", type=bool, default=False, help="rolling")
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

    mse_list, mae_list, mape_list = [], [], []
    for exp_idx in range(args.itr):
        args.exp_idx = exp_idx
        print('\nNo%d experiment ~~~' % exp_idx)

        exp = EXP_MAP[args.exp_name](args)
        print(exp)
        exp.test()
        # exit(0)
            
    #     mae, mse, mape = exp.test()
    #     if args.report_filepath:
    #         exp.to_report()
    #     mae_list.append(mae)
    #     mse_list.append(mse)
    #     mape_list.append(mape)

    # mae_list = np.array(mae_list)  
    # mape_list = np.array(mape_list)
    # mse_list = np.array(mse_list)

    # exp._logger.info('--------- Final Results ------------')
    # exp._logger.info('MAE | mean: {:.4f}'.format(mae_list.mean(0)))
    # exp._logger.info('MSE | mean: {:.4f}'.format(mse_list.mean(0)))
    # exp._logger.info('MAPE | mean: {:.4f}'.format(mape_list.mean(0)))
    
import os
import torch
import importlib
from torch.utils.tensorboard import SummaryWriter

from models import dynamic_patching_model_GAN
from models import APN_adapter
from models import DLinear_adapter
from models import PatchTST_adapter
from models import TimeMosaic_adapter
from models import PDF, PatchTST
from models import ROSE

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'ROSE': ROSE,
            'APN_adapter': APN_adapter,
            'DLinear_adapter': DLinear_adapter,
            'PatchTST_adapter': PatchTST_adapter,
            'TimeMosaic_adapter': TimeMosaic_adapter,
            'dynamic_patching_model_GAN': dynamic_patching_model_GAN,
            'PDF': PDF,
            'PatchTST': PatchTST
        }
        self.device = self._acquire_device(args.GPU)
        self.model = self._build_model().to(self.device) #self._build_model() 实际调用的是子类 Exp_Ailgn_Forecasting 的 _build_model() 方法

    def _build_model(self):
        raise NotImplementedError

    def _build_TB_logger(self, setting):
        # TB_logger
        log_dir = os.path.join(self.args.TB_dir, setting)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger = SummaryWriter(log_dir)

        return logger

    def _acquire_device(self, args):
        if args.use_gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
            # print('Use GPU: cuda:{}'.format(args.gpu))
        else:
            device = torch.device('cpu')
            # print('Use CPU')
        return device

    def _get_data(self, **kwargs):
        pass

    def vali(self, **kwargs):
        pass

    def train(self, **kwargs):
        pass

    def test(self, **kwargs):
        pass

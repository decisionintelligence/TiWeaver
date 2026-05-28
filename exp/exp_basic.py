import os
import torch
from torch.utils.tensorboard import SummaryWriter

from models import TiWeaver

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TiWeaver': TiWeaver,
        }
        self.device = self._acquire_device(args.GPU)
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _build_TB_logger(self, setting):
        log_dir = os.path.join(self.args.TB_dir, setting)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger = SummaryWriter(log_dir)

        return logger

    def _acquire_device(self, args):
        if args.use_gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device('cpu')
        return device

    def _get_data(self, **kwargs):
        pass

    def vali(self, **kwargs):
        pass

    def train(self, **kwargs):
        pass

    def test(self, **kwargs):
        pass

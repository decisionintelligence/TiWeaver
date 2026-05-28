from exp.exp_basic import Exp_Basic
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm

from utils.utils import get_logger
from utils.metrics import *
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, LRScheduler
from utils.tools import EarlyStopping, count_parameters

warnings.filterwarnings('ignore')

class Exp_Forecasting(Exp_Basic):
    def __init__(self, args):

        if args.to_log_file:
            self._log_dir = self._get_log_dir(args)
        else:
            self._log_dir = None
        self._logger = get_logger(self._log_dir, args.model_name, f'{args.data.data_path.split(".")[0]}_{args.model.seq_len}_{args.model.pred_len}.log',
                                  level=args.log_level, to_stdout=args.to_stdout)
        args.logger = self._logger

        super(Exp_Forecasting, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model_name].Model(self.args).float()
        self._logger.info("Model created")
        self._logger.info(
            "Total trainable parameters {}".format(count_parameters(model))
        )
        self._logger.info("Model setting:\n{}".format(model.setting))
        if self.args.GPU.use_multi_gpu and self.args.GPU.use_gpu:
            print("Using multiple GPUs for training")
            model = nn.DataParallel(model, device_ids=self.args.GPU.device_ids)
        return model

    def _get_data(self, flag):
        if self.args.data.data_name == 'event': # event-driven data
            from Data_Provider.data_factory_event import data_provider
        else: # others
            from Data_Provider.data_factory import data_provider
            
        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.train.lr, eps=1e-8)
        return model_optim

    def _select_criterion(self):
        if self.args.model.loss.criterion == "mse":
            criterion = lambda x, x_hat, mask: torch.pow((x - x_hat) * mask, 2).sum() / (mask.sum())
        elif self.args.model.loss.criterion == "mae":
            criterion = lambda x, x_hat, mask: torch.abs((x - x_hat) * mask).sum() / (mask.sum())
        else:
            criterion = lambda x, x_hat, mask: torch.abs((x - x_hat) * mask).sum() / (mask.sum())
        return criterion

    def _select_lr_scheduler(self, optimizer: optim.Optimizer, train_loader) -> LRScheduler:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** epoch)
        return scheduler

    @staticmethod
    def _get_log_dir(args):
        log_dir = args.train.get('log_dir')
        if log_dir is None:
            run_id = '%s_%s/' % (
                args.model_name, time.strftime('%m-%d-%H-%M-%S'))
            base_dir = args.log_base_dir
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir
        
        
    def vali(self, vali_data, vali_loader):
        with torch.no_grad():
            self.model.eval()

            losses = []
            all_preds, all_trues, all_masks = [], [], []
            for i, batch in tqdm(enumerate(vali_loader), total=len(vali_loader), desc='EVALUATE', leave=False):

                batch_x = batch['x'].float().to(self.device)
                batch_y = batch['y'].float().to(self.device)
                x_mask = batch['x_mask'].float().to(self.device)
                y_mask = batch['y_mask'].float().to(self.device)
                x_mark = batch['x_mark'].float().to(self.device).squeeze(-1)
                y_mark = batch['y_mark'].float().to(self.device).squeeze(-1)
                
                outputs = self.model(batch_x, x_mask=x_mask, x_mark=x_mark, y_mask=y_mask, y_mark=y_mark)

                loss = self.criterion(outputs[:,:, -self.args.model.main_dim:], 
                                      batch_y[:,:, -self.args.model.main_dim:],
                                      y_mask[:,:, -self.args.model.main_dim:])
                losses.append(loss.detach().cpu().numpy()*batch_x.shape[0])


                pred = outputs[:,-self.args.model.pred_len:,:].detach().cpu().numpy()
                true = batch_y[:,-self.args.model.pred_len:,:].detach().cpu().numpy()
                true_mask = y_mask[:,-self.args.model.pred_len:,:].detach().cpu().numpy()

                all_preds.append(pred)
                all_trues.append(true)
                all_masks.append(true_mask)

            all_preds = np.concatenate(all_preds, axis=0)
            all_trues = np.concatenate(all_trues, axis=0)
            all_masks = np.concatenate(all_masks, axis=0)
            maes, mses, mapes = compute_all_metrics_with_mask(all_preds, all_trues, all_masks)
            self._logger.info('Evaluation {:s}: - mae - {:.6f} - mse - {:.6f} - mape - {:.6f}'
                                .format("Vali", maes, mses, mapes))

            return np.sum(losses)/ len(vali_data)

    def train(self):
        self.start = time.time()
        self._logger.info('END TIME: {:.6f}'.format(self.start))
        self._logger.info('Model mode: train')
        train_data, train_loader = self._get_data(flag='train')
        self._logger.info('{}: {}'.format('train_len', len(train_data)))
        vali_data, vali_loader = self._get_data(flag='val')
        self._logger.info('{}: {}'.format('vali_len', len(vali_data)))
        self.criterion = self._select_criterion()
        

        model_save_path = os.path.join(self.args.checkpoints, self.model.setting)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
            
        optimizer = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.train.patience, verbose=True, logger=self._logger)
        lr_scheduler = self._select_lr_scheduler(optimizer, train_loader)

        time_now = time.time()
        train_steps = len(train_loader)

        self._logger.info('Start training ...')
        num_batches = self.args.data.batch_size
        self._logger.info("num_batches: {}".format(num_batches))
        

        for epoch_num in range(1, self.args.train.epochs + 1):

            self.model.train()

            losses = []
            iter_count = 0
            for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch_num}', leave=False):

                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch['x'].float().to(self.device)
                batch_y = batch['y'].float().to(self.device)
                x_mask = batch['x_mask'].float().to(self.device)
                y_mask = batch['y_mask'].float().to(self.device)
                x_mark = batch['x_mark'].float().to(self.device).squeeze(-1)
                y_mark = batch['y_mark'].float().to(self.device).squeeze(-1)

                outputs = self.model(batch_x, x_mask=x_mask, x_mark=x_mark, y_mask=y_mask, y_mark=y_mark)

                loss = self.criterion(outputs[:,:,-self.args.model.main_dim:], 
                                      batch_y[:,:, -self.args.model.main_dim:],
                                      y_mask[:,:,-self.args.model.main_dim:])
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

                del outputs, loss, batch_x, batch_y, x_mark, y_mark
                torch.cuda.empty_cache()

            val_loss = self.vali(vali_data, vali_loader)

            if (epoch_num % self.args.train.log_every) == self.args.train.log_every - 1:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.args.train.epochs - epoch_num) * train_steps - i)
                message = ('Epoch [{}/{}] train_loss: {:.6f}, val_loss: {:.6f}, lr: {:.6f}'
                           .format(epoch_num, self.args.train.epochs,
                                   np.mean(losses), val_loss, optimizer.param_groups[0]['lr']))
                self._logger.info(message)
                self._logger.info('speed: {:.6f}s/iter; left time: {:.6f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            lr_scheduler.step(val_loss)

            early_stopping(val_loss, self.model, model_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            self._logger.info("---" * 30)
            
    def test(self):
        self.infer_start = time.time()
        test_data, test_loader = self._get_data(flag='test')
        self._logger.info('{}: {}'.format('test_len', len(test_data)))

        self._logger.info('Loading model: test')
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, self.model.setting, 'checkpoint.pth')))

        
        with torch.no_grad():
            self.model.eval()
            
            all_preds, all_trues, all_masks = [], [], []
            for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test', leave=False):

                batch_x = batch['x'].float().to(self.device)
                batch_y = batch['y'].float().to(self.device)
                x_mask = batch['x_mask'].float().to(self.device)
                y_mask = batch['y_mask'].float().to(self.device)
                x_mark = batch['x_mark'].float().to(self.device).squeeze(-1)
                y_mark = batch['y_mark'].float().to(self.device).squeeze(-1)

                outputs = self.model(batch_x, x_mask=x_mask, x_mark=x_mark, y_mask=y_mask, y_mark=y_mark)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                true_mask = y_mask.detach().cpu().numpy()

                all_preds.append(pred)
                all_trues.append(true)
                all_masks.append(true_mask)

        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        maes, mses, mapes = compute_all_metrics_with_mask(all_preds, all_trues, all_masks)
        self._logger.info('Evaluation {:s}: - mae - {:.6f} - mse - {:.6f} - mape - {:.6f}'
                            .format("Test", maes, mses, mapes))
        end = time.time()
        self._logger.info('END TIME: {:.6f}'.format(end))
        self._logger.info('INFER TIME: {:.6f}'.format(end-self.infer_start))
        self._logger.info('TOTAL TIME: {:.6f}'.format(end-self.start))


    def _batch_compute_loss(self, y_true, y_predicted, y_mask):

        maes, mses, mapes = [], [], []
        for i in range(self.args.model.main_dim):
            
            # inverse transform
            batch_true = self.inverse_transform(y_true.reshape((-1, y_true.shape[-1]))).reshape((-1, self.args.model.pred_len, y_true.shape[-1])) 
            batch_pred = self.inverse_transform(y_predicted.reshape((-1, y_predicted.shape[-1]))).reshape((-1, self.args.model.pred_len, y_true.shape[-1])) 

            mae, mse, mape = compute_all_metrics_with_mask(batch_pred[:, :, -(i+1)], batch_true[:, :, -(i+1)], y_mask[:, :, -(i+1)])
            maes.append(mae * y_true.shape[0])
            mses.append(mse * y_true.shape[0])
            mapes.append(mape * y_true.shape[0])

            del batch_true, batch_pred
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        return maes, mses, mapes
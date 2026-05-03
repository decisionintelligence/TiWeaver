from exp.exp_basic import Exp_Basic
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.utils import get_logger
from Data_Provider.data_factory import data_provider
from utils.metrics import *
from utils.tools import EarlyStopping, count_parameters
from tqdm import tqdm
warnings.filterwarnings('ignore')

class Exp_Regular_Forecasting(Exp_Basic):
    def __init__(self, args):

        if args.to_log_file:
            self._log_dir = self._get_log_dir(args)
        else:
            self._log_dir = None
        self._logger = get_logger(self._log_dir, 
                                  args.model_name, 
                                  f'{args.data.data_path.split(".")[0]}_{args.data.freq}_{args.model.seq_len}_{args.model.pred_len}.log',
                                  level=args.log_level, to_stdout=args.to_stdout)
        args.logger = self._logger

        super(Exp_Regular_Forecasting, self).__init__(args)


    def _build_model(self):
        model = self.model_dict[self.args.model_name].Model(self.args).float()
        self._logger.info("Model created")
        self._logger.info(
            "Total trainable parameters {}".format(count_parameters(model))
        )
        self._logger.info("Model setting:\n{}".format(model.setting))
        if self.args.GPU.use_multi_gpu and self.args.GPU.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.GPU.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.train.lr, eps=1e-8)
        return model_optim

    def _select_criterion(self):
        if self.args.model.loss.criterion == "mse":
            criterion = nn.MSELoss()
        elif self.args.model.loss.criterion == "mae":
            criterion = nn.L1Loss()
        elif self.args.model.loss.criterion == "huber":
            criterion = nn.HuberLoss(delta=0.5)
        else:
            criterion = nn.L1Loss()
        return criterion

    def _select_lr_scheduler(self, optimizer, train_loader):
        if self.args.train.lradj == 'MultiStep':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.train.steps,
                                                                gamma=self.args.train.lr_decay_ratio)
        elif self.args.train.lradj == 'TST':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                               steps_per_epoch=len(train_loader),
                                                               pct_start=self.args.train.pct_start,
                                                               epochs=self.args.train.epochs,
                                                               max_lr=self.args.train.lr)
        else:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        return lr_scheduler

    def vali(self, vali_data, vali_loader):
        with torch.no_grad():
            self.model.eval()

            losses = []
            all_preds, all_trues = [], []
            for i, (batch_x, batch_y, x_mark, y_mark) in tqdm(enumerate(vali_loader), total=len(vali_loader), desc='EVALUATE', leave=False):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                x_mark = x_mark.float().to(self.device)
                y_mark = y_mark.float().to(self.device)

                if self.args.time_mark:
                    outputs = self.model(batch_x, x_mark)
                elif self.args.model.loss.recon_loss:
                    outputs, recon = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                # 计算每个batch的loss并保存
                loss = self.criterion(outputs[:,-self.args.model.pred_len:, -self.args.model.main_dim:], 
                                      batch_y[:,-self.args.model.pred_len:, -self.args.model.main_dim:])
                if self.args.model.loss.recon_loss:
                    loss += self.args.model.loss.recon_loss_weight * self.criterion(recon, batch_x)
                losses.append(loss.detach().cpu().numpy()*batch_x.shape[0])


                pred = outputs[:,-self.args.model.pred_len:,:].detach().cpu().numpy()
                true = batch_y[:,-self.args.model.pred_len:,:].detach().cpu().numpy()

                # maes, mses, mapes = compute_all_metrics(pred, true)
                all_preds.append(pred)
                all_trues.append(true)
                
            del outputs, loss, batch_x, batch_y, x_mark, y_mark
            torch.cuda.empty_cache()

            # maes = np.sum(np.array(all_maes), axis=0) / len(vali_data)
            # mses = np.sum(np.array(all_mses), axis=0)/ len(vali_data)
            # mapes = np.sum(np.array(all_mapes), axis=0) / len(vali_data)
            all_preds = np.concatenate(all_preds, axis=0)
            all_trues = np.concatenate(all_trues, axis=0)
            maes, mses, mapes = compute_all_metrics(all_preds, all_trues)
            self._logger.info('Evaluation {:s}: - mae - {:.4f} - mse - {:.4f} - mape - {:.4f}'
                                .format("Vali", maes, mses, mapes))

            return np.sum(losses)/ len(vali_data)

    def train(self):
        if self.args.TB_dir:
            self.TB_logger = self._build_TB_logger(self.model.setting)
        else:
            self.TB_logger = None
        self._logger.info('Model mode: train')
        train_data, train_loader = self._get_data(flag='train')
        self._logger.info('{}: {}'.format('train_len', len(train_data)))
        self.target_name = train_data.target_name
        vali_data, vali_loader = self._get_data(flag='val')
        self._logger.info('{}: {}'.format('vali_len', len(vali_data)))
        self.inverse_transform = train_data.inverse_transform
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

            # print('\nTrain epoch %s:' % (epoch_num))
            self.model.train()
            losses = []
            iter_count = 0
            for i, (batch_x, batch_y, x_mark, y_mark) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch_num}', leave=False):

                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                x_mark = x_mark.float().to(self.device)
                y_mark = y_mark.float().to(self.device) 
                
                if self.args.time_mark:
                    outputs = self.model(batch_x, x_mark)
                elif self.args.model.loss.recon_loss:
                    outputs, recon = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                
                loss = self.criterion(outputs[:,-self.args.model.pred_len:,-self.args.model.main_dim:], batch_y[:,-self.args.model.pred_len:, -self.args.model.main_dim:])
                if self.args.model.loss.recon_loss:
                    loss += self.args.model.loss.recon_loss_weight * self.criterion(recon, batch_x)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

                if self.args.train.lradj == 'TST':
                    lr_adjust = {epoch_num: lr_scheduler.get_last_lr()[0]}
                    if epoch_num in lr_adjust.keys():
                        lr = lr_adjust[epoch_num]
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr
                    lr_scheduler.step()

                del outputs, loss, batch_x, batch_y, x_mark, y_mark
                torch.cuda.empty_cache()

            val_loss = self.vali(vali_data, vali_loader)

            if (epoch_num % self.args.train.log_every) == self.args.train.log_every - 1:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.args.train.epochs - epoch_num) * train_steps - i)
                message = ('Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}'
                           .format(epoch_num, self.args.train.epochs,
                                   np.mean(losses), val_loss, optimizer.param_groups[0]['lr']))
                self._logger.info(message)
                self._logger.info('speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            # 学习率动态调整
            if self.args.train.lradj == 'MultiStep':
                lr_scheduler.step()
            elif self.args.train.lradj == 'TST':
                pass
            else:
                lr_scheduler.step(val_loss)

            early_stopping(val_loss, self.model, model_save_path)
            if early_stopping.early_stop:
                break

            self._logger.info("---" * 30)
            
    def test(self):

        test_data, test_loader = self._get_data(flag='test')
        self._logger.info('{}: {}'.format('test_len', len(test_data)))
        self.target_name = test_data.target_name
        self.inverse_transform = test_data.inverse_transform
        
        for i, (batch_x, batch_y, x_mark, y_mark) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test', leave=False):
            pass

        self._logger.info('Loading model: test')
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, self.model.setting, 'checkpoint.pth')))
        
        with torch.no_grad():
            self.model.eval()
            
            all_maes, all_mses, all_mapes = [], [], []
            all_preds, all_trues = [], []
            for i, (batch_x, batch_y, x_mark, y_mark) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test', leave=False):
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                x_mark = x_mark.float().to(self.device) 
                y_mark = y_mark.float().to(self.device)


                if self.args.time_mark:
                    outputs = self.model(batch_x, x_mark)
                elif self.args.model.loss.recon_loss:
                    outputs, recon = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)


                pred = outputs[:,-self.args.model.pred_len:,:].detach().cpu().numpy()
                true = batch_y[:,-self.args.model.pred_len:,:].detach().cpu().numpy()

                all_preds.append(pred)
                all_trues.append(true)
                
        del outputs, batch_x, batch_y, x_mark, y_mark
        torch.cuda.empty_cache()
        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        maes, mses, mapes = compute_all_metrics(all_preds, all_trues)
        self._logger.info('Evaluation {:s}: - mae - {:.4f} - mse - {:.4f} - mape - {:.4f}'
                                .format("Vali", maes, mses, mapes))
        


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

        
    def _batch_compute_loss(self, y_true, y_predicted):

        # inverse transform
        # batch_true = self.inverse_transform(y_true.reshape((-1, y_true.shape[-1]))).reshape((-1, self.args.model.pred_len, y_true.shape[-1]))  # 保持原始形状
        # batch_pred = self.inverse_transform(y_predicted.reshape((-1, y_predicted.shape[-1]))).reshape((-1, self.args.model.pred_len, y_true.shape[-1]))  # 保持原始形状
            
        mae, mse, mape = compute_all_metrics(y_predicted, y_true)
        # 清理内存
        del batch_true, batch_pred
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        return mae, mse, mape
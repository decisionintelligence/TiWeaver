from exp.exp_basic import Exp_Basic
import torch.nn as nn
from torch import optim
import os
import time
from tqdm import tqdm

import warnings
import numpy as np
from utils.utils import get_logger
from Data_Provider.data_factory import data_provider
from utils.metrics import *
from utils.tools import EarlyStopping, count_parameters
from Data_analysis.visual import visual

warnings.filterwarnings('ignore')

class Exp_Irregular_TimeCHEAT(Exp_Basic):
    def __init__(self, args):

        if args.to_log_file:
            self._log_dir = self._get_log_dir(args)
        else:
            self._log_dir = None
        self._logger = get_logger(self._log_dir, args.model_name, f'{args.data.data_path.split(".")[0]}_{args.data.freq}_{args.model.seq_len}_{args.model.pred_len}.log',
                                  level=args.log_level, to_stdout=args.to_stdout)
        args.logger = self._logger

        super(Exp_Irregular_TimeCHEAT, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model_name].Model(self.args).float()
        self._logger.info("Model created")
        self._logger.info(
            "Total trainable parameters {}".format(count_parameters(model))
        )
        if self.args.GPU.use_multi_gpu and self.args.GPU.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.GPU.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

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
        
    def mse_with_mask_torch(self, x, x_hat, mask):
        return torch.pow((x - x_hat) * mask, 2).sum() / (mask.sum())

    def mae_with_mask_torch(self, x, x_hat, mask):
        return torch.abs((x - x_hat) * mask).sum() / (mask.sum())

    def vali(self, vali_data, vali_loader):
        with torch.no_grad():
            self.model.eval()

            # preds = []
            # truths = []
            losses = []
            all_maes, all_mses, all_mapes = [], [], []
            for i, (batch_x, batch_y, x_mark, y_mark, x_t, y_t, mask) in tqdm(enumerate(vali_loader), desc='Vali', total=len(vali_loader), leave=False):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                x_mark = x_mark.float().to(self.device)
                y_mark = y_mark.float().to(self.device)
                x_t = x_t.float().to(self.device).squeeze(-1)
                y_t = y_t.float().to(self.device).squeeze(-1)
                mask = mask.float().to(self.device)

                outputs = self.model(batch_x, x_mark, x_t)
                

                # 计算每个batch的loss并保存
                loss = self.criterion(outputs[:,-self.args.model.pred_len:, -self.args.model.main_dim:], 
                                      batch_y[:,-self.args.model.pred_len:, -self.args.model.main_dim:],
                                      y_mark[:,-self.args.model.pred_len:, -self.args.model.main_dim:])
                losses.append(loss.detach().cpu().numpy()*batch_x.shape[0])


                if self.args.model_name == 'pathCHEAT':
                    outputs = torch.cat([batch_y[:,:, :-3], outputs], dim=-1)
                pred = outputs[:,-self.args.model.pred_len:,:].detach().cpu().numpy()
                true = batch_y[:,-self.args.model.pred_len:,:].detach().cpu().numpy()
                true_mask = y_mark[:,-self.args.model.pred_len:,:].detach().cpu().numpy()

                maes, mses, mapes = self._batch_compute_loss(true, pred, true_mask)
                all_maes.append(maes)
                all_mses.append(mses)
                all_mapes.append(mapes)
                
            del outputs, loss, batch_x, batch_y, x_mark, y_mark, x_t, y_t, mask
            torch.cuda.empty_cache()

            maes = np.sum(np.array(all_maes), axis=0) / len(vali_data)
            mses = np.sum(np.array(all_mses), axis=0)/ len(vali_data)
            mapes = np.sum(np.array(all_mapes), axis=0) / len(vali_data)
            for i in range(self.args.model.main_dim):

                self._logger.info('Evaluation {:s} {:s}: - mae - {:.4f} - mse - {:.4f} - mape - {:.4f}'
                                .format("Vali", self.target_name[self.args.model.main_dim-i-1], maes[i], mses[i], mapes[i]))

            return np.sum(losses)/ len(vali_data)

    def train(self):

        self._logger.info('Model mode: train')
        train_data, train_loader = self._get_data(flag='train')
        self._logger.info('{}: {}'.format('train_len', len(train_data)))
        self.target_name = train_data.target_name
        vali_data, vali_loader = self._get_data(flag='val')
        self._logger.info('{}: {}'.format('vali_len', len(vali_data)))
        self.inverse_transform = train_data.inverse_transform
        # self.criterion = self.mse_with_mask_torch
        self.criterion = self.mae_with_mask_torch

        model_save_path = os.path.join(self.args.checkpoints, self.model.setting)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        early_stopping = EarlyStopping(patience=self.args.train.patience, verbose=True, logger=self._logger)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.train.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, min_lr=1e-5, verbose=True)

        time_now = time.time()
        train_steps = len(train_loader)

        self._logger.info('Start training ...')
        num_batches = self.args.data.batch_size
        self._logger.info("num_batches: {}".format(num_batches))

        for epoch_num in range(1, self.args.train.epochs + 1):

            print('\nTrain epoch %s:' % (epoch_num))
            self.model.train()

            losses, sch_loss = [], torch.tensor(0., device=self.device)
            iter_count = 0
            for i, (batch_x, batch_y, x_mark, y_mark, x_t, y_t, mask) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch_num}', leave=False):

                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                x_mark = x_mark.float().to(self.device)
                y_mark = y_mark.float().to(self.device)
                x_t = x_t.float().to(self.device).squeeze(-1)
                y_t = y_t.float().to(self.device).squeeze(-1)
                mask = mask.float().to(self.device)
                
                outputs = self.model(batch_x, x_mark, x_t)

                loss = self.criterion(outputs[:,-self.args.model.pred_len:,-self.args.model.main_dim:], 
                                      batch_y[:,-self.args.model.pred_len:, -self.args.model.main_dim:],
                                      y_mark[:,-self.args.model.pred_len:,-self.args.model.main_dim:])
                # loss += recon_loss
                sch_loss += loss
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

                del outputs, loss, batch_x, batch_y, x_mark, y_mark, x_t, y_t, mask
                torch.cuda.empty_cache()

            lr_scheduler.step(sch_loss / len(train_loader))
            
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
            # lr_scheduler.step(val_loss)

            early_stopping(val_loss, self.model, model_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            self._logger.info("---" * 30)

        # torch.save(self.model.state_dict(), model_save_path + '/' + 'checkpoint.pth')
            
    def test(self):

        test_data, test_loader = self._get_data(flag='test')
        self._logger.info('{}: {}'.format('test_len', len(test_data)))
        self.target_name = test_data.target_name
        self.inverse_transform = test_data.inverse_transform

        self._logger.info('Loading model: test')
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, self.model.setting, 'checkpoint.pth')))
        
        with torch.no_grad():
            self.model.eval()
            
            all_maes, all_mses, all_mapes = [], [], []
            for i, (batch_x, batch_y, x_mark, y_mark, x_t, y_t, mask) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test', leave=False):
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                x_mark = x_mark.float().to(self.device) 
                y_mark = y_mark.float().to(self.device)
                x_t = x_t.float().to(self.device).squeeze(-1)
                y_t = y_t.float().to(self.device).squeeze(-1)
                mask = mask.float().to(self.device)

                outputs = self.model(batch_x, x_mark, x_t)
                
                if self.args.model_name == 'pathCHEAT':
                    outputs = torch.cat([batch_y[:,:, :-3], outputs], dim=-1)
                    
                # def modify_values_tensor(data1, data2):
                #     mask = abs(data1) < 0.1
                #     # data1[mask] = 0
                #     data2[mask] = 0
                #     return data2

                # outputs = modify_values_tensor(batch_y, outputs)

                pred = outputs[:,-self.args.model.pred_len:,:].detach().cpu().numpy()
                true = batch_y[:,-self.args.model.pred_len:,:].detach().cpu().numpy()
                input = batch_x[:,-self.args.model.seq_len:,:].detach().cpu().numpy()
                true_mask = y_mark[:,-self.args.model.pred_len:,:].detach().cpu().numpy()

                # for j in range(len(pred)):
                #     visual(input[j], pred[j],true[j],str(i)+'_'+str(j))

                '''
                # 初始化长序列（用第一个样本的输入作为起始）
                long_input = input[0]             # (seq_len, var)
                long_pred = [] 
                long_true = []

                # 拼接预测和真实值
                for j in range(len(pred)):
                    long_pred.append(pred[j][-1])   # 每个窗口只取最后一个预测点
                    long_true.append(true[j][-1])   # 同理真实值

                long_pred = np.stack(long_pred, axis=0)   # (batch_size, var)
                long_true = np.stack(long_true, axis=0)   # (batch_size, var)

                # 最终拼接：输入 + 预测
                # long_series_pred = np.concatenate([long_input, pred[0][:-1], long_pred], axis=0)
                # long_series_true = np.concatenate([long_input, true[0][:-1], long_true], axis=0)
                long_series_pred = np.concatenate([pred[0][:-1], long_pred], axis=0)
                long_series_true = np.concatenate([true[0][:-1], long_true], axis=0)

                # 绘图
                visual(long_input, long_series_pred, long_series_true, str(i))
                '''

                maes, mses, mapes = self._batch_compute_loss(true, pred, true_mask)
                all_maes.append(maes)
                all_mses.append(mses)
                all_mapes.append(mapes)
                # print(maes, mses, mapes)
        

        maes = np.sum(np.array(all_maes), axis=0) / len(test_loader)
        mses = np.sum(np.array(all_mses), axis=0) / len(test_loader)
        mapes = np.sum(np.array(all_mapes), axis=0) / len(test_loader)
        for i in range(self.args.model.main_dim):

            self._logger.info('Evaluation {:s} {:s}: - mae - {:.4f} - mse - {:.4f} - mape - {:.4f}'
                              .format("Test", self.target_name[self.args.model.main_dim-i-1], maes[i], mses[i], mapes[i]))
            
        # visual(batch_x[0], pred[0], true[0], "pathCHEAT", 0, "test", "BAOWU")


    def _batch_compute_loss(self, y_true, y_predicted, y_mask):

        maes, mses, mapes = [], [], []
        for i in range(self.args.model.main_dim):
            
            # inverse transform
            batch_true = self.inverse_transform(y_true.reshape((-1, y_true.shape[-1]))).reshape((-1, self.args.model.pred_len, y_true.shape[-1]))  # 保持原始形状
            batch_pred = self.inverse_transform(y_predicted.reshape((-1, y_predicted.shape[-1]))).reshape((-1, self.args.model.pred_len, y_true.shape[-1]))  # 保持原始形状

            mask = batch_true <= 0.1
            y_mask = y_mask * ~mask

            mae, mse, mape = compute_all_metrics_with_mask(batch_pred[:, :, -(i+1)], batch_true[:, :, -(i+1)], y_mask[:, :, -(i+1)])
            print(mae, mse, mape)
            # maes.append(mae * y_true.shape[0])
            # mses.append(mse * y_true.shape[0])
            # mapes.append(mape * y_true.shape[0])

            maes.append(mae)
            mses.append(mse)
            mapes.append(mape)

            # 清理内存
            del batch_true, batch_pred
            # if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        return maes, mses, mapes
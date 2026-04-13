import os
import torch
import numpy as np
import datetime
import time
from exp.exp_basic import Exp_basic
from data.data_loader import Dataset
from exp.losses import Losses, TensorboardObserver


from torch import optim, nn
from torch.utils.data import DataLoader
from data.data_provider import data_provider
from matplotlib import pyplot as plt
from data.data_loader import *
from utils.tools import *
from utils.metrics import metric
from sklearn.metrics import r2_score
from itertools import chain


class Exp_Short_Term_Forecasting(Exp_basic):
    
    def __init__(self, args):
        super(Exp_Short_Term_Forecasting, self).__init__(args)

        
        self.loss = Losses(args)
    
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_cuda:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
            
        return model.to(self.device)
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim
    
    def _select_gt(self, x_enc,  x_dec, x_mark_enc, x_mark_dec, batch_y, flag ='train',c_enc=None):

        """
        x_enc: [B, T, D] input sequence
        x_dec: [B, L + H, D] decoder input, sequence label sequence (L) + zero_padding sequence (H)
        batch_y: [B, L + H, D]  label sequence (L) + target sequence (H)
        """
        batch_x = x_enc # [B, T, D]
        batch_c = c_enc
        if self.args.model in ['VRNN']: # VRNN
            
            return {
                'y_true': batch_y[:,-self.args.pred_len:,-self.args.C_out:],
                'x_true': batch_x  
            }
        elif self.args.model in ['DMVAER']: # DMVAER 
            if flag == 'train':
                return {
                    'y_true': batch_y[:,-self.args.pred_len:],
                    'x_true': batch_x,
                    'c_true': batch_c
            }
            else:
                return batch_y[:,-self.args.pred_len:]

        elif self.args.model in ['CVAESMC']:
            if flag == 'train':
                return batch_y[:, -self.args.pred_len:, -self.args.C_out: ] # 通过x(1:t)和y(1:t-1)预测y(t)
            else:
                return batch_y[:, -self.args.pred_len:, :]
          
           
        else: # 其他模型
            if flag == 'train':
                return batch_y[:,-self.args.pred_len:,-self.args.C_out:]
            else:
                return batch_y[:,-self.args.pred_len:, -self.args.C_out:] # for inverse, not select target dimension
           
        
        
    def _select_pred(self, outputs,flag ='train'):
        if self.args.model in ['VRNN','DMVAER']:
            if flag == 'train':  
                return outputs
            else:
                return outputs['y_pred']
            
        elif self.args.model in ['TCVAE']:
            if flag == 'train':
                return outputs
            else:
                outputs = outputs['dec_out_infer']
                return outputs[:, -self.args.pred_len:, :]
        elif self.args.model in ['CVAESMC']:
            if flag == 'train':
                return outputs
            else:
                return outputs
        else:
            if flag == 'train':
                return outputs[:, -self.args.pred_len:, -self.args.C_out:]
            else:
                return outputs[:, -self.args.pred_len:, -self.args.C_out:]
    

    def train(self, logger):
        
            
        # 检查是否启用tensorboard，优先使用args中的设置，否则默认为False
        use_tensorboard = getattr(self.args, 'use_tensorboard', False)
        if use_tensorboard:
            self.writer = TensorboardObserver(self.args.save_dir)
            
            
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='valid')

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()

        for epoch in range(self.args.epoch):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                loss_list = []
                iter_count += 1
                model_optim.zero_grad()

                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                
                outputs = self.model(**batch)
                
                outputs = self._select_pred(outputs)
                trues = self._select_gt(**batch)
                
                total_loss = self.loss.calculate_loss(outputs, trues, flag = 'train')
                total_loss.backward()
                loss_list.append(total_loss.item())
                
                model_optim.step()
               
            vali_loss = self.vali(val_data, val_loader)
              
            logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            # 记录验证损失和训练损失到TensorBoard
            if hasattr(self, 'writer'):
                self.writer.add_scalar('Loss/val', vali_loss, epoch)
                # 使用当前epoch的平均训练损失而不是访问内部列表
                train_loss = np.mean(loss_list) if loss_list else 0
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                
            
            self.loss.print_loss(epoch, train_steps, vali_loss, logger)            
            

            early_stopping(vali_loss, self.model, self.args.save_dir)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            
            
        best_model_path = self.args.save_dir + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
    
    
    def vali(self, val_data, val_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                
                outputs = self._select_pred(outputs)
                trues = self._select_gt(**batch)
                
                
                loss = self.loss.calculate_loss(outputs, trues, flag = 'valid')
                total_loss.append(loss.item())
                
            total_loss = np.average(total_loss)
            
        self.model.train()
        return total_loss
    
    
    def test(self, logger):
        test_data, test_loader = self._get_data(flag='test')
        
        
        preds = []
        trues = []
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
             
                outputs = self.model(**batch, flag='test')

                gt = self._select_gt(**batch, flag='test')
                outputs = self._select_pred(outputs, flag='test')
                
                gt = gt.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                
                if self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], shape[2])).reshape(shape)
                    gt = test_data.inverse_transform(gt.reshape(shape[0] * shape[1], shape[2])).reshape(shape)

                outputs = outputs[:,:,-self.args.C_out:]
                gt = gt[:,:,-self.args.C_out:]
                
                preds.append(outputs)
                trues.append(gt)
      
      
      
        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)
      
        print('test shape:', preds.shape, 'trues shape:', trues.shape)
        
        mae,mse,rmse,mape,mspe = metric(preds, trues)
        logger.info(f"mae:{mae:.4f}, mse:{mse:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}")
        
        np.save(self.args.save_dir + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(self.args.save_dir + 'pred.npy', preds)
        np.save(self.args.save_dir + 'true.npy', trues)



       
       
        # 只有在启用了TensorBoard的情况下才记录超参数和关闭writer
        if hasattr(self, 'writer'):
            hparams = select_tensorboard_hparams(self.args)
            hparams['mae/hparam'] = float(mae)
            hparams['mse/hparam'] = float(mse)
            hparams['rmse/hparam'] = float(rmse)
            metrics = {}
            self.writer.add_hparams(hparams, metrics)
            self.writer.flush()
            self.writer.close()
            
        
        plt.figure()
        plt.plot(trues[:, 0, 0], label='GT')  # Assuming trues is a 3D array, adjust indices as needed
        plt.plot(preds[:, 0, 0], label='Predicted')  # Assuming preds is a 3D array, adjust indices as needed
        plt.legend()
        plt.savefig(self.args.save_dir + 'test.png')  # Save the plot as a PDF file
        logger.info(f"Saved plot as 'test.png' in {self.args.save_dir}")

        plt.show()
        return
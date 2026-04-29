import os
import torch
import numpy as np
import copy
import time
from exp import Exp_basic, Losses, TensorboardObserver
from utils import metric, adjust_learning_rate, EarlyStopping


from torch import optim, nn
from data import data_provider
from matplotlib import pyplot as plt


import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('TkAgg')

class Exp_Soft_Sensor(Exp_basic):
    
    def __init__(self, args):
        super(Exp_Soft_Sensor, self).__init__(args)
        
        self.loss = Losses(args)


    
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_cuda:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
        print(model)
        return model.to(self.device)    
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim
    
    def _select_gt(self, x_enc, x_dec, x_mark_enc, x_mark_dec, batch_y, c_enc=None, flag = 'Train'):
        batch_x = x_enc
        if self.args.model in ['VRNN']:
            if flag == 'Train':
                return {
                    'y_true': batch_y,
                    'x_true': batch_x  
            }
            else:
                return batch_y
        elif self.args.model in ['DMVAER']:
            if flag == 'Train':
                return {
                    'y_true': batch_y,
                    'x_true': batch_x,  
                    'c_true': c_enc
                }
            else:
                return batch_y.squeeze(1)
        else:
            return batch_y.squeeze(1)
            
    def _select_pred(self, outputs):
        if self.args.model in ['VRNN']:
            return outputs['y_pred']
        elif self.args.model in ['DMVAER']:
            return outputs['y_pred']
        else:
            return outputs
    

    def train(self, logger):
        
        folder_path = self.args.save_dir
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        use_tensorboard = getattr(self.args, 'use_tensorboard', False)
        if use_tensorboard:
            self.writer = TensorboardObserver(self.args.save_dir)
        
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='valid')

        train_time = 0.0
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        
        for epoch in range(self.args.epoch):
            iter_count = 0
            
            self.model.train()
            
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):

                batch = {k: v.to(self.device) for k, v in batch.items()}
                model_optim.zero_grad()  ##梯度清零

                outputs = self.model(**batch)

                
                trues = self._select_gt(**batch)

                loss = self.loss.calculate_loss(outputs, trues, flag = 'train')
                
                
                loss.backward()
                model_optim.step()
                
            vali_loss = self.vali(val_data, val_loader)
            epoch_time = time.time() - epoch_time
            train_time += epoch_time  # Add epoch time to train_time
            logger.info("Epoch: {} cost time: {}".format(epoch + 1, epoch_time))
            
            # 记录验证损失和训练损失到TensorBoard
            if hasattr(self, 'writer'):
                self.writer.add_scalar('Loss/val', vali_loss, epoch)
              
            
            self.loss.print_loss(epoch, train_steps, vali_loss, logger)
            
            early_stopping(vali_loss, self.model, self.args.save_dir)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        logger.info("Training time: {:.4f}".format(train_time))
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
            
                trues = self._select_gt(**batch)
                
                loss = self.loss.calculate_loss(outputs, trues, flag = 'valid')

                total_loss.append(loss.item())
                
            total_loss = np.average(total_loss)
            
        self.model.train()
        return total_loss
    
    
    def test(self,logger):
        test_data, test_loader = self._get_data(flag='test')
 
        
        preds = []
        trues = []
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
          
                
                
                outputs = self.model(**batch)
                outputs = self._select_pred(outputs)
                gt = self._select_gt(**batch, flag='test')

                gt = gt.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()

                if self.args.inverse:
                    outputs = test_data.inverse_transform(outputs)
                    gt = test_data.inverse_transform(gt)
                
                
                preds.append(outputs)
                trues.append(gt)
      
        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)
      
        print('test shape:', preds.shape, 'trues shape:', trues.shape)
        
        mae,mse,rmse,mape,mspe,r2,corr = metric(preds, trues, self.args.task)
        logger.info(f"mae:{mae:.4f}, mse:{mse:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}, r2:{r2:.4f}, corr:{corr:.4f}")
        np.save(self.args.save_dir + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2, corr]))
        np.save(self.args.save_dir + 'pred.npy', preds)
        np.save(self.args.save_dir + 'true.npy', trues)
        
        # 只有在启用了TensorBoard的情况下才记录超参数
        if hasattr(self, 'writer'):
            hparams = {
                'lr': self.args.learning_rate,
                'batch_size': self.args.batch_size,
                'model': self.args.model,
                'epochs': self.args.epoch,
            }
            metrics = {
                'mae/test': 6.5,
                'mse/test': mse,
                'rmse/test': rmse,
            }
            self.writer.add_hparams(hparams, metrics)
        
        plt.figure()
        plt.plot(trues, label='GT')  # Assuming trues is a 3D array, adjust indices as needed
        plt.plot(preds, label='Predicted')  # Assuming preds is a 3D array, adjust indices as needed
        plt.legend()
        plt.savefig(self.args.save_dir + 'test.png')  # 
        logger.info(f"Saved plot as 'test.png' in {self.args.save_dir}")

        plt.show()
        return
import torch.nn.functional as F  
import torch
import numpy as np
from torch import nn
import datetime
import os
from torch.utils.tensorboard import SummaryWriter


class TensorboardObserver():
    """
    Create a TensorBoard observer for logging metrics and hparams.
    """
    
    def __init__(self, folder_path) -> None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(folder_path, f"exp_{timestamp}")
        print(log_dir)
        self._writer = SummaryWriter(log_dir)
    
    @property
    def observer(self):
        return self._writer
    
    def add_scalar(self, tag, scalar_value, global_step=None):
        self._writer.add_scalar(tag, scalar_value, global_step)

    def add_hparams(self, hparams, metrics, global_step=None):
        self._writer.add_hparams(hparams, metrics)
    
    def close(self):
        self._writer.close()
    def flush(self):
        self._writer.flush()

class BaseLoss(nn.Module):
    """损失函数基类，提供通用的损失列表管理和清空功能"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._loss_lists = {}  # 存储所有损失列表
    
    def _register_loss_list(self, name, initial_list=None):
        """注册一个新的损失列表"""
        if initial_list is None:
            initial_list = []
        self._loss_lists[name] = initial_list
        # setattr(self, f'{name}_list', initial_list)
    
    def _append_loss(self, list_name, value):
        """向指定的损失列表添加值"""
        self._loss_lists[list_name].append(value)
    
    def _clear_loss_lists(self):
        """清空所有损失列表"""
        for name, loss_list in self._loss_lists.items():
            loss_list.clear()
    
    @property
    def mean_total_loss(self):
        """计算总损失的平均值"""
        # if hasattr(self, 'loss_list') and self.loss_list:
        return np.mean(self._loss_lists['loss'])
    
    def print_loss_details(self):
        """打印特定模型的损失详情，子类可重写"""
        pass

class Losses:  
    def __init__(self, args, folder_path = None):
        self.calculate_loss = losses_dict[args.model](args)
        self.args = args
    
        
    def __call__(self, *args, **kwargs):
        return self.calculate_loss(*args, **kwargs)

    def print_loss(self, epoch, train_steps, valid_loss, logger):
        # 打印通用损失信息
        message = f'Epoch [{epoch + 1}/{self.args.epoch}], Steps: {train_steps} Train Loss: {self.calculate_loss.mean_total_loss:.4f}, Valid Loss: {valid_loss:.4f}'
        print(message)
        
        logger.info(message)
        

            
        # 打印特定模型的详细损失信息
        self.calculate_loss.print_loss_details()
        
        # 清空所有损失列表
        self.calculate_loss._clear_loss_lists()
    


class MSE_Loss(BaseLoss):
    def __init__(self, args):
        super().__init__(args)
        self.MSEloss = nn.MSELoss(reduction='mean')
        self._register_loss_list('loss')
    
    def forward(self, y_pred, y_true, flag='train'):
        loss = self.MSEloss(y_pred, y_true)     
        
        if flag == 'train':
            self._append_loss('loss', loss.item())
        
        return loss

class CVAESMC_Loss(BaseLoss):
    def __init__(self, args):
        super().__init__(args)
        self.MSEloss = nn.MSELoss(reduction='mean')
        self._register_loss_list('loss')
        self._register_loss_list('kl_loss')
        self._register_loss_list('recon_loss')

    
    @property
    def mean_kl_loss(self):
        """计算KL损失的平均值"""
        return np.mean(self._loss_lists['kl_loss'])
  
    @property
    def mean_recon_loss(self):
        """计算重建损失的平均值"""
        return np.mean(self._loss_lists['recon_loss'])

    def forward(self, preds,true, flag='train'):

        z_mu_p = preds['mu_posterior']
        z_logvar_p = preds['logvar_posterior']
        y_mu = preds['mean_dec'] # [num_samples,B  C_y]
        y_logvar = preds['logvar_dec'] # [ num_samples, B, C_y]
        y_true = true

        y_true = y_true.squeeze(1)

        y_true = y_true.unsqueeze(0).expand(y_mu.shape[0], -1, -1)


        sigma_w = torch.exp(y_logvar)
        mse = torch.sum((y_mu - y_true)**2, dim=-1)
        mse = mse.mean(dim=(0,1))
        recon_loss = 0.5 *(mse/(2 * sigma_w) + 0.5 *torch.log(2*torch.pi * sigma_w))
        
        kl = -0.5 * (
            1 + z_logvar_p - z_mu_p.pow(2) - z_logvar_p.exp()
        )
        kl_loss = kl.sum()
        kl_loss = kl_loss.mean()

        loss = recon_loss +  kl_loss
        
        if flag == 'train':
            self._append_loss('loss', loss.item())
            self._append_loss('kl_loss', kl_loss.item())
            self._append_loss('recon_loss', recon_loss.item())
        return loss
    def print_loss_details(self):
        """打印SMCVAE模型的详细损失信息"""
        print(f'KL Loss: {self.mean_kl_loss:.4f}, Recon Loss: {self.mean_recon_loss:.4f}')


class DMVAER_Loss(BaseLoss):
    def __init__(self, args):
        super().__init__(args)
        self.MSEloss = nn.MSELoss(reduction='mean')
        self._register_loss_list('loss')
        self._register_loss_list('zs_kl_loss')
        self._register_loss_list('zt_kl_loss')
        self._register_loss_list('c_kl_loss')
        self._register_loss_list('x_recon_loss')
        self._register_loss_list('y_recon_loss')
    
    @property
    def mean_zs_kl_loss(self):
        """计算z_s KL损失的平均值"""
        return np.mean(self._loss_lists['zs_kl_loss'])
  
    
    @property
    def mean_zt_kl_loss(self):
        """计算z_t KL损失的平均值"""
        return np.mean(self._loss_lists['zt_kl_loss'])
    
    @property
    def mean_c_kl_loss(self):
        """计算c KL损失的平均值"""
        return np.mean(self._loss_lists['c_kl_loss'])
    
    @property
    def mean_x_recon_loss(self):
        """计算x重建损失的平均值"""
        return np.mean(self._loss_lists['x_recon_loss'])
      
    
    @property
    def mean_y_recon_loss(self):
        """计算y重建损失的平均值"""
        return np.mean(self._loss_lists['y_recon_loss'])
    
    def print_loss_details(self):
        """打印DMVAER模型的详细损失信息"""
        print(f'z_s KL Loss: {self.mean_zs_kl_loss:.4f}, z_t KL Loss: {self.mean_zt_kl_loss:.4f}, c KL Loss: {self.mean_c_kl_loss:.4f}, x Recon Loss: {self.mean_x_recon_loss:.4f}, y Recon Loss: {self.mean_y_recon_loss:.4f}')
    
    def forward(self, preds, true, flag = 'train'):
        x_true = true['x_true']
        y_true = true['y_true']
        c_true = true['c_true']
        
        x_pred = preds['x_pred']
        y_pred = preds['y_pred']
        c_pred = preds['c_pred']
        mu_zt = preds['mu_zt']
        logvar_zt = preds['logvar_zt']
        zs_mu_p = preds['mu_zs']
        zs_logvar_p = preds['logvar_zs']
        
        recon_x_loss = self.MSEloss(x_pred, x_true)
        recon_y_loss = self.MSEloss(y_pred, y_true)
        
        # KL Divergence for z_t
        kl_zt = 0.0
        for mu,logvar in zip(mu_zt, logvar_zt):
            kl_zt += torch.sum(-0.5 * (1 + logvar - mu ** 2 - logvar.exp()), dim=1).mean()
        
        # KL Divergence for z_s
        kl_zs = 0.0
        for k in range(self.args.n_components):
            mu_q = zs_mu_p[k]
            logvar_q = zs_logvar_p[k]
            
            mu_p, logvar_p = torch.zeros_like(mu_q), torch.zeros_like(logvar_q)

            
            kl = 0.5 * torch.sum( c_pred[:, :,k] * (
                    logvar_p - logvar_q +
                    (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp() - 1),
                    dim=-1
                )
            kl_zs += kl.mean()

   
        # KL divergence c label
        log_qc = torch.log(c_pred + 1e-12)
        log_pc = torch.log(c_true + 1e-12)
        kl_c = torch.sum(c_pred * (log_qc - log_pc), dim=-1).mean()
        balance = [ 0.1, 1, 1, 1, 0.01]
        loss = recon_x_loss + recon_y_loss + kl_zt + kl_zs + kl_c
        loss = balance[0] * recon_x_loss + balance[1] * recon_y_loss + balance[2] * kl_zt + balance[3] * kl_zs + balance[4] * kl_c
        
        if flag == 'train':
            self._append_loss('loss', loss.item())
            self._append_loss('zs_kl_loss', kl_zs.item())
            self._append_loss('zt_kl_loss', kl_zt.item())
            self._append_loss('c_kl_loss', kl_c.item())
            self._append_loss('x_recon_loss', recon_x_loss.item())
            self._append_loss('y_recon_loss', recon_y_loss.item())
        return loss


class TCVAE_loss(BaseLoss):
    def __init__(self, args):
        super().__init__(args)
        self.MSEloss = nn.MSELoss(reduction='mean')
        self._register_loss_list('loss')
        self._register_loss_list('pred_loss')
        self._register_loss_list('KL_loss')
        
    @property
    def mean_pred_loss(self):
        """计算预测损失的平均值"""
        return np.mean(self._loss_lists['pred_loss'])
    
    @property
    def mean_KL_loss(self):
        """计算KL散度损失的平均值"""
        
        return np.mean(self._loss_lists['KL_loss'])
    
    def print_loss_details(self):
        """打印TCVAE模型的详细损失信息"""
        print(f'Pred Loss: {self.mean_pred_loss:.4f}, KL Loss: {self.mean_KL_loss:.4f}')
        
    
    def forward(self, preds, true, flag = 'train'):
        
        pred_y = preds['dec_out_train'][:, -self.args.pred_len:, :]
        pred_loss = self.MSEloss(pred_y, true)
        
        p_mean = preds['p_mean']
        p_logvar = preds['p_logvar']
        q_mean = preds['q_mean']
        q_logvar = preds['q_logvar']
        KL_loss = 0.5 * (p_logvar - q_logvar + (torch.exp(q_logvar) + 
                                                          (q_mean - p_mean)**2) / (torch.exp(p_logvar) + 1e-6) - 1)
        KL_loss = KL_loss.mean()
        loss = pred_loss + KL_loss
        if flag == 'train':
            self._append_loss('loss', loss.item())
            self._append_loss('pred_loss', pred_loss.item())
            self._append_loss('KL_loss', KL_loss.item())
        return loss
    
class VRNN_loss(BaseLoss):
    
    def __init__(self, args):
        super().__init__(args)
        self.MSEloss = nn.MSELoss(reduction='mean')
        self._register_loss_list('loss')
        self._register_loss_list('recon_loss_x')
        self._register_loss_list('recon_loss_y')
        self._register_loss_list('KL_loss')
    
    @property
    def mean_recon_x_loss(self):
        """计算x重建损失的平均值"""
        return np.mean(self._loss_lists['recon_loss_x'])

    
    @property
    def mean_recon_y_loss(self):
        """计算y重建损失的平均值"""

        return np.mean(self._loss_lists['recon_loss_y'])

    
    @property
    def mean_KL_loss(self):
        """计算KL散度损失的平均值"""
        return np.mean(self._loss_lists['KL_loss'])

    # def mean_total_loss(self):
    #     return super().mean_total_loss
    
    def print_loss_details(self):
        """打印VRNN模型的详细损失信息"""
        print(f'Recon Loss X: {self.mean_recon_x_loss:.4f}, Recon Loss Y: {self.mean_recon_y_loss:.4f}, KL Loss: {self.mean_KL_loss:.4f}')
    
    def forward(self, preds, trues, trues_rec=None, flag = 'train'):
        y_pred = preds['y_pred']
        x_pred = preds['x_pred']
        x_true = trues['x_true']
        y_true = trues['y_true']
        
        recon_x_loss = self.MSEloss(x_pred, x_true)
        
        recon_y_loss = self.MSEloss(y_pred, y_true)
        
        KL_loss =  0.5 * (preds['logvar_prior'] - preds['logvar_posterior'] + (torch.exp(preds['logvar_posterior']) + 
                                                    (preds['mean_posterior'] - preds['mean_prior'])**2) / (torch.exp(preds['logvar_prior']) + 1e-6) - 1)
        KL_loss = KL_loss.mean()
        
        loss = recon_x_loss + recon_y_loss + KL_loss
        
        if flag == 'train':
            self._append_loss('loss', loss.item())
            self._append_loss('recon_loss_x', recon_x_loss.item())
            self._append_loss('recon_loss_y', recon_y_loss.item())
            self._append_loss('KL_loss', KL_loss.item())
        return loss
    


class HuberLoss(BaseLoss):
    def __init__(self, args):
        super().__init__(args)
        self.huber_loss = nn.HuberLoss(reduction='mean',delta=0.8)
        self._register_loss_list('loss')
    def forward(self, preds, true, flag = 'train'):
        loss = self.huber_loss(preds, true)
        if flag == 'train':
            self._append_loss('loss', loss.item())
        return loss
    
losses_dict = {
    'Nystroformer': MSE_Loss,
    'DAGRU': MSE_Loss,
    'DMVAER': DMVAER_Loss,
    'VRNN': VRNN_loss,
    'TCVAE': TCVAE_loss,
    'iTransformer': MSE_Loss,
    'Transformer': MSE_Loss,
    'EnvFormer': MSE_Loss,
    'Fredformer': MSE_Loss,
    'HSAM_dGRUs': HuberLoss,
    'PatchTST': MSE_Loss,
    'Autoformer': MSE_Loss,
    'DLinear': MSE_Loss,
    'ARDNN': MSE_Loss,
    'MSACNN': MSE_Loss,
    'CVAESMC': CVAESMC_Loss,
    'LDCNN': MSE_Loss,
    'Nonstationary_Transformer': MSE_Loss,
    'DMRIFormer': MSE_Loss,
}









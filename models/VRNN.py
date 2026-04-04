import torch
import numpy as np
from torch import nn



# VRNN A Recurrent Latent Variable Model for Sequential Data

def reparameterize(mu, logvar):
    std = torch.exp(logvar * 0.5)
    eps = torch.randn_like(std)
    return mu + eps * std


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        """
        x_dim: int, 输入维度
        d_model: int, 隐状态维度
        z_dim: int, 隐变量维度
       
        z_embed_dim: int, 隐变量嵌入维度
        x_embed_dim: int, 输入嵌入维度
        """

        try:
            self.activation = getattr(nn, configs.activation)()
        except:
            raise NameError(f'Invalid activation name \'{configs.activation} \'. Please Checkout activation name')
        self.configs = configs
        
        self.GRUcell_Decoder = nn.GRUCell(configs.z_embed_dim+configs.x_embed_dim, configs.d_model)
    
        self.prior_dense = nn.Sequential(
            nn.Linear(configs.d_model, configs.z_dim*2),
            self.activation
        )
        self.posterior_dense = nn.Sequential(
            nn.Linear(configs.d_model+configs.x_embed_dim, configs.z_dim*2),
            self.activation
        )

        self.dense_x = nn.Sequential(
            nn.Linear(configs.C_in, configs.x_embed_dim),
            self.activation
        )
        self.dense_zt = nn.Sequential(
            nn.Linear(configs.z_dim, configs.z_embed_dim),
            self.activation
        )

        self.dense_z_x = nn.Sequential(
            nn.Linear(configs.z_embed_dim+configs.d_model, configs.C_in),
            self.activation
        )


        # self.GRU_y = nn.GRU(self.configs.d_model, self.configs.d_model2, num_layers=1, batch_first=True)

        self.dense_y = nn.Sequential(
            Permute(0,2,1),
            nn.Linear(1, self.configs.pred_len),
            Permute(0,2,1),
        )
        self.dense_output_y = nn.Linear(configs.d_model * self.configs.seq_len, configs.C_out * configs.pred_len)
        
        self.LayerNorm1 = nn.LayerNorm(self.configs.z_embed_dim)
        self.LayerNorm2 = nn.LayerNorm(self.configs.d_model)
        
        if self.configs.task == 'soft_sensor':
            self.projection = nn.Linear(self.configs.d_model, self.configs.C_out)
        
        elif self.configs.task == 'short_term_forecasting':
            self.projection = nn.Sequential(
                Permute(0,2,1),
                nn.Linear(self.configs.seq_len, self.configs.pred_len),
                Permute(0,2,1),
                nn.Linear(self.configs.d_model, self.configs.C_out))
        
    def encoder(self, x_enc, x_mark_enc,x_dec, x_mark_dec):
        """
        Encoder for the DMVAER model
        Model Input:
        For soft sensor task: x_enc: [B, T, C_in]
        For short-term forecasting task: x_enc: [B, T + pred_len, C_in]
        Model Output:
        For soft sensor task: x_dec_out: [B, T, C_in], y_dec_out: [B, 1, C_out]
        For short-term forecasting task: x_dec_out: [B, T + pred_len, C_in], y_dec_out: [B, pred_len, C_out]
        """
        x_enc = x_enc[:,:,:self.configs.C_in] 
        x_mark_enc = x_mark_enc[:,:,:self.configs.C_in] 
        x_dec = x_dec[:,:,:self.configs.C_in] 
        x_mark_dec = x_mark_dec[:,:,:self.configs.C_in] 
        
        
        B, T, D = x_enc.shape
        h = torch.randn(B, self.configs.d_model,dtype=torch.float32,requires_grad=False).to(self.configs.device)
        mean_posterior_list = []
        logvar_posterior_list = []
        h_list = []

        mean_prior_list = []
        logvar_prior_list = []
        x_pred_list = []
        z_posterior_list = []

        # Inference and Generation
        for t in range(T):

            x_t = x_enc[:, t, :] # [batch_size, C_in]
            x_f = self.dense_x(x_t) # [batch_size, x_embed_dim]

            # Inference
            # 求近似后验
            posterior_mu_logvar = self.posterior_dense(torch.cat([x_f, h], dim=-1)) # [batch_size, z_dim*2]
            mu_posterior, logvar_posterior = torch.chunk(posterior_mu_logvar, 2, dim=-1) # [batch_size, z_dim]
            z_posterior = reparameterize(mu_posterior, logvar_posterior) # [batch_size, z_dim]

            # Generation
            # 求先验
            prior_mu_logvar = self.prior_dense(h) # [batch_size, z_dim*2]
            mu_prior, logvar_prior = torch.chunk(prior_mu_logvar, 2, dim=-1) # [batch_size, z_dim]
            z_prior = reparameterize(mu_prior, logvar_prior) # [batch_size, z_dim]

            z_prior_f = self.dense_zt(z_prior) # [batch_size, z_embed_dim]
            z_prior_f = self.LayerNorm1(z_prior_f)
            x_pred = self.dense_z_x(torch.cat([z_prior_f, h], dim=-1))

            # 更新h
            h = self.GRUcell_Decoder(torch.cat([z_prior_f, x_f], dim=-1), h) # [batch_size, d_model]

            mean_posterior_list.append(mu_posterior)
            logvar_posterior_list.append(logvar_posterior)
            mean_prior_list.append(mu_prior)
            logvar_prior_list.append(logvar_prior)
            x_pred_list.append(x_pred)
            h_list.append(h)
            z_posterior_list.append(z_posterior)

        
        hd = torch.stack(h_list, dim=1) # [batch_size, seq_len, d_model]
        hd = self.LayerNorm2(hd)
        
        mean_posterior = torch.stack(mean_posterior_list, dim=1) # [batch_size, seq_len, z_dim]
        logvar_posterior = torch.stack(logvar_posterior_list, dim=1) # [batch_size, seq_len, z_dim]
        mean_prior = torch.stack(mean_prior_list, dim=1) # [batch_size, seq_len, z_dim]
        logvar_prior = torch.stack(logvar_prior_list, dim=1) # [batch_size, seq_len, z_dim]
        z_posterior = torch.stack(z_posterior_list, dim=1) # [batch_size, seq_len, z_dim]
        x_pred = torch.stack(x_pred_list, dim=1) # [batch_size, seq_len, C_in]
        
        outputs = {
            'hd': hd, # hidden variable
            'mean_posterior': mean_posterior,
            'logvar_posterior': logvar_posterior,
            'mean_prior': mean_prior,
            'logvar_prior': logvar_prior,
            'z_posterior': z_posterior,
            'x_pred': x_pred
        } 

        
        return outputs
        
        
    def soft_sensor(self, x_enc,x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.encoder(x_enc,x_mark_enc, x_dec, x_mark_dec)
        hd = dec_out['hd'][:, -1, :] # [batch_size, d_model]
        
        y_pred = self.projection(hd)
        dec_out['y_pred'] = y_pred
        return dec_out
    
    def short_term_forecasting(self, x_enc,x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.encoder(x_enc,x_mark_enc, x_dec, x_mark_dec)
        hd = dec_out['hd']# [batch_size, pred_len, d_model]
        
        y_pred = self.projection(hd)
        dec_out['y_pred'] = y_pred
        return dec_out
        
    
    def forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec, batch_y, flag='train'):
        
        if self.configs.task == 'soft_sensor':
            dec_out = self.soft_sensor(x_enc,x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        elif self.configs.task == 'short_term_forecasting':
            dec_out = self.short_term_forecasting(x_enc,x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        else:
            raise ValueError('Invalid task name')
        
        


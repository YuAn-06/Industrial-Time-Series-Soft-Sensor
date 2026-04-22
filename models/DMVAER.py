# Copyright (C) 2021 #
# @Time    : 2025/1/25 15:58
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : DGMVAE.py
# @Software: PyCharm

import torch
from torch import nn
from layers.Embedding import *
from collections import namedtuple
from layers.Output_Layer import *


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.n_components = configs.n_components
        self.z_global_dim = configs.z_global_dim
        self.z_local_dim = configs.z_local_dim
        self.enc_in = configs.enc_in
        self.C_out = configs.C_out
        self.d_model = configs.d_model
        self.configs = configs

        num_modes = configs.n_components
        # q(c | x_1:T)
        self.gru_c = nn.GRU(self.enc_in, self.d_model, batch_first=True)
        self.q_c = nn.Linear(self.d_model, num_modes)  # logits

        # q(zs | x, c)
        self.q_zs = nn.ModuleList([
            nn.Linear(self.d_model, 2 * self.z_global_dim) for _ in range(num_modes)
        ])

        # q(zt | z<t, zs, x)
        self.q_zt_gru = nn.GRU(self.enc_in + self.z_global_dim + self.z_local_dim, self.d_model, batch_first=True)
        self.q_zt_mu = nn.Linear(self.d_model, self.z_local_dim)
        self.q_zt_logvar = nn.Linear(self.d_model, self.z_local_dim)

        # p(zt | z<t, zs)
        self.p_zt_gru = nn.GRU(self.z_global_dim + self.z_local_dim, self.d_model, batch_first=True)
        self.p_zt_mu = nn.Linear(self.d_model, self.z_local_dim)
        self.p_zt_logvar = nn.Linear(self.d_model, self.z_local_dim)

        # 基于 GRU 的解码器和回归器
        self.decoder_gru_x = nn.GRU(self.z_local_dim, self.d_model, batch_first=True)
        self.decoder_output_x = nn.Linear(self.d_model, self.enc_in)
        self.decoder_gru_y = nn.GRU(self.z_local_dim, self.d_model, batch_first=True)
        self.decoder_output_y = nn.Linear(self.d_model, self.C_out)
        # self.dense_output_t = nn.Sequential(
        #     nn.Linear(configs.seq_len, configs.d_model),  
            
        #     nn.Linear(configs.d_model, configs.pred_len)
        # )

        if configs.task == 'soft_sensor':
            self.projection = nn.Linear(configs.d_model, configs.C_out)
            self.discriminator = self.discriminator_SS
        elif configs.task == 'short_term_forecasting':
            # self.projection = nn.Sequential(
            #     Permute(0,2,1),
            #     nn.Linear(self.configs.seq_len, self.configs.pred_len),
            #     Permute(0,2,1),
            #     nn.Linear(self.configs.d_model, self.configs.C_out))
            self.projection = nn.Linear(configs.d_model, configs.C_out)
            self.discriminator = self.discriminator_LSF
            
        # self.dense_output_y = nn.Linear(configs.d_model * configs.seq_len, configs.pred_len * configs.C_out)  # 用于预测未来时间步长的线性层
     
    
    def discriminator_SS(self, h):
        # h = h[:, -1, :]
        # c = self.q_c(h).unsqueeze(1)
        c = self.q_c(h)
        return c
    
    def discriminator_LSF(self, h):
        h = h
        c = self.q_c(h)
        return c
    


    def encoder(self, x_enc):
        """
        Encoder for the DMVAER model
        Model Input:
        For soft sensor task: x_enc: [B, T, C_in]
        For short-term forecasting task: x_enc: [B, T + pred_len, C_in+C_out ]
        Model Output:
        For soft sensor task: x_dec_out: [B, T, C_in], y_dec_out: [B, 1, C_out]
        For short-term forecasting task: x_dec_out: [B, T + pred_len, C_in], y_dec_out: [B, pred_len, C_out]
        """
        
        B, T, D = x_enc.shape
        h_c, _ = self.gru_c(x_enc)  # (B, T, d_model)

        logit_c = self.discriminator(h_c)  # if SS: (B, num_modes), if LSF: (B,s eq_len, num_modes)
        q_c = F.softmax(logit_c, dim=-1)
        h_T = h_c[:, -1, :]
        # Global z_s
        zs_samples, mu_zs_all, logvar_zs_all = [], [], []
        zs_samples_list = []
        for k in range(self.n_components):
            mu_logvar = self.q_zs[k](h_T)
            mu, logvar = mu_logvar[:, :self.z_global_dim], mu_logvar[:, self.z_global_dim:]
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            zs = mu + eps * std
            zs_samples.append(zs)
            mu_zs_all.append(mu)
            logvar_zs_all.append(logvar)
            zs_samples_list.append(zs.detach().cpu().numpy())
            # for k in range(self.n_components):
            #     zs = zs_samples[k]
            #     q_c = q_c[:, :, k]
            #     zs = q_c[:, :, k:k+1] * zs_samples[k]
        zs = sum([q_c[:,-1, k:k+1] * zs_samples[k] for k in range(self.n_components)])  # weighted avg
        # zs = sum([q_c[:, k:k+1] * zs_samples[k] for k in range(self.n_components)])  # weighted avg

        # Dynamic z_t
        zt = torch.zeros(B, 1, self.z_local_dim, device=x_enc.device)
        zts, mu_zt_all, logvar_zt_all = [], [], []
        hx = None
        for t in range(T):
            x_t = x_enc[:, t:t+1, :]
            zs_exp = zs.unsqueeze(1)
            inp = torch.cat([x_t, zs_exp, zt], dim=-1)
            out, hx = self.q_zt_gru(inp, hx)
            mu_zt = self.q_zt_mu(out)
            logvar_zt = self.q_zt_logvar(out)
            std_zt = torch.exp(0.5 * logvar_zt)
            eps = torch.randn_like(std_zt)
            zt = mu_zt + eps * std_zt
            zts.append(zt)
            mu_zt_all.append(mu_zt)
            logvar_zt_all.append(logvar_zt)

        zts = torch.cat(zts, dim=1)  # (B, T, z_local)

        return zts, zs, q_c, mu_zt_all, logvar_zt_all, mu_zs_all, logvar_zs_all

    def decoder(self, zts):

        h_dec_x, _ = self.decoder_gru_x(zts)
        x_recon = self.decoder_output_x(h_dec_x)  # (B, T, self.enc_in)

        # === GRU-based Decoder for y ===
        h_dec_y, _ = self.decoder_gru_y(zts)
        return x_recon, h_dec_y

    def soft_sensor(self,x_enc, x_dec, c_enc, x_mark_enc, x_mark_dec):
        zts, zs, q_c, mu_zt_all, logvar_zt_all, mu_zs_all, logvar_zs_all = self.encoder(
            x_enc)

        x_dec_out, y_dec_out = self.decoder(zts)

        y_dec_out = y_dec_out[:,-1] # [batch_size, 1, d_model]
        y_dec_out = self.projection(y_dec_out)
        
        dec_out = {
            "x_pred": x_dec_out,
            "y_pred": y_dec_out,
            "mu_zt": mu_zt_all,
            "logvar_zt": logvar_zt_all,
            "mu_zs": mu_zs_all,
            "logvar_zs": logvar_zs_all,
            "c_pred": q_c,
            "zts": zts,
            "zs": zs
        }
        
        return dec_out

    def short_term_forecasting(self, x_enc, x_dec, c_enc, x_mark_enc, x_mark_dec):
        zts, zs, q_c, mu_zt_all, logvar_zt_all, mu_zs_all, logvar_zs_all = self.encoder(x_enc)
        x_dec_out, y_dec_out = self.decoder(zts)
        y_dec_out = self.projection(y_dec_out)
        y_dec_out = y_dec_out[:,-self.configs.pred_len:,:]
        
        dec_out = {
            "x_pred": x_dec_out,
            "y_pred": y_dec_out,
            "mu_zt": mu_zt_all,
            "logvar_zt": logvar_zt_all,
            "mu_zs": mu_zs_all,
            "logvar_zs": logvar_zs_all,
            "c_pred": q_c,
            "zts": zts,
            "zs": zs
        }
        return dec_out

    def forward(self, x_enc, x_mark_enc, c_enc, batch_y, x_dec=None, x_mark_dec=None, flag='train'):
        if self.configs.task ==  'soft_sensor':
            dec_out = self.soft_sensor(x_enc, x_dec, c_enc, x_mark_enc, x_mark_dec)
        elif self.configs.task == 'short_term_forecasting':
            dec_out = self.short_term_forecasting(x_enc, x_dec, c_enc, x_mark_enc, x_mark_dec)
        return dec_out
    



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
A Novel Soft Sensor Modeling Approach Based on Difference-LSTM for Complex Industrial Process, IEEE TII 2021
Thanks to authors: Jiayi Zhou; Xiaoli Wang; Chunhua Yang; Wei Xiong
"""


class DLSTMLayer(nn.Module):
    def __init__(self, enc_in, hidden_dim):
        super().__init__()
        self.hidden = hidden_dim
      
        self.W = nn.Parameter(torch.randn(enc_in, hidden_dim * 4))
        self.U = nn.Parameter(torch.randn(hidden_dim, hidden_dim * 4))
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))
        self.Wd = nn.Parameter(torch.randn(enc_in, hidden_dim))

        self._init_weights()
    
    def _init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.normal_(param, mean=0, std=1. / np.sqrt(self.hidden))

    def forward(self, x):
        B, T, _ = x.size()

        hidden_seq = []

        h_t = torch.zeros(B, self.hidden).to(x.device)
        c_t = torch.zeros(B, self.hidden).to(x.device)

        for i in range(T):
            x_t = x[:, i, :]
            x_d = x[:, i, :] - x[:, i-1, :] if i > 0 else torch.zeros_like(x_t)

            gates = x_t @ self.W + h_t @ self.U + self.b

            i_t, f_t, g_t, o_t = torch.split(gates, self.hidden, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t + x_d @ self.Wd)


            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t)

        hidden_seq = torch.stack(hidden_seq, dim=1)
        return hidden_seq, (h_t, c_t)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task = configs.task

        self.DLSTM = nn.ModuleList([DLSTMLayer(configs.C_in if i == 0 else configs.hidden_dim, configs.hidden_dim) 
        for i in range(configs.e_layers)])

 
        if self.task == 'soft_sensor':
            self.projection = nn.Linear(configs.hidden_dim, configs.C_out)
        elif self.task == 'short_term_forecasting':
            self.projection = nn.Linear(configs.hidden_dim * configs.seq_len, configs.pred_len)
    def soft_sensor(self, x_enc):
        ht_list, ct_list = [], []
        for i in range(len(self.DLSTM)):
            x_enc, (h_t, c_t) = self.DLSTM[i](x_enc)
            ht_list.append(h_t)
            ct_list.append(c_t)
        
        dec_out = self.projection(x_enc[:, -1, :])
        
        return dec_out
    
    def short_term_forecasting(self, x_enc):
        ht_list, ct_list = [], []
        for i in range(len(self.DLSTM)):
            x_enc, (h_t, c_t) = self.DLSTM[i](x_enc)
            ht_list.append(h_t)
            ct_list.append(c_t)
        
        x_enc = x_enc.reshape(x_enc.shape[0], -1)
        dec_out = self.projection(x_enc)

        
        return dec_out.unsqueeze(-1)
    
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag='train'):
        """前向传播逻辑描述"""
        if self.task == 'short_term_forecasting':
            return self.short_term_forecasting(x_enc)
        elif self.task == 'soft_sensor':
            return self.soft_sensor(x_enc)
        else:
            raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor')

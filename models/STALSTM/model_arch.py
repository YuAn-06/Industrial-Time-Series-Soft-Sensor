from pyexpat import model
import torch
import numpy as np
from torch import nn
import math

"""
Deep Learning With Spatiotemporal Attention-Based LSTM for Industrial Soft Sensor Model Development, IEEE TII 2021

Thanks to authors: Xiaofeng Yuan; Lin Li; Yuri A. W. Shardt; Yalin Wang; Chunhua Yang
"""


class TemporalLSTM(nn.Module):
    def __init__(self, C_in, hidden_dim, seq_len, C_out):
        super(TemporalLSTM, self).__init__()
        self.C_in = C_in
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.C_out = C_out

        self.Wa = nn.Parameter(torch.Tensor(C_in, seq_len), requires_grad=True)
        self.Ua = nn.Parameter(torch.Tensor(hidden_dim , seq_len), requires_grad=True)
        self.ba = nn.Parameter(torch.Tensor(seq_len), requires_grad=True)
        self.Va = nn.Parameter(torch.Tensor(seq_len, seq_len), requires_grad=True)
        self.Softmax = nn.Softmax(dim=1)

        self.W = nn.Parameter(torch.Tensor(C_in, hidden_dim * 4), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4), requires_grad=True)

        self.Wy = nn.Parameter(torch.Tensor(C_out, hidden_dim * 4), requires_grad=True)

        self.projection = nn.Linear(hidden_dim, C_out, bias=True)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, h, y_lag):
        B, T, D = h.size()
        hidden_state_list = []
        s_t = torch.zeros(B, self.hidden_dim, device=h.device)
        LSTM_c_t = torch.zeros(B, self.hidden_dim, device=h.device)
        LSTM_h_t = torch.zeros(B, self.hidden_dim, device=h.device)

        hd  = h
        for i in range(T):
            h_t = h[:, i, :]
            
            attn = torch.tanh(h_t @ self.Wa + s_t @ self.Ua + self.ba) @ self.Va

            attn = self.Softmax(attn)
            attn = attn.unsqueeze(-1).repeat(1,1,D) # [B, T, D]
            h_t = attn * hd
            h_t = torch.sum(h_t, dim=1) # [B, D]

            gates = h_t @ self.W + LSTM_h_t @ self.U + y_lag @ self.Wy + self.bias

            i_t, f_t, o_t, g_t = torch.split(gates, self.hidden_dim, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            o_t = torch.sigmoid(o_t)
            g_t = torch.tanh(g_t)

            LSTM_c_t = f_t * LSTM_c_t + i_t * g_t
            LSTM_h_t = o_t * torch.tanh(LSTM_c_t)

            hidden_state_list.append(LSTM_h_t) # [B, T, hidden_dim]

            y_lag = self.projection(LSTM_h_t)
        hidden_state_list = torch.stack(hidden_state_list, dim=1)

        return y_lag, hidden_state_list

class SpatialLSTM(nn.Module):
    def __init__(self, C_in, hidden_dim):
        super(SpatialLSTM, self).__init__()
        self.C_in = C_in
        self.hidden_dim = hidden_dim

        self.W = nn.Parameter(torch.Tensor(self.C_in, hidden_dim * 4), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4), requires_grad=True)

        # 注意力的参数
        self.Wa = nn.Parameter(torch.Tensor(self.C_in, self.C_in), requires_grad=True)
        self.Ua = nn.Parameter(torch.Tensor(hidden_dim * 2, self.C_in), requires_grad=True)
        self.ba = nn.Parameter(torch.Tensor(self.C_in), requires_grad=True)
        self.Va = nn.Parameter(torch.Tensor(self.C_in, self.C_in), requires_grad=True)
        self.Softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, x):
        B, T, C = x.size()
        hidden_state_list = []
        h_t = torch.zeros(B, self.hidden_dim, device=x.device)
        c_t = torch.zeros(B, self.hidden_dim, device=x.device)

        for i in range(T):
            x_t = x[:, i, :]
            attn = torch.tanh(x_t @ self.Wa + torch.cat([h_t,c_t], dim=1) @ self.Ua + self.ba) @ self.Va
            attn = self.Softmax(attn) # [B, C]
            x_t = attn * x_t
            
            gates = x_t @ self.W + h_t @ self.U + self.bias

            i_t, f_t, o_t, g_t = torch.split(gates, self.hidden_dim, dim=1)

            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            o_t = torch.sigmoid(o_t)
            g_t = torch.tanh(g_t)
            
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            
            hidden_state_list.append(h_t) # [B, T, hidden_dim]
        hidden_state_list = torch.stack(hidden_state_list, dim=1)
        return hidden_state_list, (h_t, c_t), attn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task = configs.task
        self.seq_len = configs.seq_len
        self.SA = SpatialLSTM(configs.C_in, configs.SA_dim)
        self.TA = TemporalLSTM(configs.SA_dim, configs.TA_dim, configs.seq_len, configs.C_out)


    def soft_sensor(self, x_enc, y_enc):

        y_enc = y_enc[:,:self.seq_len, :]

        hidden_state_list, _ , attn = self.SA(x_enc)
        y_pred,_ = self.TA(hidden_state_list, y_enc[:,0,:])


        return y_pred



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag='train'):
        """前向传播逻辑描述"""
        if self.task == 'short_term_forecasting':
            pass
        elif self.task == 'soft_sensor':
            y_enc = x_enc[:, :self.seq_len, -self.configs.C_out:]
            x_enc = x_enc[:, :self.seq_len, :-self.configs.C_out]
            return self.soft_sensor(x_enc, y_enc)
        else:
            raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor')
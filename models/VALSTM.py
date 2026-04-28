from numpy._core import einsumfunc
import torch
import numpy as np
from torch import nn

from layers.Output_Layer import Permute


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.hidden_dim = configs.hidden_dim
        self.task = configs.task
        self.W = nn.Parameter(torch.Tensor(configs.C_in, configs.hidden_dim * 4), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(configs.hidden_dim, configs.hidden_dim * 4), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(configs.hidden_dim * 4), requires_grad=True)

        self.Wa = nn.Parameter(torch.Tensor(configs.C_in, configs.C_in), requires_grad=True)
        self.Ua = nn.Parameter(torch.Tensor(configs.hidden_dim * 2, configs.C_in), requires_grad=True)
        self.ba = nn.Parameter(torch.Tensor(configs.C_in), requires_grad=True)
        self.Va = nn.Parameter(torch.Tensor(configs.C_in, configs.C_in), requires_grad=True)
        self.Softmax = nn.Softmax(dim=1)

        if configs.task == 'soft_sensor':
            self.projection = nn.Linear(configs.hidden_dim, configs.C_out, bias=True)
        elif configs.task == 'short_term_forecasting':
            self.projection = nn.Sequential(
                nn.Linear(configs.hidden_dim, configs.d_model, bias=True),
                Permute(0, 2, 1),
                nn.Linear(1, configs.pred_len, bias=True))
        

    def encoder(self, x_enc):
        B, T, D = x_enc.size()
        h_t = torch.zeros(B,  self.hidden_dim, device=x_enc.device)
        c_t = torch.zeros(B,  self.hidden_dim, device=x_enc.device)
        h_t_list= []

        for i in range(T):
            x_t = x_enc[:, i, :]
            attn = torch.tanh(x_t @ self.Wa +  torch.cat([h_t, c_t], dim=1) @ self.Ua + self.ba) @ self.Va
            attn = self.Softmax(attn)

            x_t = attn * x_t

            gates = x_t @ self.W + h_t @ self.U + self.bias

            i_t, f_t, o_t, g_t = torch.split(gates, self.hidden_dim, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            c_t = f_t *c_t +i_t *g_t

            h_t = o_t * torch.tanh(c_t)
            h_t_list.append(h_t)

        
        h_t_list = torch.stack(h_t_list, dim=1)

        h_t_last = h_t_list[:, -1, :]

        return h_t_last


    
    def soft_sensor(self, x_enc, ):
        enc_out = self.encoder(x_enc)

        dec_out = self.projection(enc_out)
        
        return dec_out

    def short_term_forecasting(self, x_enc):
        enc_out = self.encoder(x_enc)
        enc_out = enc_out.unsqueeze(1)
        dec_out = self.projection(enc_out)
        
        return dec_out

    
    def forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec, batch_y, flag='train'):
 
        # Embedding
        if self.task == 'short_term_forecasting':                                                                                                                      
            return self.short_term_forecasting(x_enc)
        
        elif self.task == 'soft_sensor':
            return self.soft_sensor(x_enc)
        
        else:
            raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor')




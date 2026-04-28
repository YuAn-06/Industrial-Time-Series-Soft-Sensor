import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Output_Layer import Permute

class Model(nn.Module):

    def __init__(self, configs):
        """
        """
        super().__init__()
        self.task = configs.task
        self.lstm = nn.LSTM(
            configs.enc_in, configs.hidden_dim, configs.e_layers, dropout=configs.dropout, batch_first=True
        )

        if self.task == 'soft_sensor':
            self.projection = nn.Linear(configs.hidden_dim, configs.C_out, bias=True)

        else:
            self.projection = nn.Sequential(
                nn.Linear(configs.hidden_dim, configs.C_out, bias=True),
                Permute(-1,-2),
                nn.Linear(1, configs.pred_len, bias=True),
                Permute(-1,-2)
            )

    def soft_sensor(self, x_enc):
        enc_out, _ = self.lstm(x_enc)
        enc_out = self.projection(enc_out[:, -1, :])
        return enc_out
    
    def short_term_forecasting(self, x_enc):
        enc_out, _ = self.lstm(x_enc)
        enc_out = self.projection(enc_out[:,-1:])
        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag='train'):
         
        if self.task == 'short_term_forecasting':
            dec_out = self.short_term_forecasting(x_enc)
            return dec_out
        elif self.task == 'soft_sensor':
            dec_out = self.soft_sensor(x_enc)
            return dec_out
        else:
            raise ValueError(f'Invalid task type: {self.task_name}. Supporting short_term_forecasting and soft_sensor')

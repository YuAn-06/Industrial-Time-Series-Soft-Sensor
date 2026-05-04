import torch
import torch.nn as nn
import math
from layers.Embedding import PositionalEmbedding

class ISRU(nn.Module):
    """
    逆平方根激活单元 (Inverse Square Root Unit)
    对应公式 (10): ISRU(x) = x / sqrt(1 + alpha * x^2)
    """
    def __init__(self, alpha):
        super(ISRU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return x / torch.sqrt(1 + self.alpha * torch.pow(x, 2))

class GCU(nn.Module):
    """
    门控卷积神经网络单元 (Gated CNN Unit)
    对应公式 (6), (7), (8)
    """
    def __init__(self, input_dim, cnn_out_dim, kernel_size=3):
        super(GCU, self).__init__()
        # 沿时间维度进行1D卷积，不共享权重。使用 padding 保证序列长度不变。
        self.conv_f = nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_dim, 
                                kernel_size=kernel_size, padding=kernel_size//2)
        self.conv_g = nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_dim, 
                                kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        # 输入 x 形状: (batch_size, seq_len, input_dim)
        # 转置为 1D 卷积要求的形状: (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1) 
        
        Hf = torch.tanh(self.conv_f(x))      # 特征提取核
        Hg = torch.sigmoid(self.conv_g(x))   # 门控核
        Hc = Hf * Hg                         # 逐元素相乘
        
        # 返回形状: (batch_size, seq_len, cnn_out_dim)
        return Hc.permute(0, 2, 1) 


class HighwayNetwork(nn.Module):
    """
    高速公路网络 (Highway Network)
    """
    def __init__(self, input_size):
        super(HighwayNetwork, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU()
        )
        self.transform_gate = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        H = self.transform(x)
        T = self.transform_gate(x)
        return H * T + x * (1 - T)



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task = configs.task 
        self.alpha = 0.5
        self.gcu = GCU(configs.C_in, configs.d_model, kernel_size=3)
        self.enc_embedding = PositionalEmbedding(configs.d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model=configs.d_model, dim_feedforward=configs.d_ff, nhead=configs.n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=configs.e_layers)


        self.h_steps = max(1, configs.label_len)
        highway_input_dim = configs.C_in * self.h_steps
        self.highway = HighwayNetwork(highway_input_dim)

        self.linear = nn.Linear(configs.d_model, highway_input_dim)
        if self.task == 'short_term_forecasting':
            self.projection = nn.Linear(highway_input_dim, configs.pred_len)
        elif self.task == 'soft_sensor':
            self.projection = nn.Linear(highway_input_dim, configs.C_out)
        self.isru = ISRU(self.alpha)
        

    def short_term_forecasting(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag='train'):
        embedding = self.gcu(x_enc)
        embedding = self.enc_embedding(x_enc)

        enc_out = self.transformer_encoder(embedding)

        Ht = enc_out[:, -1, :]
        Hh_raw = x_enc[:, -self.h_steps:, :]
        Hh_raw = Hh_raw.reshape(Hh_raw.size(0), -1) # [batch_size, h_steps * input_dim]
        highway_out = self.highway(Hh_raw)


        combined = self.linear(Ht) + highway_out
        dec_out = self.projection(combined)

        dec_out = self.isru(dec_out)

        return dec_out.unsqueeze(-1)

    def soft_sensor(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag='train'):

        y_enc = x_enc[:, :, -self.configs.C_out:]
        last_y = torch.mean(y_enc, dim=1, keepdim=True)
        y_enc[:,-1:,:] = last_y
        x_enc = x_enc[:, :, :-self.configs.C_out]
        
        x_enc = torch.cat([x_enc, y_enc], dim=-1) # [batch_size, seq_len, c_in + c_out]

        embedding = self.gcu(x_enc)
        embedding = self.enc_embedding(x_enc)

        enc_out = self.transformer_encoder(embedding)

        Ht = enc_out[:, -1, :]
        Hh_raw = x_enc[:, -self.h_steps:, :]
        Hh_raw = Hh_raw.reshape(Hh_raw.size(0), -1) # [batch_size, h_steps * input_dim]
        highway_out = self.highway(Hh_raw)


        combined = self.linear(Ht) + highway_out
        dec_out = self.projection(combined)

        dec_out = self.isru(dec_out)

        return dec_out

    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag='train'):
        """前向传播逻辑描述"""
        if self.task == 'short_term_forecasting':
            return self.short_term_forecasting(x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag)
        elif self.task == 'soft_sensor':
            return self.soft_sensor(x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag)
        else:
            raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor')
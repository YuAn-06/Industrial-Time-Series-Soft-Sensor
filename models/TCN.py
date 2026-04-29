import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.utils import weight_norm

class ChainedCausalConv(nn.Module):
    """ 实现因果卷积后的裁剪，确保输出长度与输入一致 """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(ChainedCausalConv, self).__init__()
        # 使用 weight_norm 加快收敛
        self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.chomp = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.conv, self.chomp, self.relu)

    def forward(self, x):
        return self.net(x)

class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = ChainedCausalConv(n_inputs, n_outputs, kernel_size, stride, dilation, padding)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = ChainedCausalConv(n_outputs, n_outputs, kernel_size, stride, dilation, padding)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.dropout1, self.conv2, self.dropout2)
        
        # 如果输入输出维度不同，使用 1x1 卷积匹配维度
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        layers = []
        num_levels = len(configs.num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 典型的指数增长膨胀率
            in_channels = configs.C_in if i == 0 else configs.num_channels[i-1]
            out_channels = configs.num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, configs.kernel_size, stride=1, 
                                     dilation=dilation_size,
                                     padding=(configs.kernel_size-1) * dilation_size, 
                                     dropout=configs.dropout)]

        self.encoder = nn.Sequential(*layers)

        if configs.task == 'short_term_forecasting':
            self.projection = nn.Linear(configs.num_channels[-1], configs.pred_len)
        elif configs.task == 'soft_sensor':
            self.projection = nn.Linear(configs.num_channels[-1], configs.C_out)


        
        self.task = configs.task

    def short_term_forecasting(self, x_enc):
        x_enc = x_enc.transpose(1, 2) # 
        enc_out = self.encoder(x_enc) # [B, C, T]
        enc_out = enc_out[:, :, -1:].permute(0, 2, 1)
        enc_out = self.projection(enc_out)


        return enc_out.permute(0, 2, 1)

    def soft_sensor(self, x_enc):
        x_enc = x_enc.transpose(1, 2) # 
        enc_out = self.encoder(x_enc) # [B, C, T]
        enc_out = enc_out[:, :, -1]
        enc_out = self.projection(enc_out)

        return enc_out
        
    
    def forward(self, x_enc,x_mark_enc,x_dec,x_mark_dec, batch_y,flag = 'train'):
        if self.task == 'short_term_forecasting':
            return self.short_term_forecasting(x_enc)
        
        elif self.task == 'soft_sensor':
            return self.soft_sensor(x_enc)
           
        else:
            raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor')

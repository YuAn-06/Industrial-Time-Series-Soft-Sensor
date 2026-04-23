import torch
import torch.nn as nn
import torch.nn.functional as F


"""
HSAM_dGRUs model, Migrating from Keras to PyTorch version, author: Yuan_L, 2026
"""

from layers.Embedding import DataEmbedding
import numpy as np

from layers.HybridAttention import *



class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder, self).__init__()
        self.configs = configs
        self.MCMSA = MCMSALayer(
            configs,
            MultiChannelMSA(
                mask_flag = False,  # 对于软传感器任务，不需要掩码
                factor = self.configs.factor,
                attention_dropout = self.configs.dropout,
                output_attention = False
            ),
            d_model = self.configs.d_model,
            n_heads = self.configs.n_heads,

            )

        d_keys = self.configs.d_model // self.configs.n_heads
        d_values =  self.configs.d_model // self.configs.n_heads
   

        self.QRSAM_List = nn.ModuleList([QualityRelatedAttention(configs) for i in range(self.configs.n_heads)])


    def forward(self, x_enc, y_enc):
        x_enc, attn = self.MCMSA(x_enc,x_enc,x_enc, attn_mask=None) # [B,T,h, d_keys]
        f_list = []
        for i in range(self.configs.n_heads):
            QM_layer = self.QRSAM_List[i]
            f, attn = QM_layer(queries=y_enc, keys=x_enc[:, :, i])
            f_list.append(f)
        f_list = torch.stack(f_list, dim=1) # [B, n_heads, T, d_keys]
        return f_list, attn



class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()
        self.configs = configs

        d_keys = self.configs.d_model // self.configs.n_heads
 

        self.dGRUs = DistributedGRUs(
            configs,
            d_keys = d_keys,
            )

    def forward(self, x_dec):
        x_dec = self.dGRUs(x_dec)
        return x_dec

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        try:
            self.activation = getattr(nn, self.configs.activation)()
        except:
            raise NameError(
                f"Invalid activation name '{self.configs.activation}'. Please Checkout activation name")
                
        # self.embedding = DataEmbedding(configs.C_in, configs.d_model, freq = configs.freq, dropout = configs.dropout)
        # self.y_embedding = DataEmbedding(configs.C_out, configs.C_out, freq = configs.freq, dropout = configs.dropout)
        self.encoder = Encoder(configs)
        self.decoder = Decoder(configs).to(configs.device)
        # self.embedding = nn.Linear(configs.C_in, configs.d_model)
        self.norm = nn.LayerNorm(configs.d_model)  # 添加层归一化以稳定训练
        
        self.projection = nn.Linear(configs.n_heads * configs.hidden_dim, configs.C_out)


    def soft_sensor(self, x_enc, x_mark_enc, y_enc):



        x_enc, attn = self.encoder(x_enc,  y_enc)
        
        x_dec = self.decoder(x_enc)
        x_dec = self.projection(x_dec)
        x_dec = torch.sigmoid(x_dec)
        return  x_dec


    def short_term_forecasting(self, x_enc, x_mark_enc, y_enc):

        return

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag='train'):

        B, T, D = x_enc.shape
        
        y_enc = x_enc[:,:, -self.configs.C_out:]
        x_enc = x_enc[:,:, :-self.configs.C_out]


        # new_y_enc = torch.zeros_like(y_enc)
        # new_y_enc[:, 1:] = y_enc[:, :-1] # 将原 tensor 的 [:-1] 部分赋给新 tensor 的 [1:]
        # new_y_enc[:, 0] = y_enc[:, 0]
        # y_enc = new_y_enc
        y_enc = y_enc.squeeze(-1)
        y_enc = y_enc[:,-2:-1]

        y_enc = y_enc.repeat(1, T)

        if self.configs.task == 'soft_sensor':
            return self.soft_sensor(x_enc, x_mark_enc, y_enc)
        elif self.configs.task == 'short_term_forecasting':
            return self.short_term_forecasting(x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y)

        else:
            raise NameError(
                f'Invalid task name \'{self.configs.task} \'. Please Checkout task name')
    
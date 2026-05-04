import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Novel Distributed GRUs Based on Hybrid Self-Attention Mechanism for Dynamic Soft Sensing, IEEE TASE 2023

Thanks to authors: Yan-Lin He; Xing-Yuan Li; Yuan Xu; Qun-Xiong Zhu; Shan Lu
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

        # d_keys = self.configs.d_model // self.configs.n_heads
        # d_values =  self.configs.d_model // self.configs.n_heads
   

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
        self.task = configs.task
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

        self.lsf_projection = nn.Linear(configs.n_heads * configs.hidden_dim, configs.pred_len * configs.C_out)

    def soft_sensor(self, x_enc, x_mark_enc, y_enc):



        x_enc, attn = self.encoder(x_enc,  y_enc)
        
        x_dec = self.decoder(x_enc)
        x_dec = self.projection(x_dec)
        x_dec = torch.sigmoid(x_dec)
        return  x_dec

    def short_term_forecasting(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y):
        B, T, D = x_enc.shape

        y_enc_query = x_enc[:, -1, -1:].repeat(1, T)

        f_list, attn = self.encoder(x_enc, y_enc_query)

        x_dec = self.decoder(f_list)

        x_dec = self.lsf_projection(x_dec)

        #  [B, T, D]
        B = x_enc.shape[0]
        x_dec = x_dec.view(B, self.configs.pred_len, self.configs.C_out)

        return x_dec


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag='train'):

        x_features = x_enc[:, :, :self.configs.enc_in]
        y_label = x_enc[:, :, -self.configs.C_out:]

        y_enc_query = y_label.squeeze(-1)  # [B, T]

        if self.configs.task == 'soft_sensor':
            y_query_single = y_enc_query[:, -1:].repeat(1, x_features.shape[1])
            return self.soft_sensor(x_features, x_mark_enc, y_query_single)

        elif self.configs.task == 'short_term_forecasting':
            return self.short_term_forecasting(x_features, x_mark_enc, x_dec, x_mark_dec, batch_y)
        else:
             raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor')
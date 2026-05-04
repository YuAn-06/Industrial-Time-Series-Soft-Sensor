from turtle import forward
import torch 
import torch.nn as nn

from torch.nn import functional as F
from layers.Embedding import *
from collections import namedtuple
from layers.Output_Layer import *
from layers.Transformer_EncDec import *
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Output_Layer import Flatten_head

"""
# Fredformer: Frequency Debiased Transformer for Time Series Forecasting
# https://ieeexplore.ieee.org/document/10699388
"""

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task = configs.task
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,configs.dropout)
        
        padding = configs.stride
        
        self.patch_embedding = PatchFourierEmbedding(configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)
        
        self.encoder =  Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        
        patch_num = int((configs.seq_len - configs.patch_len)/configs.stride + 1)
        self.flatten_dim = configs.pd_model * 2 * patch_num
        
        
        
        self.attention_projection = nn.Linear(configs.d_model, configs.pd_model * 2)
        self.real_projection = nn.Linear(configs.pd_model*2, configs.pd_model*2)
        self.img_projection = nn.Linear(configs.pd_model*2, configs.pd_model*2)

        self.flatten_head_1 = Flatten_head(configs.individual, configs.C_in, self.flatten_dim, configs.pred_len, configs.dropout)
        self.flatten_head_2 = Flatten_head(configs.individual, configs.C_in, self.flatten_dim, configs.pred_len, configs.dropout)
        
        self.dropout = nn.Dropout(configs.dropout)
        
        self.projection = nn.Linear(configs.pred_len *2, configs.pred_len)
        
        

    def FourierModeling(self, B, enc_out):
        """
        Frequency domain modeling block
        Args:
            B: batch size
            enc_out: [bs* patch_num, n_vars, d_model]
        """
        
        BP, N, D = enc_out.shape
        
        P = int(BP / B)
        
        enc_out = self.dropout(enc_out) 
        
        enc_out = self.attention_projection(enc_out) # [bs* patch_num, n_vars, pd_model * 2]
        
        enc_out_real = self.real_projection(enc_out) # [bs* patch_num n_vars, pd_model * 2]
        enc_out_img = self.img_projection(enc_out) # [bs* patch_num, n_vars, pd_model * 2]
        
        enc_out_real = torch.reshape(enc_out_real, (B ,P, N, enc_out_real.shape[-1])) # shape: [bs, patch_num, n_vars, pd_model * 2]
        enc_out_img = torch.reshape(enc_out_img, (B ,P, N, enc_out_img.shape[-1])) # shape: [bs, patch_num, n_vars, pd_model * 2]
        
        enc_out_real = enc_out_real.permute(0,2,1,3) # shape: [bs, n_vars, patch_num, pd_model * 2]
        enc_out_img = enc_out_img.permute(0,2,1,3) # shape: [bs, n_vars, patch_num, pd_model * 2]

        enc_out_real = self.flatten_head_1(enc_out_real) # [bs x nvars x target_window] 
        enc_out_img = self.flatten_head_2(enc_out_img) # [bs x nvars x target_window] 

        enc_out_fft =torch.fft.ifft(torch.complex(enc_out_real, enc_out_img))

        enc_out_real = enc_out_fft.real
        enc_out_img = enc_out_fft.imag
        
        return torch.cat((enc_out_real, enc_out_img), dim=-1)
    
    
    
    def short_term_forecasting(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        B, L, N = x_enc.shape
        
        x_enc = self.patch_embedding(x_enc) # [bs* patch_num, n_vars, d_model]
        
        enc_out, attns = self.encoder(x_enc) # [bs* patch_num, n_vars, d_model]
        
        enc_out = self.FourierModeling(B, enc_out)
        
        enc_out = self.projection(enc_out)

        return enc_out.permute(0,2,1)
    
    
    def soft_sensor(self, x_enc,x_mark_enc, x_dec, x_mark_dec):
        
        B, L, N = x_enc.shape
        
        x_enc = self.patch_embedding(x_enc) # [bs* patch_num, n_vars, d_model]
        
        enc_out, attns = self.encoder(x_enc) # [bs* patch_num, n_vars, d_model]
        
        enc_out = self.FourierModeling(B, enc_out) # [bs* patch_num, n_vars, 2 * pred_len]
        
        enc_out = self.projection(enc_out)
        
        
        return enc_out.permute(0,2,1)


    def forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec, batch_y, flag='train'):
        if self.task == 'short_term_forecasting':
            return self.short_term_forecasting(x_enc, x_mark_enc, x_dec, x_mark_dec)

        elif self.task == 'soft_sensor':
            return self.soft_sensor(x_enc, x_mark_enc, x_dec, x_mark_dec)[:,:,-self.configs.C_out]

        else:
            raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor')

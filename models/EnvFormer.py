
from turtle import forward
import torch 
import torch.nn as nn

from torch.nn import functional as F
from layers.Embedding import *
from collections import namedtuple
from layers.Output_Layer import *
from layers.EnvFormer_EncDec import *
from layers.SelfAttention_Family import FullAttention, AttentionLayer
# EnvFormer: A Decomposition-Based Transformer for Multistep Burn-Through Point Prediction in the Sintering Process
# https://ieeexplore.ieee.org/document/10699388

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task = configs.task
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,configs.dropout)
        self.EnvD = EnvDecomp(configs.kernel_size)

        if configs.seq_len < configs.label_len * 2:
            raise ValueError('seq_len must be greater than label_len * 2')

        self.encoder =  Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    configs.kernel_size,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        if self.configs.task == 'short_term_forecasting':
            self.decoder = Decoder(
                    [
                        DecoderLayer(
                            AttentionLayer(
                                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                                configs.d_model, configs.n_heads),
                            AttentionLayer(
                                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                                configs.d_model, configs.n_heads),
                            configs.d_model,
                            configs.kernel_size,
                            configs.C_in,
                            d_ff = configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation,
                            
                        )
                        for l in range(configs.d_layers)
                    ],norm_layer=torch.nn.LayerNorm(configs.d_model),
                    projection=nn.Linear(configs.d_model, configs.C_out, bias=True)
                )
        else:
            self.decoder = nn.Linear(configs.d_model, configs.C_out, bias=True)
        
        
    def short_term_forecasting(self,x_enc,x_mark_enc,x_dec,x_mark_dec):
        B, L , D = x_enc.size()
        x_enc_trend, x_enc_residual = self.EnvD(x_enc)
        
      
        x_dec_residual = torch.concat([x_enc_residual[:,int(L/2):],x_dec[:,-self.configs.pred_len:]],dim=1)
        
        mean = torch.mean(x_enc,dim=1).unsqueeze(1).repeat(1, self.configs.pred_len, 1)
        x_dec_trend = torch.concat([x_enc_trend[:,int(L/2):],mean],dim=1)
        
        x_enc = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attn = self.encoder(x_enc,x_mark_enc, x_dec, x_mark_dec)
        
        x_dec_residual = self.dec_embedding(x_dec_residual, x_mark_dec)
        dec_out = self.decoder(x_dec_residual ,enc_out, x_dec_trend)
        
        return dec_out

    def soft_sensor(self,x_enc,x_mark_enc,x_dec=None,x_mark_dec=None):
        B, L , D = x_enc.size()
        x_enc_trend, x_enc_residual = self.EnvD(x_enc)
        
      
    
        
        mean = torch.mean(x_enc,dim=1).unsqueeze(1).repeat(1, self.configs.pred_len, 1)
     
        
        x_enc = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attn = self.encoder(x_enc, x_mark_enc, x_dec, x_mark_dec)
        enc_out = enc_out[:,-1,:]
        dec_out = self.decoder(enc_out)
        return dec_out
        
    def forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec, batch_y, flag='train'):
       
        if self.task == 'short_term_forecasting':
            dec_out = self.short_term_forecasting(x_enc, x_mark_enc, x_dec, x_mark_dec) # [B, L/2 + Pred_len, d_model]
            
        
            return dec_out
        elif self.task == 'soft_sensor':
            dec_out = self.soft_sensor(x_enc, x_mark_enc)
            return dec_out
        else:
            raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor')


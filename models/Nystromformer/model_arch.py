import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.NystromAttention import NystromAttention
from layers.Embedding import DataEmbedding
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
        self.configs = configs
        if configs.num_landmarks is None or configs.num_landmarks > configs.C_in:
            raise ValueError('num_landmarks must be less than or equal to C_in')
        self.task = self.configs.task
        self.encoder =  Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        NystromAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False, num_landmarks=configs.num_landmarks), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        if self.task == 'short_term_forecasting':
            # Prediction Head
            self.projection = nn.Linear(configs.seq_len * configs.d_model, configs.C_in * configs.pred_len, bias=True)
        elif self.task == 'soft_sensor':
            self.projection = nn.Linear(configs.d_model, configs.C_out, bias=True)

    
    
    def shor_term_forecasting(self, x_enc, x_mark_enc=None,x_dec=None, x_mark_dec=None):
        
        # Normalization
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out.view(enc_out.size(0), -1)
        enc_out = self.projection(enc_out).view(enc_out.size(0), self.configs.pred_len, self.configs.C_in)
        
        # enc_out = enc_out.permute(0,2,1)
        
        
        # enc_out = self.projection(enc_out)
        
        # enc_out = enc_out.permute(0,2,1)
        return enc_out
    def soft_sensor(self, x_enc, x_mark_enc=None,x_dec=None, x_mark_dec=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        enc_out = enc_out[:,-1] # [batch_size, 1, d_model]
        
        enc_out = self.projection(enc_out)
       
        return enc_out

    def forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec, batch_y, flag='train'):
        if self.task == 'short_term_forecasting':
            enc_out = self.shor_term_forecasting(x_enc, x_mark_enc, x_dec, x_mark_dec) # [batch_size, seq_len, C_out]
            return enc_out
        elif self.task == 'soft_sensor':
            enc_out = self.soft_sensor(x_enc, x_mark_enc,x_dec, x_mark_dec) #[batch_size, 1, C_out]
            return enc_out
        
        else:
            raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor')
    
    




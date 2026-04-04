import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embedding import DataEmbedding, DataEmbedding_inverted
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        self.configs = configs

        self.encoder = Encoder(
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
        
        if self.configs.task == 'short_term_forecasting':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        elif self.configs.task == 'soft_sensor':
            self.projection = nn.Linear(configs.C_in, configs.C_out, bias=True)


    def short_term_forecasting(self, x_enc, x_dec, x_mark_enc, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))

        return dec_out

    def soft_sensor(self, x_enc, x_dec, x_mark_enc, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
       


        # dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = enc_out.permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.d_model, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.d_model, 1)) # [B, D, N]
        dec_out = dec_out[:,-1] # 只选择最后一个Time_step

        dec_out = self.projection(dec_out)

        return dec_out

        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag='train'):
         
        if self.configs.task == 'short_term_forecasting':
            dec_out = self.short_term_forecasting(x_enc, x_dec, x_mark_enc, x_mark_dec)
        elif self.configs.task == 'soft_sensor':
            dec_out = self.soft_sensor(x_enc, x_dec, x_mark_enc, x_mark_dec)
        
        return dec_out
       

        
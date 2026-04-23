import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embedding import DataEmbedding, PatchEmbedding
import numpy as np



class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x) # [bs, nvars, d_model*patch_num]
        x = self.linear(x) # [bs, nvars, target_window]
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task = configs.task
        self.C_out = configs.C_out
        self.pred_len = configs.pred_len

        padding = configs.stride
        
        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len=configs.patch_len, stride=configs.stride, padding=padding, dropout=configs.dropout )


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
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        self.head_nf = configs.d_model * \
                       int((configs.seq_len - configs.patch_len) / configs.stride + 2)

        if self.task == 'short_term_forecasting':
            self.head_nf = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout)
        elif self.task == 'soft_sensor':
            self.head_nf = FlattenHead(configs.enc_in, self.head_nf, configs.C_out, head_dropout=configs.dropout)
        else:
            raise ValueError("task type not supported")


    def short_term_forecasting(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        means = x_enc.mean(1, keepdim=True).detach()

        x_enc = x_enc - means

        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)

        x_enc /= stdev

        # start patching

        x_enc = x_enc.permute(0, 2, 1) # [B,C,T]

        x_enc, n_vars = self.patch_embedding(x_enc) # [B*C, patch_num, d_model]

        enc_out, attns = self.encoder(x_enc) # [B*C, patch_num, d_model]

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])) # [B, C, patch_num, d_model]
        
        enc_out = enc_out.permute(0, 1, 3, 2) # [B, C, d_model, patch_num]

        dec_out = self.head_nf(enc_out) #  [B,C,pred_len]

        dec_out = dec_out.permute(0, 2, 1) # [B, C, pred_len]


        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def soft_sensor(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()

        x_enc = x_enc - means

        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)

        x_enc /= stdev

        # start patching

        x_enc = x_enc.permute(0, 2, 1) # [B,C,T]

        x_enc, n_vars = self.patch_embedding(x_enc) # [B*C, patch_num, d_model]

        enc_out, attns = self.encoder(x_enc) # [B*C, patch_num, d_model]

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])) # [B, C, patch_num, d_model]
        
        enc_out = enc_out.permute(0, 1, 3, 2) # [B, C, d_model, patch_num]

        dec_out = self.head_nf(enc_out) #  [B,C,pred_len]

        dec_out = dec_out.permute(0, 2, 1) # [B, pred_len, C]

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.C_out, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.C_out, 1))

        dec_out = dec_out[:, :, -self.C_out:]
        return dec_out.squeeze(-1)


    def forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec, batch_y, flag='train'):
        if self.task == 'short_term_forecasting':
            return self.short_term_forecasting(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task == 'soft_sensor':
            return self.soft_sensor(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor')


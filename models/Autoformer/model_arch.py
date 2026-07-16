import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embedding import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task = configs.task
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        if self.task == 'short_term_forecasting':
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.C_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.C_out, bias=True)
            )
        if self.task == 'soft_sensor':
            self.projection = nn.Linear(
                configs.d_model, configs.C_out, bias=True)
    def short_term_forecasting(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                             x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enC_out = self.enc_embedding(x_enc, x_mark_enc)
        enC_out, attns = self.encoder(enC_out, attn_mask=None)
        # dec
        deC_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(deC_out, enC_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        deC_out = trend_part + seasonal_part
        return deC_out
    
    def soft_sensor(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        seasonal_init, trend_init = self.decomp(x_enc)
        enC_out = self.enc_embedding(x_enc, x_mark_enc)
        enC_out, attns = self.encoder(enC_out, attn_mask=None) # [B, L, D]
        deC_out = self.projection(enC_out)
        return deC_out


    def forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec, batch_y, flag='train'):
        if self.task == 'short_term_forecasting':
            deC_out = self.short_term_forecasting(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return deC_out[:, -self.pred_len:, :]


        elif self.task == 'soft_sensor':
            deC_out = self.soft_sensor(
                x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            return deC_out[:, -1, :]

        else:
            raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor')


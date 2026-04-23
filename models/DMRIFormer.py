
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.DMRIFormer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DMRIAttention, DMRIAttentionLayer
from layers.Embedding import DataEmbedding


"""Data Mode Related Interpretable Transformer Network for Predictive Modeling and Key Sample Analysis in Industrial Processes, IEEE Transactions on Industrial Informatics, 2022
Thanks to authors: Diju Liu, Yalin Wang, et al"""


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task = configs.task
        self.C_in = configs.C_in
        self.C_out = configs.C_out
        self.d_model = configs.d_model

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    DMRIAttentionLayer(
                        DMRIAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        

        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)

        if self.task == 'short_term_forecasting':
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        DMRIAttentionLayer(
                            DMRIAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        DMRIAttentionLayer(
                            DMRIAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.C_out, bias=True)
            )
        else:
            self.decoder = nn.Linear(self.d_model, self.C_out)

    def short_term_forecasting(self, x_enc,x_mark_enc, c_enc, c_dec, x_dec,x_mark_dec, batch_y, flag='train'):



        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attn = self.encoder(enc_out, c_enc, attn_mask=None)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, c_enc, c_dec, x_mask=None, cross_mask=None)

        return dec_out

    def soft_sensor(self, x_enc,x_mark_enc, c_enc, x_dec, x_mark_dec, batch_y, flag='train'):
        
     

        enc_out = self.enc_embedding(x_enc, x_mark_enc, c_enc)
        enc_out, attn = self.encoder(enc_out, c_enc, attn_mask=None)
        dec_out = self.decoder(dec_out)

        return dec_out


    def forward(self, x_enc,x_mark_enc, c_enc, x_dec,x_mark_dec, batch_y, flag='train'):
        c_enc = torch.argmax(c_enc, dim=-1) # convert one-hot to index

        c_dec = c_enc[:,self.configs.seq_len-self.configs.label_len:,]
        c_enc = c_enc[:, : self.configs.seq_len]

        x_dec = x_dec[:,:, :-self.C_out]
        if self.task == 'short_term_forecasting':
            dec_out = self.short_term_forecasting(x_enc,x_mark_enc, c_enc, c_dec, x_dec,x_mark_dec, batch_y, flag='train')
            dec_out = dec_out.repeat(1, 1, self.C_in + self.C_out)
            return dec_out
        elif self.task == 'soft_sensor':
            dec_out = self.soft_sensor(x_enc,x_mark_enc, c_enc, x_dec,x_mark_dec, batch_y, flag='train')
            return dec_out
        else:
            raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor')


        

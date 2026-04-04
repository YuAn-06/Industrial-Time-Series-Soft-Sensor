
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.DMRIFormer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DMRIAttention, DMRIAttentionLayer
from layers.Embedding import DataEmbedding


"""Data Mode Related Interpretable Transformer Network for Predictive Modeling and Key Sample Analysis in Industrial Processes, IEEE Transactions on Industrial Informatics, 2022
Thanks to authors: Diju Liu, Yalin Wang, et al"""


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.task = config.task
        self.C_in = config.C_in
        self.C_out = config.C_out
        self.d_model = config.d_model

        self.enc_embedding = DataEmbedding(config.enc_in, config.d_model, config.embed, config.freq,
                                           config.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    DMRIAttentionLayer(
                        DMRIAttention(False, config.factor, attention_dropout=config.dropout,
                                      output_attention=False), config.d_model, config.n_heads),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation
                ) for l in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )
        

        self.dec_embedding = DataEmbedding(config.dec_in, config.d_model, config.embed, config.freq,
                                            config.dropout)

        if self.task == 'short_term_forecasting':
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        DMRIAttentionLayer(
                            DMRIAttention(False, config.factor, attention_dropout=config.dropout,
                                        output_attention=False),
                            config.d_model, config.n_heads),
                        DMRIAttentionLayer(
                            DMRIAttention(False, config.factor, attention_dropout=config.dropout,
                                        output_attention=False),
                            config.d_model, config.n_heads),
                        config.d_model,
                        config.d_ff,
                        dropout=config.dropout,
                        activation=config.activation,
                    )
                    for l in range(config.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(config.d_model),
                projection=nn.Linear(config.d_model, config.C_out, bias=True)
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

        c_dec = c_enc[:,self.config.seq_len-self.config.label_len:,]
        c_enc = c_enc[:, : self.config.seq_len]

        x_dec = x_dec[:,:, :-self.C_out]
        if self.task == 'short_term_forecasting':
            dec_out = self.short_term_forecasting(x_enc,x_mark_enc, c_enc, c_dec, x_dec,x_mark_dec, batch_y, flag='train')
            dec_out = dec_out.repeat(1, 1, self.C_in + self.C_out)
            return dec_out
        elif self.task == 'soft_sensor':
            dec_out = self.soft_sensor(x_enc,x_mark_enc, c_enc, x_dec,x_mark_dec, batch_y, flag='train')
            return dec_out
        else:
            raise ValueError("task is not defined")


        

from decimal import Decimal
from pickle import FALSE
from sympy import false
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import TCVAE_Encoder, TCVAE_EncoderLayer

from layers.SelfAttention_Family import FullAttention, AttentionLayer, TCVAEAttention
from layers.Embedding import DataEmbedding
import numpy as np

from torch.nn import MultiheadAttention
from utils.masking import TCVAECausalMask


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        # self.c_dim = configs.c_dim
        try:
            self.activation = getattr(nn, self.configs.activation)()
        except:
            raise NameError(
                f'Invalid activation name \'{self.configs.activation} \'. Please Checkout activation name')
        self.encoder = TCVAE_Encoder(
            [
                TCVAE_EncoderLayer(
                    AttentionLayer(
                        TCVAEAttention(True, configs.factor, attention_dropout=configs.dropout,
                                       output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.prior_net = nn.Sequential(
            nn.Linear(configs.d_model, 16),
            self.activation,
            nn.Linear(16, configs.z_dim*2)
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(configs.d_model, configs.z_dim*2),
            self.activation,

        )

        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.enc_attn = nn.MultiheadAttention(
            configs.d_model, configs.n_heads, dropout=configs.dropout, batch_first=True)
        self.dec_attn = nn.MultiheadAttention(
            configs.d_model, configs.n_heads, dropout=configs.dropout, batch_first=True)

        if self.configs.use_cuda:
            self.rand_query = nn.Parameter(torch.randn(
                1, self.configs.seq_len+self.configs.pred_len, self.configs.d_model), requires_grad=True).cuda()
        else:
            self.rand_query = nn.Parameter(torch.randn(
                1, self.configs.seq_len+self.configs.pred_len, self.configs.d_model), requires_grad=True)

        self.z_embedding = nn.Linear(configs.z_dim, configs.d_model)
        self.projection = nn.Linear(
            configs.z_dim + configs.d_model, configs.C_out)

    def TCVAEencoder(self, x_enc, x_mark_enc):
        """
        Return result of TCVAE encoder, including prior and posterior variables and encoder outputs
        
        """

        # encoder_prior p(z|x)
        mask_future = torch.zeros(
            (x_enc.size(0), self.configs.pred_len, x_enc.size(-1)), device=x_enc.device)
        x_enc_p = torch.cat(
            [x_enc[:, :-self.configs.pred_len], mask_future], dim=1)
        enc_out_p = self.enc_embedding(x_enc_p, x_mark_enc)
        enc_out_p, attns, enc_out_p_list = self.encoder(
            enc_out_p, attn_mask=None, x_d=None)

        # encoder_posterior q(z|x,y)
        x_enc_q = x_enc
        enc_out_q = self.enc_embedding(x_enc_q, x_mark_enc)
        enc_out_q, attns, enc_out_q_list = self.encoder(
            enc_out_q, attn_mask=None, x_d=None)

        rand_query_expanded = self.rand_query.expand(x_enc.size(0), -1, -1)

        # MSA to get prior  p(z|x)
        attn_out_p, _ = self.enc_attn(
            rand_query_expanded, enc_out_p, enc_out_p)
        p_mean_logvar = self.prior_net(attn_out_p)
        p_mean, p_logvar = p_mean_logvar.chunk(2, dim=-1)
        z_p = reparameterize(p_mean, p_logvar)

        # MSA to get posterior q(z|x,y)
        attn_out_q, _ = self.enc_attn(
            rand_query_expanded, enc_out_q, enc_out_q)
        q_mean_logvar = self.posterior_net(attn_out_q)
        q_mean, q_logvar = q_mean_logvar.chunk(2, dim=-1)
        z_q = reparameterize(q_mean, q_logvar)
        outputs = {
            'p_mean': p_mean,
            'p_logvar': p_logvar,
            'q_mean': q_mean,
            'q_logvar': q_logvar,
        }

        return z_p, z_q,  outputs, enc_out_q_list

    def TCVAEdecoder(self, x_dec, x_mark_dec, z_p, z_q, enc_out_q_list):

        B, L, _ = x_dec.size()

        mask_future = torch.zeros(
            (x_dec.size(0), self.configs.pred_len, x_dec.size(-1)), device=x_dec.device)

        x_dec = torch.cat(
            [x_dec[:, :-self.configs.pred_len], mask_future], dim=1)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        attn_mask = TCVAECausalMask(B, L, device=dec_out.device)
        
        dec_out, _, _ = self.encoder(
        dec_out, x_d=enc_out_q_list, attn_mask=attn_mask)
        

        dec_out_train = torch.concat([z_q, dec_out], dim=-1)
        dec_out_infer = torch.concat([z_p, dec_out], dim=-1)

        return dec_out_train, dec_out_infer

    def short_term_forecasting(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        z_p, z_q, outputs, enc_out_q_list = self.TCVAEencoder(
            x_dec, x_mark_dec)
        dec_out_train, dec_out_infer = self.TCVAEdecoder(
            x_dec, x_mark_dec,  z_p, z_q, enc_out_q_list,)

        dec_out_train = self.projection(dec_out_train)
        dec_out_infer = self.projection(dec_out_infer)

        outputs['dec_out_train'] = dec_out_train
        outputs['dec_out_infer'] = dec_out_infer

        return outputs

    def forward(self, x_enc,x_enc_mark,x_dec,x_dec_mark, batch_y,flag = 'train'):
        if self.configs.task == 'short_term_forecasting':
            dec_out = self.short_term_forecasting(
                x_enc, x_mark_enc, x_dec, x_mark_dec)

            return dec_out

        else:
            raise ValueError(
                "Invalid task type. Please choose 'short_term_forecasting'.")


def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std

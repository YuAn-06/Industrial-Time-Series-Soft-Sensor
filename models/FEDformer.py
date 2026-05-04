import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embedding import DataEmbedding
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, configs, version='fourier', mode_select='random', modes=32):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(Model, self).__init__()
        self.task = configs.task
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes

        self.C_out = configs.C_out

        # Decomp
        self.decomp = series_decomp(configs.moving_avg)
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.modes,
                                                  ich=configs.d_model,
                                                  base='legendre',
                                                  activation='tanh')
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            n_heads=configs.n_heads,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            n_heads=configs.n_heads,
                                            seq_len=self.seq_len // 2 + self.pred_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                      out_channels=configs.d_model,
                                                      seq_len_q=self.seq_len // 2 + self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.modes,
                                                      mode_select_method=self.mode_select,
                                                      num_heads=configs.n_heads)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
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
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.C_in,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.C_in, bias=True)
        )

        if self.task == 'soft_sensor':
            self.projection = nn.Linear(configs.d_model, configs.C_out, bias=True)

    def short_term_forecasting(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)  # x - moving_avg, moving_avg
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out[:, -self.pred_len:, -self.C_out:]

    def soft_sensor(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = self.projection(enc_out[:, -1, :]) #
        
        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag='train'):
        """前向传播逻辑描述"""
        if self.task == 'short_term_forecasting':
            dec_out = self.short_term_forecasting(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        elif self.task == 'soft_sensor':
            dec_out = self.soft_sensor(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        else:
            raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor')
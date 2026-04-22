from turtle import forward
import torch
import torch
import numpy as np
from torch import nn
from layers.Embedding import *


"""
Dual attention-based encoder–decoder: A customized sequence-to-sequence learning for soft sensor development, IEEE TNNLS 2020

Thanks to authors: Liangjun Feng, Chunhui Zhao, Youxian Sun
"""


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.encoder = nn.GRUCell(args.enc_in, hidden_size=args.d_model)
        self.Wt = nn.Linear(args.d_model, args.seq_len, bias=False)
        self.Ut = nn.Linear(args.seq_len, args.seq_len, bias=False)
        self.Zt = nn.Parameter(torch.randn(args.seq_len, 1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_dec_mark=None):
        B, T, D = x_enc.shape
        x = x_enc[:, :, :-1]  # [batch_size, seq_len, C_in]
        y = x_enc[:, :, -1:]  # [batch_size, seq_len, C_out]
        ht = torch.randn(B, self.args.d_model).to(x.device)
        H = []
        for t in range(T):
            # [batch_size,C_in,  hidden_dim]
            ht_temp = ht.unsqueeze(-1).repeat(1, 1, D -
                                              1).permute(0, 2, 1).to(x.device)
            # et = self.Zt.T() @ nn.tanh(self.Wt(ht_temp) + self.Ut @ x)
            # [batch_size, C_in]
            et = self.tanh(self.Wt(ht_temp) +
                           self.Ut(x.permute(0, 2, 1))) @ self.Zt
            at = self.softmax(et)
            xt = x[:, t, :]  # [batch_size, C_in]
            xt = xt * at.squeeze(-1)  # [batch_size, C_in]
            ht = self.encoder(xt, ht)  # [batch_size, hidden_dim]
            H.append(ht.unsqueeze(1))
        H = torch.cat(H, dim=1)  # [batch_size, seq_len, hidden_dim]

        return H, y


class Decoder_SS(nn.Module):
    def __init__(self, args):
        super(Decoder_SS, self).__init__()
        self.args = args
        self.decoder = nn.GRUCell(args.d_model, hidden_size=args.d_ff)
        self.Wd = nn.Linear(args.d_ff, args.d_model, bias=False)
        self.Ud = nn.Linear(args.d_model, args.d_model, bias=False)

        self.Vd = nn.Parameter(torch.randn(args.d_model, 1))
        self.GRU_decoder_l0 = nn.GRU(
            args.d_model, args.d_ff,  batch_first=True)
        self.GRU_decoder_l1 = nn.GRU(
            args.d_model, args.d_ff,  batch_first=True)
        self.GRU_list = [nn.GRU(args.d_model+1, args.d_ff, 1, batch_first=True)
                         for _ in range(self.args.C_out - 1)]
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.projection = nn.Linear(args.d_ff, args.C_out)
    def forward(self, H, y_mark):
        B, _, _ = H.shape
        st = torch.randn(B, self.args.d_ff).to(H.device)
        y_pred_list = []

        # Layer 0
        st, _ = self.GRU_decoder_l0(H, st.unsqueeze(0))
        e = self.tanh(self.Wd(st) + self.Ud(H)) @ self.Vd
        b = self.softmax(e)  # [batch_size, seq_len, 1]
        v = b * H  # v1
        v = torch.sum(v, dim=1)  # [batch_size, hidden_dim]

        # Layer 1
        s_t, _ = self.GRU_decoder_l1(v.unsqueeze(1))
        y_pred = self.projection(s_t)
        y_pred_list.append(y_pred)
        e = self.tanh(self.Wd(s_t) + self.Ud(H)) @ self.Vd
        b = self.softmax(e)  # [batch_size, seq_len, 1]
        v = b * H  # v2
        v = torch.sum(v, dim=1)  # [batch_size, hidden_dim]

        # Layer 2 and so on
        for i in range(self.args.C_out-1):
            vy = torch.cat((v, y_pred), dim=-1)
            s_t, _ = self.GRU_list[i](vy.unsqueeze(1))
            y_pred = self.projection(s_t)
            y_pred_list.append(y_pred)
            e = self.tanh(self.Wd(s_t) + self.Ud(H)) @ self.Vd
            b = self.softmax(e)  # [batch_size, seq_len, 1]
            v = b * H
            v = torch.sum(v, dim=1)  # [batch_size, hidden_dim]


        return torch.cat(y_pred_list, dim=-1)


class Decoder_LSF(nn.Module):
    def __init__(self, args):
        super(Decoder_LSF, self).__init__()
        self.args = args

        self.decoder = nn.GRUCell(args.d_model, hidden_size=args.d_ff)
        self.Wd = nn.Linear(args.d_ff, args.d_model, bias=False)
        self.Ud = nn.Linear(args.d_model, args.d_model, bias=False)

        self.Vd = nn.Parameter(torch.randn(args.d_model, 1))
        self.GRU_attn = nn.GRU(args.d_model, hidden_size=args.d_ff)

        self.GRU_decoder = nn.GRUCell(args.d_model+args.C_out, hidden_size=args.d_ff)
        self.projection_y = nn.Linear(args.d_ff, args.C_out, bias=False)
        
        self.projection = nn.Linear(args.d_ff + args.d_model, args.C_out)
        
    def forward(self, H, y_mark):

        B, T, _ = y_mark.shape # [batch_size, seq_len, C_out]
        zeros = torch.zeros(B, 1, self.args.C_out).to(H.device)
        y_mark = torch.concat((zeros, y_mark[:, 1:]), dim=1) # [batch_size, seq_len, C_out] 选取yt-1, 0 作为start_token
        d_t = torch.randn(B, self.args.d_ff).to(H.device)
        s_t = torch.randn(B, self.args.d_ff).to(H.device)
        d_t_list = []
        c_t_list = []
        # Layer 0
        for i in range(T):
            L_t = torch.tanh(self.Wd(torch.concat((d_t, s_t), dim=-1)) + self.Ud(H)) @ self.Vd
            beta_t = torch.softmax(L_t, dim=1)
            c_t = beta_t * H
            c_t = torch.sum(c_t, dim=1)
            y_t = self.projection_y(torch.concat((c_t, y_mark[:,i]), dim=-1))
            d_t = self.GRU_decoder(torch.concat((c_t, y_t), dim=-1))
            d_t_list.append(d_t)
            c_t_list.append(c_t)

        d_t = torch.concat(d_t_list, dim=1)
        c_t = torch.concat(c_t_list, dim=1)
        dec_out = torch.concat((d_t, c_t), dim=-1)
        dec_out = self.projection(dec_out)
        return dec_out # [batch_size, seq_len, C_out + C_in]

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        self.encoder = Encoder(args)
       

        if self.args.task == 'short_term_forecasting':
            self.decoder = Decoder_LSF(args)
            self.projection = nn.Linear(
                args.seq_len, args.pred_len )

        elif self.args.task == 'soft_sensor':
            self.decoder = Decoder_SS(args)
            

        else:
            raise ValueError("Invalid task type. Supported tasks are 'soft_sensor' and 'short_term_forecasting'. Please check your configuration.")

    def soft_sensor(self, x_enc, x_mark_enc=None, x_dec=None, x_dec_mark=None):

        enc_output, y_mark = self.encoder(x_enc, x_mark_enc)
        # [batch_size, seq_len, hidden_dim + x_dim]
        dec_output = self.decoder(enc_output, y_mark)
        dec_output = dec_output.squeeze(1)

        return dec_output  # [batch_size, pred_len, C_out]

    def short_term_forecasting(self, x_enc, x_mark_enc=None, x_dec=None, x_dec_mark=None):

        enc_output, y_mark = self.encoder(x_enc, x_mark_enc)
        
        dec_output = self.decoder(enc_output, y_mark)

        dec_output = self.projection(dec_output)  # [batch_size, pred_len, output_dim]

        return dec_output  # [batch_size, pred_len, C_out]

    def forward(self, x_enc,x_mark_enc,x_dec,x_mark_dec, batch_y, flag='train'):
        """
        x:  [batch_size, seq_len, C_in]
        """
        if self.args.task == 'soft_sensor':
            return self.soft_sensor(x_enc, x_dec, x_mark_enc, x_mark_dec)
        elif self.args.task == 'short_term_forecasting':
            dec_out = self.short_term_forecasting(x_enc, x_dec, x_mark_enc, x_mark_dec)
            return dec_out[:,-self.args.pred_len:]
        else:
            raise ValueError("Invalid task type. Supported tasks are 'soft_sensor' and 'short_term_forecasting'. Please check your configuration.")



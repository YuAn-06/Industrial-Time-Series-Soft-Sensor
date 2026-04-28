import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.Embedding import PositionalEmbedding
from layers.Normalization import StandardNorm
from layers.TimeFilter_layers import TimeFilter_Backbone


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)

    def forward(self, x):
        # x: [B, N, T]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, N*L, P]
        x = self.patch_proj(x)  # [B, N*L, D]
        if self.pos:
            x += self.pe(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.args = configs
        self.task = configs.task
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.C_in
        self.dim = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = configs.patch_len
        self.stride = self.patch_len
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)  # L

        # Filter
        self.alpha = 0.1 if configs.alpha is None else configs.alpha
        self.top_p = 0.5 if configs.top_p is None else configs.top_p

        # embed
        self.patch_embed = PatchEmbed(self.dim, self.patch_len, self.stride, configs.pos)

        # TimeFilter.sh Backbone
        self.backbone = TimeFilter_Backbone(self.dim, self.n_vars, self.d_ff,
                                            configs.n_heads, configs.e_layers, self.top_p, configs.dropout,
                                            self.seq_len * self.n_vars // self.patch_len)

        # head
        # self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)

        if self.task == 'short_term_forecasting':
            self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)
        elif self.task == 'soft_sensor':
            self.head = nn.Linear(self.dim * self.num_patches, configs.seq_len)
            self.projection = nn.Linear(configs.C_in, configs.C_out)

        # Without RevIN
        self.use_RevIN = False
        self.norm = StandardNorm(configs.enc_in, affine=self.use_RevIN)

    def _get_mask(self, device):
        dtype = torch.float32
        L = self.args.seq_len * self.args.C_in // self.args.patch_len
        N = self.args.seq_len // self.args.patch_len
        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(device)
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N) & (torch.arange(L) != k)).to(
                dtype).to(device)
            ST = torch.ones(L).to(dtype).to(device) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))
        masks = torch.stack(masks, dim=0)
        return masks

    def shor_term_forecasting(self, x_enc, masks, x_dec, x_mark_dec):
        # x: [B, T, C]
        B, T, C = x_enc.shape
        # Normalization
        x_enc = self.norm(x_enc, 'norm')
        # x: [B, C, T]
        x_enc = x_enc.permute(0, 2, 1).reshape(-1, C * T)  # [B, C*T]
        x_enc = self.patch_embed(x_enc)  # [B, N, D]  N = [C*T / P]

        x_enc, moe_loss = self.backbone(x_enc, self._get_mask(x_enc.device), self.alpha)

        # [B, C, T/P, D]
        x_enc = self.head(x_enc.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2))  # [B, C, T]
        x_enc = x_enc.permute(0, 2, 1)
        # De-Normalization
        x_enc = self.norm(x_enc, 'denorm')

        return x_enc
    
    def soft_sensor(self, x_enc, masks, x_dec, x_mark_dec):
        # x: [B, T, C]
        B, T, C = x_enc.shape
        # Normalization
        x_enc = self.norm(x_enc, 'norm')
        # x: [B, C, T]
        x_enc = x_enc.permute(0, 2, 1).reshape(-1, C * T)  # [B, C*T]
        x_enc = self.patch_embed(x_enc)  # [B, N, D]  N = [C*T / P]

        x_enc, moe_loss = self.backbone(x_enc, self._get_mask(x_enc.device), self.alpha)

        # [B, C, T/P, D]
        x_enc = self.head(x_enc.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2))  # [B, C, T]
        x_enc = x_enc.permute(0, 2, 1)
        # De-Normalization
        x_enc = self.norm(x_enc, 'denorm')

        x_enc = self.projection(x_enc[:, -1, :])

        return x_enc



    def forward(self, x_enc,x_mark_enc,x_dec,x_mark_dec, batch_y,flag = 'train'):
        if self.task == 'short_term_forecasting':
            dec_out = self.shor_term_forecasting(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        
        elif self.task == 'soft_sensor':
            return self.soft_sensor(x_enc,x_mark_enc, x_dec, x_mark_dec)
           
        else:
            raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor')
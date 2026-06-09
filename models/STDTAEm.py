import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Output_Layer import Permute

def _build_mlp(in_dim, hidden_dims, out_dim, dropout):
    layers = []
    last_dim = in_dim
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(last_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


class TemporalAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, latent_dim, num_layers, dropout, backbone, hidden_dims=None):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.backbone = backbone.lower()
        hidden_dims = hidden_dims or [hidden_dim]

        if self.backbone == "mlp":
            self.encoder = nn.Sequential(
                Permute(0, 2, 1),
                _build_mlp(seq_len, hidden_dims, latent_dim, dropout),
            )
        elif self.backbone == "lstm":
            rnn_dropout = dropout if num_layers > 1 else 0.0
            self.encoder = nn.LSTM(
                seq_len,
                hidden_dim,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True,
            )
            self.to_latent = nn.Linear(seq_len * hidden_dim, latent_dim)
        else:
            raise ValueError("tae_backbone must be one of ['mlp', 'lstm']")

        # decoder_hidden_dims = list(reversed(hidden_dims))
        # self.decoder = _build_mlp(latent_dim, decoder_hidden_dims, seq_len * input_dim, dropout)
        self.decoder = nn.LSTM(latent_dim, seq_len, num_layers)

    def encode(self, x):
        # x [B, S, D]
        if self.backbone == "mlp":
            # first invert the permutation
            return self.encoder(x) # [B, D, H]
        enc_out, _ = self.encoder(x)
        return enc_out # [B, S, H]
        # return self.to_latent(enc_out.reshape(enc_out.shape[0], -1))

    def decode(self, z):
        dec_out, _ = self.decoder(z) # [B D, C]
        # return self.decoder(z).reshape(z.shape[0], self.seq_len, self.input_dim)
        return dec_out.permute(0,2,1)

    def forward(self, x):
        z = self.encode(x) # [B, D ,H]
        rec = self.decode(z) # [B, D, C]
        return z, rec


class Model(nn.Module):
    """
    STD-TAEm: seasonal-trend decomposition + two temporal autoencoders.

    model_stage='pretrain' trains the TAE branches with reconstruction and
    triplet temporal-shape loss. model_stage='finetune' uses the TAE features
    for the supervised soft-sensor/forecasting objective already handled by exp.
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task = configs.task
        self.stage = getattr(configs, "model_stage", "finetune")
        self.C_in = configs.C_in
        self.C_out = configs.C_out
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.std_window = max(1, int(getattr(configs, "std_window", configs.moving_avg)))
        self.aug_noise_std = float(getattr(configs, "tae_noise_std", 0.05))
        self.aug_mask_ratio = float(getattr(configs, "tae_mask_ratio", 0.1))
        self.tae_backbone = getattr(configs, "tae_backbone", "mlp").lower()

        latent_dim = int(getattr(configs, "latent_dim", configs.hidden_dim))
        tae_hidden_dims = getattr(configs, "tae_hidden_dims", None)
        self.trend_tae = TemporalAutoencoder(
            self.C_in,
            self.seq_len,
            configs.hidden_dim,
            latent_dim,
            configs.e_layers,
            configs.dropout,
            self.tae_backbone,
            tae_hidden_dims,
        )
        self.seasonal_tae = TemporalAutoencoder(
            self.C_in,
            self.seq_len,
            configs.hidden_dim,
            latent_dim,
            configs.e_layers,
            configs.dropout,
            self.tae_backbone,
            tae_hidden_dims,
        )

        if self.task == "short_term_forecasting":
            out_dim = self.pred_len * self.C_out
            self.trend_regressor = nn.Linear(latent_dim, out_dim)
            self.seasonal_regressor = nn.Linear(latent_dim, out_dim)
        elif self.task == "soft_sensor":
            # out_dim = self.seq_len * self.C_out
            # self.trend_regressor = nn.Linear(latent_dim, out_dim)
            # self.seasonal_regressor = nn.Linear(latent_dim, out_dim)
            self.trend_regressor = nn.LSTM(self.C_in, 128, configs.e_layers)
            self.seasonal_regressor = nn.LSTM(self.C_in, 128, configs.e_layers)

            self.projection_trend = nn.Linear(128, self.C_out)
            self.projection_seasonal = nn.Linear(128, self.C_out)

        else:
            raise ValueError(
                f"Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor"
            )

        self.trend_to_t_dim = nn.Linear(latent_dim, configs.seq_len)
        self.seasonal_to_t_dim = nn.Linear(latent_dim, configs.seq_len)
        self.projection = nn.Linear(self.C_in, self.C_out)
        pretrained_ckpt = getattr(configs, "pretrained_ckpt", "")
        if pretrained_ckpt:
            self.load_pretrained(pretrained_ckpt)
            if getattr(configs, "freeze_tae", False):
                self.freeze_tae()

    def load_pretrained(self, ckpt_path, strict=False):
        state = torch.load(ckpt_path, map_location="cpu")
        state = state.get("model", state)
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        return self.load_state_dict(state, strict=strict)

    def freeze_tae(self):
        for module in (self.trend_tae, self.seasonal_tae):
            for param in module.parameters():
                param.requires_grad = False

    def _moving_average(self, x):
        if self.std_window <= 1:
            return x

        left = self.std_window // 2
        right = self.std_window - 1 - left
        x_pad = F.pad(x.transpose(1, 2), (left, right), mode="replicate")
        trend = F.avg_pool1d(x_pad, kernel_size=self.std_window, stride=1)
        return trend.transpose(1, 2)

    def _minmax_norm(self, x):
        x_min = x.amin(dim=1, keepdim=True)
        x_max = x.amax(dim=1, keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-6)

    def decompose(self, x):
        trend = self._moving_average(x)
        seasonal = x - trend
        return self._minmax_norm(trend), self._minmax_norm(seasonal)

    def _positive_sample(self, x):
        x_pos = x + torch.randn_like(x) * self.aug_noise_std
        if self.aug_mask_ratio <= 0:
            return x_pos
        keep = torch.rand_like(x_pos) > self.aug_mask_ratio
        return x_pos * keep

    def _negative_sample(self, x):
        return torch.flip(x, dims=[1])

    def _tae_pretrain_branch(self, tae, x):
        x_pos = self._positive_sample(x)
        x_neg = self._negative_sample(x)
        z, rec = tae(x)
        z_pos = tae.encode(x_pos)
        z_neg = tae.encode(x_neg)
        return {
            "z": z,
            "z_pos": z_pos,
            "z_neg": z_neg,
            "rec": rec,
            "target": x,
        }

    def pretrain_forward(self, x_enc):
        trend, seasonal = self.decompose(x_enc)
        return {
            "trend": self._tae_pretrain_branch(self.trend_tae, trend),
            "seasonal": self._tae_pretrain_branch(self.seasonal_tae, seasonal),
        }

    def finetune_forward(self, x_enc):
        trend, seasonal = self.decompose(x_enc)

        if self.task == "short_term_forecasting":
            z_t = self.trend_tae.encode(trend)
            z_s = self.seasonal_tae.encode(seasonal)

            z_t = self.trend_to_t_dim(z_t)
            z_s = self.seasonal_to_t_dim(z_s)
            y_t, _ = self.trend_regressor(z_t)
            y_s, _ = self.seasonal_regressor(z_s)
            y = y_t + y_s
            y = y.permute(0,2,1) # [B, S, D]
            return y

        z_t = self.trend_tae.encode(trend)
        z_s = self.seasonal_tae.encode(seasonal)
        z_t = self.trend_to_t_dim(z_t) # [B, D, S]
        z_s = self.seasonal_to_t_dim(z_s) # [B, D, S]
        z_t = z_t.permute(0,2,1) # [B, S, D]
        z_s = z_s.permute(0,2,1) # [B, S, D]

        y_t, _ = self.trend_regressor(z_t)
        y_t = self.projection_trend(y_t)
        y_s, _ = self.seasonal_regressor(z_s)
        y_s = self.projection_seasonal(y_s)
        y = y_t + y_s # [B, S, D]
        return y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag="train"):
        if self.stage == "pretrain":
            return self.pretrain_forward(x_enc)
        return self.finetune_forward(x_enc)

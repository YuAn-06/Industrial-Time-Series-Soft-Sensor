"""PETC-TNet architecture from He et al., IEEE Sensors Journal, 2025.

The paper specifies the module sequence and main hyperparameters but does not
publish code or fully specify the hidden TCN widths. Those details are exposed
in ``model_config.py``. A Transformer decoder consumes known historical target
values followed by a zero placeholder for the one-step-ahead prediction.
"""

import torch
from torch import nn


class CausalConv1d(nn.Module):
    """Left-padded dilated convolution that never observes future samples."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.left_padding, 0)))


class MultiHeadChannelAttention(nn.Module):
    """Equations (5)-(8): avg+max pooling and averaged MLP heads."""

    def __init__(self, channels, reduction_ratio, num_heads):
        super().__init__()
        hidden = max(1, channels // reduction_ratio)
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(channels, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, channels),
                nn.LayerNorm(channels),
            )
            for _ in range(num_heads)
        )

    def forward(self, x):
        pooled = x.mean(dim=-1) + x.amax(dim=-1)
        logits = torch.stack([head(pooled) for head in self.heads], dim=0).mean(dim=0)
        weights = torch.sigmoid(logits).unsqueeze(-1)
        return x * weights


class TCCANLayer(nn.Module):
    """Dilated causal convolution followed immediately by channel attention."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, reduction_ratio, attention_heads, dropout):
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.attention = MultiHeadChannelAttention(out_channels, reduction_ratio, attention_heads)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        features = self.activation(self.conv(x))
        features = self.dropout(self.attention(features))
        return self.activation(features + self.residual(x))


class Model(nn.Module):
    """Patch decomposition enhanced TCCAN-Transformer soft sensor."""

    def __init__(self, configs):
        super().__init__()
        self.patch_len = configs.patch_len
        self.num_patches = configs.seq_len // configs.patch_len
        self.input_channels = configs.C_in
        self.output_channels = configs.C_out
        self.pred_len = configs.pred_len

        tccan_layers = []
        in_channels = configs.C_in
        for level, out_channels in enumerate(configs.tcn_channels):
            tccan_layers.append(
                TCCANLayer(
                    in_channels,
                    out_channels,
                    configs.kernel_size,
                    dilation=2**level,
                    reduction_ratio=configs.reduction_ratio,
                    attention_heads=configs.channel_attention_heads,
                    dropout=configs.dropout,
                )
            )
            in_channels = out_channels
        self.tccan = nn.Sequential(*tccan_layers)
        self.patch_projection = nn.Linear(in_channels, configs.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, configs.d_model))
        self.decoder_embedding = nn.Linear(configs.C_out, configs.d_model)
        self.decoder_position_embedding = nn.Parameter(
            torch.zeros(1, configs.label_len + configs.pred_len, configs.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, configs.e_layers, norm=nn.LayerNorm(configs.d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            configs.d_layers,
            norm=nn.LayerNorm(configs.d_model),
        )
        self.output_layer = nn.Linear(configs.d_model, configs.C_out)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.decoder_position_embedding, std=0.02)

    def short_term_forecasting(self, x_enc, x_dec):
        batch_size, seq_len, channels = x_enc.shape
        if seq_len % self.patch_len != 0:
            raise ValueError(f"Input length {seq_len} is not divisible by patch_len={self.patch_len}.")
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {channels}.")

        # [B, L, D] -> [B*P, D, S], so every patch shares one TCCAN.
        patches = x_enc.reshape(batch_size, seq_len // self.patch_len, self.patch_len, channels)
        patches = patches.reshape(-1, self.patch_len, channels).transpose(1, 2)
        local_features = self.tccan(patches)
        patch_tokens = local_features.mean(dim=-1)
        patch_tokens = self.patch_projection(patch_tokens).reshape(batch_size, -1, self.position_embedding.size(-1))

        memory = self.encoder(patch_tokens + self.position_embedding[:, : patch_tokens.size(1)])

        decoder_targets = x_dec[..., -self.output_channels:]
        decoder_tokens = self.decoder_embedding(decoder_targets)
        decoder_tokens = decoder_tokens + self.decoder_position_embedding[:, :decoder_tokens.size(1)]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            decoder_tokens.size(1), device=decoder_tokens.device
        )
        decoded = self.decoder(decoder_tokens, memory, tgt_mask=causal_mask)
        return self.output_layer(decoded[:, -self.pred_len:])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag="train"):
        return self.short_term_forecasting(x_enc, x_dec)

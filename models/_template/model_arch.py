"""Template model architecture.

Replace the example projection with the paper's architecture. Every model
architecture file must export its benchmark entry class with the name ``Model``.
The forward signature should accept the batch fields produced by the selected
dataset type and follow the output convention expected by the experiment.
"""

import torch
from torch import nn


class Model(nn.Module):
    """Minimal example implementing the benchmark's conventional interface."""

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.projection = nn.Sequential(
            nn.Linear(configs.C_in, configs.hidden_dim),
            nn.ReLU(),
            nn.Linear(configs.hidden_dim, configs.C_out),
        )

    def forward(self, x_enc: torch.Tensor, **batch):
        """Return ``[batch, pred_len, C_out]`` for the default adapter."""
        del batch
        return self.projection(x_enc[:, -self.pred_len :, :])

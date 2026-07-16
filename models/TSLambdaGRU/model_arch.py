"""Two-stream lambda-GRU soft sensor (Xie et al., IEEE TIE 2020).

The temporal stream consumes past quality measurements and uses lambda-1
cells.  The dynamic-causal stream consumes process variables and uses
lambda-2 cells.  Each recurrent layer is followed by batch normalization.
"""

import torch
from torch import nn


class LambdaGRULayer(nn.Module):
    """GRU layer with the paper's ReLU candidate and regulated memory."""

    def __init__(self, input_size, hidden_size, regulator, variant):
        super().__init__()
        if variant not in (1, 2):
            raise ValueError("variant must be 1 (short-term) or 2 (long-term)")
        self.hidden_size = hidden_size
        self.regulator = regulator
        self.variant = variant
        self.gates = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        self.candidate_x = nn.Linear(input_size, hidden_size)
        self.candidate_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        batch_size, steps, _ = x.shape
        h = x.new_zeros(batch_size, self.hidden_size)
        outputs = []
        for t in range(steps):
            x_t = x[:, t]
            z_t, r_t = torch.sigmoid(self.gates(torch.cat((x_t, h), dim=-1))).chunk(2, dim=-1)
            h_tilde = torch.relu(self.candidate_x(x_t) + self.candidate_h(h * r_t))
            if self.variant == 1:
                h = self.regulator * z_t * h + (1.0 - z_t) * h_tilde
            else:
                h = z_t * h + self.regulator * (1.0 - z_t) * h_tilde
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class LambdaGRUStream(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, regulator, variant):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_index in range(num_layers):
            layer_input = input_size if layer_index == 0 else hidden_size
            self.layers.append(LambdaGRULayer(layer_input, hidden_size, regulator, variant))
            self.norms.append(nn.BatchNorm1d(hidden_size))
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        for recurrent, norm in zip(self.layers, self.norms):
            x = recurrent(x)
            x = norm(x.transpose(1, 2)).transpose(1, 2)
        return torch.relu(self.projection(x[:, -1]))


class Model(nn.Module):
    """Paper-faithful TS-lambda-GRU architecture for soft-sensor regression."""

    def __init__(self, configs):
        super().__init__()
        if configs.task != "soft_sensor":
            raise ValueError("TSLambdaGRU supports only the soft_sensor task")
        if configs.e_layers < 1:
            raise ValueError("e_layers must be at least 1")
        if not 0.0 < configs.lambda1 < 1.0 or not 0.0 < configs.lambda2 < 1.0:
            raise ValueError("lambda1 and lambda2 must lie strictly between 0 and 1")
        if configs.seq_len < 2:
            raise ValueError("TSLambdaGRU needs seq_len >= 2 to avoid target leakage")

        process_features = configs.C_in
        self.C_out = configs.C_out
        self.temporal_stream = LambdaGRUStream(
            configs.C_out, configs.hidden_dim, configs.e_layers,
            configs.lambda1, variant=1,
        )
        self.dynamic_stream = LambdaGRUStream(
            process_features, configs.hidden_dim, configs.e_layers,
            configs.lambda2, variant=2,
        )
        self.merge = nn.Linear(2 * configs.hidden_dim, configs.hidden_dim)
        self.dropout = nn.Dropout(configs.dropout)
        self.regression = nn.Linear(configs.hidden_dim, configs.C_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag="train"):
        expected_features = self.dynamic_stream.layers[0].gates.in_features - self.dynamic_stream.layers[0].hidden_size
        if x_enc.size(-1) != expected_features + self.C_out:
            raise ValueError(
                "TSLambdaGRU expects x_enc=[process variables, historical target]; "
                f"received {x_enc.size(-1)} channels"
            )
        process_x = x_enc[:, :, :expected_features]
        # The last x_enc row aligns with the prediction target.  Only earlier
        # quality samples are observable online and are admitted to this stream.
        quality_history = x_enc[:, :-1, -self.C_out:]
        temporal = self.temporal_stream(quality_history)
        dynamic = self.dynamic_stream(process_x)
        fused = torch.relu(self.merge(torch.cat((temporal, dynamic), dim=-1)))
        return self.regression(self.dropout(fused))

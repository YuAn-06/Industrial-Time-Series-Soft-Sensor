"""DAMPNN architecture for industrial soft sensing.

The implementation follows equations (5)-(21) of Yan et al., IEEE TII 2025:
adaptive macro/micro graph learning, gated dynamic message passing, and
timewise followed by variablewise self-attention.
"""

import math

import torch
from torch import nn


class AdaptiveGraphLearning(nn.Module):
    """Learn and fuse the sample-wise macro and micro adjacency matrices."""

    def __init__(self, seq_len, projection_dim, beta, momentum_lambda, threshold):
        super().__init__()
        self.theta1 = nn.Linear(seq_len, projection_dim, bias=False)
        self.theta2 = nn.Linear(seq_len, projection_dim, bias=False)
        self.beta = beta
        self.momentum_lambda = momentum_lambda
        self.threshold = threshold

    @staticmethod
    def _macro_graph(nodes):
        distances = torch.cdist(nodes, nodes, p=2)
        rho = distances.var(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
        return torch.exp(-distances.square() / rho.square())

    def _micro_graph(self, nodes):
        c1 = torch.tanh(self.beta * self.theta1(nodes))
        c2 = torch.tanh(self.beta * self.theta2(nodes))
        skew_affinity = c1 @ c2.transpose(-1, -2) - c2 @ c1.transpose(-1, -2)
        return torch.relu(torch.tanh(self.beta * skew_affinity))

    def forward(self, nodes):
        macro = self._macro_graph(nodes)
        micro = self._micro_graph(nodes)
        discrepancy = torch.linalg.matrix_norm(macro - micro, ord="fro", dim=(-2, -1))
        eta = torch.exp(-(self.momentum_lambda + discrepancy)).view(-1, 1, 1)
        total = eta * macro + (1.0 - eta) * micro
        return total * (total >= self.threshold).to(total.dtype)


class DynamicMessagePassing(nn.Module):
    """Paper equations (12)-(18), including the switch gate."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.message_to_node = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.node_to_message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.p_node = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.p_message = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.s_node = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.s_message = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, hidden, adjacency):
        message = adjacency @ hidden
        modulated_node = 2.0 * torch.sigmoid(self.message_to_node(message)) * hidden
        modulated_message = 2.0 * torch.sigmoid(self.node_to_message(hidden)) * message
        proposal = torch.tanh(self.p_node(modulated_node) + self.p_message(modulated_message))
        switch = torch.sigmoid(self.s_node(modulated_node) + self.s_message(modulated_message))
        return switch * proposal + (1.0 - switch) * modulated_message


class AxisSelfAttention(nn.Module):
    """Single-head scaled dot-product attention with an explicit output projection."""

    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, attention_dim, bias=False)
        self.key = nn.Linear(input_dim, attention_dim, bias=False)
        self.value = nn.Linear(input_dim, attention_dim, bias=False)
        self.output = nn.Linear(attention_dim, input_dim, bias=False)
        self.scale = math.sqrt(attention_dim)

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        weights = torch.softmax((q @ k.transpose(-1, -2)) / self.scale, dim=-1)
        return self.output(weights @ v)


class DualSelfAttention(nn.Module):
    def __init__(self, num_variables, hidden_dim, attention_dim):
        super().__init__()
        # [B, variables, hidden] -> transpose -> attend over hidden/time features.
        self.timewise = AxisSelfAttention(num_variables, attention_dim)
        self.variablewise = AxisSelfAttention(hidden_dim, attention_dim)

    def forward(self, hidden):
        hidden = self.timewise(hidden.transpose(1, 2)).transpose(1, 2)
        return self.variablewise(hidden)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task = configs.task
        self.num_variables = configs.C_in
        self.output_dim = configs.C_out

        self.graph_learning = AdaptiveGraphLearning(
            seq_len=configs.seq_len,
            projection_dim=configs.graph_projection_dim,
            beta=configs.graph_beta,
            momentum_lambda=configs.graph_lambda,
            threshold=configs.graph_threshold,
        )
        self.node_encoder = nn.Linear(configs.seq_len, configs.d_model)
        self.message_passing = nn.ModuleList(
            DynamicMessagePassing(configs.d_model)
            for _ in range(configs.message_passing_layers)
        )
        self.readout = DualSelfAttention(
            num_variables=configs.C_in,
            hidden_dim=configs.d_model,
            attention_dim=configs.attention_dim,
        )
        self.regression_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.C_in * configs.d_model, configs.C_out),
        )

    def soft_sensor(self, x_enc):
        nodes = x_enc.transpose(1, 2).contiguous()
        adjacency = self.graph_learning(nodes)
        hidden = self.node_encoder(nodes)
        for layer in self.message_passing:
            hidden = layer(hidden, adjacency)
        return self.regression_head(self.readout(hidden))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag="train"):
        if self.task != "soft_sensor":
            raise ValueError("DAMPNN supports only the soft_sensor task.")
        return self.soft_sensor(x_enc)

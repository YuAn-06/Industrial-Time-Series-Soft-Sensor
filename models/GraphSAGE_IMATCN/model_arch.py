import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import gaussian_kde
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

from data.data_loader import _get_borders, _get_feature_columns, preprocess_data_dict
from layers.IMATCN import IMATCN

"""
Soft sensor model for nonlinear dynamic industrial process based on GraphSAGE-IMATCN

Thanks to authors: Benben Tuo, Xiaoqiang Zhao, et al.
"""

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, h_feats)
        self.conv3 = SAGEConv(h_feats, out_feats)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.conv3(x, edge_index)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task = configs.task
        self.input_size = configs.C_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.C_out

        self.sage_dim = configs.d_model
        self.tcn_channels = [int(ch) for ch in configs.num_channels]
        self.graph_build_method = getattr(configs, "graph_build_method", "mi")
        self.graph_threshold = getattr(configs, "graph_threshold", 0.4)
        self.graph_sample_size = getattr(configs, "graph_sample_size", 64)
        self.register_buffer("base_edge_index", self._build_base_edge_index(configs), persistent=False)

        self.sage = GraphSAGE(in_feats=self.seq_len, h_feats=self.sage_dim, out_feats=self.sage_dim)
        self.tcn = IMATCN(
            input_features_num=self.input_size,
            input_len=self.sage_dim,
            output_len=self.sage_dim,
            tcn_OutputChannelList=self.tcn_channels,
            tcn_KernelSize=configs.kernel_size,
            tcn_Dropout=configs.dropout,
            n_heads=configs.n_heads,
        )

        self.soft_sensor_head = nn.Sequential(
            nn.Linear(self.sage_dim, max(self.sage_dim // 2, self.c_out)),
            nn.ReLU(),
            nn.Linear(max(self.sage_dim // 2, self.c_out), self.c_out),
        )
        self.forecast_head = nn.Linear(self.sage_dim, self.pred_len * self.c_out)

    @staticmethod
    def _build_full_edge_index(num_nodes):
        mask = ~torch.eye(num_nodes, dtype=torch.bool)
        return mask.nonzero(as_tuple=False).t().contiguous()

    @staticmethod
    def _mutual_information(x, y):
        length = len(x)
        if length < 2:
            return 0.0
        bandwidth = (4 / (2 + 2)) ** (1 / (4 + 2)) * (length ** (-1 / (2 + 4)))
        px = gaussian_kde(x, bw_method=bandwidth)
        py = gaussian_kde(y, bw_method=bandwidth)
        pxy = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)
        px_val = np.clip(px(x), 1e-12, None)
        py_val = np.clip(py(y), 1e-12, None)
        pxy_val = np.clip(pxy(np.vstack([x, y])), 1e-12, None)
        return float(np.mean(np.log2(pxy_val / px_val / py_val)))

    @staticmethod
    def _abs_corr(x, y):
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        return float(abs(np.corrcoef(x, y)[0, 1]))

    def _build_edge_index_from_array(self, data):
        edges = [[], []]
        sample_size = min(self.graph_sample_size, data.shape[0])
        data = data[:sample_size]

        for i in range(self.input_size):
            for j in range(i + 1, self.input_size):
                x = data[:, i]
                y = data[:, j]
                try:
                    score = (
                        self._mutual_information(x, y)
                        if self.graph_build_method == "mi"
                        else self._abs_corr(x, y)
                    )
                except Exception:
                    score = self._abs_corr(x, y)
                if score >= self.graph_threshold:
                    edges[0].append(i)
                    edges[1].append(j)

        if not edges[0]:
            return self._build_full_edge_index(self.input_size)

        edge_index = torch.tensor(edges, dtype=torch.long)
        return to_undirected(edge_index, num_nodes=self.input_size).contiguous()

    def _build_base_edge_index(self, configs):
        data_path = getattr(configs, "data_path", "")
        if not data_path:
            return self._build_full_edge_index(self.input_size)

        try:
            df_raw = pd.read_csv(data_path)
            _, border2s, _, _ = _get_borders(len(df_raw), configs.seq_len, 0)
            columns_with_x = _get_feature_columns(
                df_raw,
                configs.data_name,
                configs.target,
                include_target_when_no_x=False,
            )
            if configs.data_name in ['DC', 'SRU'] and configs.data_aug:
                df_raw, columns_with_x = preprocess_data_dict[configs.data_name](df_raw, configs.target)
            data = df_raw[columns_with_x].values[:border2s[0], :self.input_size]
            return self._build_edge_index_from_array(data)
        except Exception:
            return self._build_full_edge_index(self.input_size)

    def _batch_edge_index(self, batch_size, device):
        edge_index = self.base_edge_index.to(device)
        offsets = torch.arange(batch_size, device=device) * self.input_size
        edge_index = edge_index.unsqueeze(1) + offsets.view(1, batch_size, 1)
        return edge_index.permute(0, 2, 1).reshape(2, -1).contiguous()

    def _encode(self, x_enc):
        batch_size = x_enc.size(0)
        x = x_enc.transpose(1, 2).contiguous().view(batch_size * self.input_size, self.seq_len)
        edge_index = self._batch_edge_index(batch_size, x_enc.device)
        x = self.sage(x, edge_index)
        x = x.view(batch_size, self.input_size, self.sage_dim).transpose(1, 2)
        return self.tcn(x)

    def soft_sensor(self, x_enc):
        x = self._encode(x_enc)
        return self.soft_sensor_head(x)

    def short_term_forecasting(self, x_enc):
        x = self._encode(x_enc)
        x = self.forecast_head(x)
        return x.view(-1, self.pred_len, self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, flag='train'):
        if self.task == 'short_term_forecasting':
            return self.short_term_forecasting(x_enc)
        if self.task == 'soft_sensor':
            return self.soft_sensor(x_enc)
        raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting and soft_sensor')

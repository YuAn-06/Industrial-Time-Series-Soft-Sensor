"""Model-specific configuration for MSGNet."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class MSGNetConfig(BaseExpConfig):
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    n_heads: int = field(default=8, metadata={'help': 'Number of attention heads', 'prefix': 'nh', 'order': 102})
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers', 'prefix': 'el', 'order': 103})
    d_ff: int = field(default=1024, metadata={'help': 'Feed forward dimension', 'prefix': 'dff', 'order': 101})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    individual: bool = field(default=False, metadata={'help': 'If use individual linear layer for each forecast'})
    top_k: int = field(default=5, metadata={'help': 'Top k for TimesNet'})
    conv_channel: int = field(default=32, metadata={'help': 'Convolution channel for GCN', 'prefix': 'cc', 'order': 106})
    skip_channel: int = field(default=32, metadata={'help': 'Skip channel for GCN', 'prefix': 'sc', 'order': 107})
    gcn_depth: int = field(default=2, metadata={'help': 'GCN depth', 'prefix': 'gcn', 'order': 109})
    node_dim: int = field(default=10, metadata={'help': 'Node dimension for GCN', 'prefix': 'nd', 'order': 105})
    propalpha: float = field(default=0.1, metadata={'help': 'Propagation alpha for GCN', 'prefix': 'pa', 'order': 108})

MODEL_CONFIG = MSGNetConfig

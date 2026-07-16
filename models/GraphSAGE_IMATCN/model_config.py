"""Model-specific configuration for GraphSAGE_IMATCN."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class GraphSAGE_IMATCNConfig(BaseExpConfig):
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    n_heads: int = field(default=8, metadata={'help': 'Number of attention heads', 'prefix': 'nh', 'order': 103})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    kernel_size: int = field(default=4, metadata={'help': 'Kernel size for EnvFormer & TCN', 'prefix': 'ks', 'order': 102})
    num_channels: list[int] = field(default_factory=lambda: [16, 32, 64], metadata={'help': 'Number of channels for TCN', 'prefix': 'nc', 'order': 101})
    graph_build_method: str = field(default='mi', metadata={'help': 'Graph construction method for GraphSAGE_IMATCN', 'prefix': 'gm', 'order': 104})
    graph_threshold: float = field(default=0.4, metadata={'help': 'Edge threshold for GraphSAGE_IMATCN graph construction', 'prefix': 'gt', 'order': 105})
    graph_sample_size: int = field(default=64, metadata={'help': 'Training samples used to estimate GraphSAGE_IMATCN graph edges'})

MODEL_CONFIG = GraphSAGE_IMATCNConfig

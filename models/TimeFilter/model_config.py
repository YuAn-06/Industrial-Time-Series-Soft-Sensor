"""Model-specific configuration for TimeFilter."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class TimeFilterConfig(BaseExpConfig):
    patch_len: int = field(default=8, metadata={'help': 'Patch length'})
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    n_heads: int = field(default=8, metadata={'help': 'Number of attention heads', 'prefix': 'nh', 'order': 102})
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers', 'prefix': 'el', 'order': 103})
    d_ff: int = field(default=1024, metadata={'help': 'Feed forward dimension', 'prefix': 'dff', 'order': 101})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    alpha: float = field(default=0.1, metadata={'help': 'KNN for Graph Construction'})
    top_p: float = field(default=0.5, metadata={'help': 'Dynamic Routing in MoE'})
    pos: list[int] = field(default_factory=lambda: [1], metadata={'help': 'Positional Embedding. Set pos to 0 or 1'})

MODEL_CONFIG = TimeFilterConfig

"""Model-specific configuration for Fredformer."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class FredformerConfig(BaseExpConfig):
    patch_len: int = field(default=8, metadata={'help': 'Patch length'})
    stride: int = field(default=1, metadata={'help': 'Stride'})
    factor: int = field(default=1, metadata={'help': 'Factor of Attention'})
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    pd_model: int = field(default=32, metadata={'help': 'Patch embedding dimension after the encoder.'})
    n_heads: int = field(default=8, metadata={'help': 'Number of attention heads', 'prefix': 'nh', 'order': 102})
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers', 'prefix': 'el', 'order': 103})
    d_ff: int = field(default=1024, metadata={'help': 'Feed forward dimension', 'prefix': 'dff', 'order': 101})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    activation: str = field(default='gelu', metadata={'help': 'Activation function'})
    individual: bool = field(default=False, metadata={'help': 'If use individual linear layer for each forecast'})

MODEL_CONFIG = FredformerConfig

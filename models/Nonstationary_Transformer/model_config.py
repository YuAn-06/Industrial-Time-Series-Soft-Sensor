"""Model-specific configuration for Nonstationary_Transformer."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class Nonstationary_TransformerConfig(BaseExpConfig):
    factor: int = field(default=1, metadata={'help': 'Factor of Attention'})
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    n_heads: int = field(default=8, metadata={'help': 'Number of attention heads', 'prefix': 'nh', 'order': 102})
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers', 'prefix': 'el', 'order': 103})
    d_layers: int = field(default=1, metadata={'help': 'Decoder layers', 'prefix': 'dl', 'order': 104})
    d_ff: int = field(default=1024, metadata={'help': 'Feed forward dimension', 'prefix': 'dff', 'order': 101})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    activation: str = field(default='gelu', metadata={'help': 'Activation function'})
    p_hidden_dims: list[int] = field(default_factory=lambda: [128, 128], metadata={'help': 'Hidden dimensions for projection network'})
    p_hidden_layers: int = field(default=2, metadata={'help': 'Number of hidden layers for projection network'})

MODEL_CONFIG = Nonstationary_TransformerConfig

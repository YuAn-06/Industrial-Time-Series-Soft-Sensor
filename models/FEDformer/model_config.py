"""Model-specific configuration for FEDformer."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class FEDformerConfig(BaseExpConfig):
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    n_heads: int = field(default=8, metadata={'help': 'Number of attention heads', 'prefix': 'nh', 'order': 102})
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers', 'prefix': 'el', 'order': 103})
    d_layers: int = field(default=1, metadata={'help': 'Decoder layers', 'prefix': 'dl', 'order': 104})
    d_ff: int = field(default=1024, metadata={'help': 'Feed forward dimension', 'prefix': 'dff', 'order': 101})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    activation: str = field(default='gelu', metadata={'help': 'Activation function'})
    moving_avg: int = field(default=25, metadata={'help': 'moving average for Autoformer'})
    version: str = field(default='fourier', metadata={'help': 'Version of FEDformer: [Fourier, Wavelets]'})
    mode_select: str = field(default='random', metadata={'help': 'Mode selection method for FEDformer: [random, low]'})
    modes: int = field(default=32, metadata={'help': 'Number of modes to be selected for FEDformer'})

MODEL_CONFIG = FEDformerConfig

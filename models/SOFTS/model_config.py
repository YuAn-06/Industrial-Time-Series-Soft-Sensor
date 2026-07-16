"""Model-specific configuration for SOFTS."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class SOFTSConfig(BaseExpConfig):
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers', 'prefix': 'el', 'order': 103})
    d_ff: int = field(default=1024, metadata={'help': 'Feed forward dimension', 'prefix': 'dff', 'order': 101})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    activation: str = field(default='gelu', metadata={'help': 'Activation function'})
    use_norm: bool = field(default=False, metadata={'help': 'If use normalization for TimesMixer'})
    d_core: int = field(default=10, metadata={'help': 'Core dimension for SOFTS'})

MODEL_CONFIG = SOFTSConfig

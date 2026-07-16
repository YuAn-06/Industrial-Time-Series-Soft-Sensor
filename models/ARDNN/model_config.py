"""Model-specific configuration for ARDNN."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class ARDNNConfig(BaseExpConfig):
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers', 'prefix': 'el', 'order': 101})

MODEL_CONFIG = ARDNNConfig

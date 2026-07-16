"""Model-specific configuration for DLSTM."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class DLSTMConfig(BaseExpConfig):
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers', 'prefix': 'el', 'order': 103})
    hidden_dim: int = field(default=10, metadata={'help': 'hidden dimension for each distributed GRU unit'})

MODEL_CONFIG = DLSTMConfig

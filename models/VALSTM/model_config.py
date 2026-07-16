"""Model-specific configuration for VALSTM."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class VALSTMConfig(BaseExpConfig):
    d_model: int = field(default=512, metadata={'help': 'Model dimension'})
    hidden_dim: int = field(default=10, metadata={'help': 'hidden dimension for each distributed GRU unit', 'prefix': 'hd', 'order': 100})

MODEL_CONFIG = VALSTMConfig

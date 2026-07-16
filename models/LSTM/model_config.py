"""Model-specific configuration for LSTM."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class LSTMConfig(BaseExpConfig):
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers', 'prefix': 'el', 'order': 103})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    hidden_dim: int = field(default=10, metadata={'help': 'hidden dimension for each distributed GRU unit'})

MODEL_CONFIG = LSTMConfig

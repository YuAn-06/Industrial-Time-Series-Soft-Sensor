"""Model-specific configuration for TSLambdaGRU."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class TSLambdaGRUConfig(BaseExpConfig):
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers', 'prefix': 'el', 'order': 101})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate', 'prefix': 'drop', 'order': 104})
    hidden_dim: int = field(default=10, metadata={'help': 'hidden dimension for each distributed GRU unit', 'prefix': 'hd', 'order': 100})
    lambda1: float = field(default=0.9, metadata={'help': 'TS-lambda-GRU short-term memory regulator', 'prefix': 'l1', 'order': 102})
    lambda2: float = field(default=0.9, metadata={'help': 'TS-lambda-GRU long-term memory regulator', 'prefix': 'l2', 'order': 103})

MODEL_CONFIG = TSLambdaGRUConfig

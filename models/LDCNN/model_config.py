"""Model-specific configuration for LDCNN."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class LDCNNConfig(BaseExpConfig):
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers', 'prefix': 'el', 'order': 103})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})

MODEL_CONFIG = LDCNNConfig

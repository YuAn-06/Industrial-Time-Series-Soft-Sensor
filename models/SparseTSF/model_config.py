"""Model-specific configuration for SparseTSF."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class SparseTSFConfig(BaseExpConfig):
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    model_type: str = field(default='linear', metadata={'help': '[linear, mlp]', 'prefix': 'mt', 'order': 102})
    period_len: int = field(default=10, metadata={'help': 'Period length for SparseTSF', 'prefix': 'prl', 'order': 101})

MODEL_CONFIG = SparseTSFConfig

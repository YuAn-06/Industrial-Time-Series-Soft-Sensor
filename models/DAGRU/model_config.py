"""Model-specific configuration for DAGRU."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class DAGRUConfig(BaseExpConfig):
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    d_ff: int = field(default=1024, metadata={'help': 'Feed forward dimension', 'prefix': 'dff', 'order': 101})

MODEL_CONFIG = DAGRUConfig

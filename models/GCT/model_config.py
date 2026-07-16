"""Model-specific configuration for GCT."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class GCTConfig(BaseExpConfig):
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    n_heads: int = field(default=8, metadata={'help': 'Number of attention heads'})
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers'})
    d_ff: int = field(default=1024, metadata={'help': 'Feed forward dimension', 'prefix': 'dff', 'order': 101})

MODEL_CONFIG = GCTConfig

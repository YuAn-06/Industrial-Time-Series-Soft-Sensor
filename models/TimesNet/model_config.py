"""Model-specific configuration for TimesNet."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class TimesNetConfig(BaseExpConfig):
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers', 'prefix': 'el', 'order': 103})
    d_ff: int = field(default=1024, metadata={'help': 'Feed forward dimension', 'prefix': 'dff', 'order': 101})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    top_k: int = field(default=5, metadata={'help': 'Top k for TimesNet'})
    num_kernels: int = field(default=5, metadata={'help': 'Number of kernels for TimesMixer'})

MODEL_CONFIG = TimesNetConfig

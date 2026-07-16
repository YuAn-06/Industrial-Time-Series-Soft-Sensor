"""Model-specific configuration for DLinear."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class DLinearConfig(BaseExpConfig):
    moving_avg: int = field(default=25, metadata={'help': 'moving average for Autoformer', 'prefix': 'ma', 'order': 100})
    individual: bool = field(default=False, metadata={'help': 'If use individual linear layer for each forecast', 'prefix': 'ind', 'order': 101})

MODEL_CONFIG = DLinearConfig

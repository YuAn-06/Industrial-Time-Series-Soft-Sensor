"""Model-specific configuration for TCN."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class TCNConfig(BaseExpConfig):
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    kernel_size: int = field(default=4, metadata={'help': 'Kernel size for EnvFormer & TCN', 'prefix': 'ks', 'order': 100})
    num_channels: list[int] = field(default_factory=lambda: [16, 32, 64], metadata={'help': 'Number of channels for TCN'})

MODEL_CONFIG = TCNConfig

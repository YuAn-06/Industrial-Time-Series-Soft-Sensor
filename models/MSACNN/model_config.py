"""Model-specific configuration for MSACNN."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class MSACNNConfig(BaseExpConfig):
    stride: int = field(default=1, metadata={'help': 'Stride'})
    reduction_ratio: int = field(default=16, metadata={'help': 'Reduction ratio for MSACNN'})
    out_per_channel: list[int] = field(default_factory=lambda: [5, 8, 10, 12], metadata={'help': 'Output per channel for MSACNN'})

MODEL_CONFIG = MSACNNConfig

"""Model-specific configuration for STALSTM."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class STALSTMConfig(BaseExpConfig):
    SA_dim: int = field(default=10, metadata={'help': 'Spatial Attention dimension for TimeFilter'})
    TA_dim: int = field(default=10, metadata={'help': 'Temporal Attention dimension for TimeFilter'})

MODEL_CONFIG = STALSTMConfig

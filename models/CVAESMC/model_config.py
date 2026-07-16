"""Model-specific configuration for CVAESMC."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class CVAESMCConfig(BaseExpConfig):
    activation: str = field(default='gelu', metadata={'help': 'Activation function'})
    hidden_dim: int = field(default=10, metadata={'help': 'hidden dimension for each distributed GRU unit'})
    num_samples: int = field(default=10, metadata={'help': 'Number of samples for CVAESMC'})
    z_dim: int = field(default=10, metadata={'help': 'Latent dimension for CVAESMC and DMVAER'})
    output_type: str = field(default='mean', metadata={'help': 'Output type for CVAESMC'})

MODEL_CONFIG = CVAESMCConfig

"""Model-specific configuration for VRNN."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class VRNNConfig(BaseExpConfig):
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    activation: str = field(default='gelu', metadata={'help': 'Activation function'})
    z_dim: int = field(default=10, metadata={'help': 'Latent dimension for CVAESMC and DMVAER', 'prefix': 'zd', 'order': 103})
    x_embed_dim: int = field(default=16, metadata={'help': 'Number of layers for VRNN', 'prefix': 'xed', 'order': 101})
    z_embed_dim: int = field(default=16, metadata={'help': 'Number of layers for VRNN', 'prefix': 'zed', 'order': 102})

MODEL_CONFIG = VRNNConfig

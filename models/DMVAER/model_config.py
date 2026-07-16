"""Model-specific configuration for DMVAER."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class DMVAERConfig(BaseExpConfig):
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    n_components: int = field(default=3, metadata={'help': 'TCAVE type'})
    z_global_dim: int = field(default=16, metadata={'help': 'Global latent dimension for DMVAER'})
    z_local_dim: int = field(default=16, metadata={'help': 'Local latent dimension for DMVAER'})
    DMVAER_loss_weight: list[float] = field(default_factory=lambda: [0.1, 1, 1, 1, 0.01], metadata={'help': '0: x reconstruction, 1: y reconstruction, 2: KL_zt, 3: KL_zs, 4: KL_C'})

MODEL_CONFIG = DMVAERConfig

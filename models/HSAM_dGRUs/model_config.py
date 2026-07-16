"""Model-specific configuration for HSAM_dGRUs."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class HSAM_dGRUsConfig(BaseExpConfig):
    factor: int = field(default=1, metadata={'help': 'Factor of Attention'})
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 102})
    n_heads: int = field(default=8, metadata={'help': 'Number of attention heads', 'prefix': 'nh', 'order': 103})
    e_layers: int = field(default=1, metadata={'help': 'Number of distributed GRU layers.'})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    activation: str = field(default='gelu', metadata={'help': 'Activation function'})
    hidden_dim: int = field(default=10, metadata={'help': 'hidden dimension for each distributed GRU unit', 'prefix': 'hd', 'order': 100})
    use_true_y_in_train: bool = field(default=True, metadata={'help': 'Use y_{t-1} instead of the current ground-truth y_t as the HSAM_dGRUs quality query during training', 'prefix': 'uty', 'order': 101})

MODEL_CONFIG = HSAM_dGRUsConfig

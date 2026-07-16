"""Model-specific configuration for STDTAEm."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class STDTAEmConfig(BaseExpConfig):
    model_stage: str = field(default='finetune', metadata={'help': 'Model stage for models with pretraining', 'prefix': 'stage', 'order': 105})
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers'})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    hidden_dim: int = field(default=10, metadata={'help': 'hidden dimension for each distributed GRU unit', 'prefix': 'hd', 'order': 100})
    moving_avg: int = field(default=25, metadata={'help': 'moving average for Autoformer'})
    latent_dim: int = field(default=10, metadata={'help': 'Latent dimension for GTFTS', 'prefix': 'ld', 'order': 101})
    std_window: int = field(default=5, metadata={'help': 'Moving average window for STDTAEm decomposition', 'prefix': 'sw', 'order': 103})
    tae_beta: float = field(default=1.0, metadata={'help': 'Triplet loss weight for STDTAEm pretraining'})
    triplet_margin: float = field(default=1.0, metadata={'help': 'Triplet margin for STDTAEm pretraining'})
    tae_noise_std: float = field(default=0.05, metadata={'help': 'Positive sample Gaussian noise for STDTAEm'})
    tae_mask_ratio: float = field(default=0.1, metadata={'help': 'Positive sample random mask ratio for STDTAEm'})
    tae_backbone: str = field(default='mlp', metadata={'help': 'TAE encoder backbone for STDTAEm', 'prefix': 'tb', 'order': 104})
    tae_hidden_dims: list = field(default_factory=lambda: [], metadata={'help': 'Hidden dimensions for STDTAEm MLP TAE', 'prefix': 'th', 'order': 102})
    freeze_tae: bool = field(default=False, metadata={'help': 'Freeze pretrained TAE branches during STDTAEm finetuning'})

MODEL_CONFIG = STDTAEmConfig

"""Model-specific configuration for GTFTS."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class GTFTSConfig(BaseExpConfig):
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    hidden_dim: int = field(default=10, metadata={'help': 'hidden dimension for each distributed GRU unit'})
    latent_dim: int = field(default=10, metadata={'help': 'Latent dimension for GTFTS'})
    n_fft: int = field(default=8, metadata={'help': 'nfft dimension in STFT for GTFTS'})

MODEL_CONFIG = GTFTSConfig

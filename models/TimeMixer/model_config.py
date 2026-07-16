"""Model-specific configuration for TimeMixer."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class TimeMixerConfig(BaseExpConfig):
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers', 'prefix': 'el', 'order': 103})
    d_ff: int = field(default=1024, metadata={'help': 'Feed forward dimension', 'prefix': 'dff', 'order': 101})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    moving_avg: int = field(default=25, metadata={'help': 'moving average for Autoformer'})
    down_sampling_window: int = field(default=4, metadata={'help': 'Down sampling window for TimesMixer and TimeKAN '})
    channel_independence: bool = field(default=False, metadata={'help': 'if channel independence for TimesNet'})
    top_k: int = field(default=5, metadata={'help': 'Top k for TimesNet'})
    decomp_method: str = field(default='none', metadata={'help': '[moving_avg, dft_decomp]'})
    down_sampling_layers: int = field(default=2, metadata={'help': 'Number of down sampling layers for TimesMixer'})
    use_norm: bool = field(default=False, metadata={'help': 'If use normalization for TimesMixer'})
    down_sampling_method: str = field(default='max_pooling', metadata={'help': '[max, avg, conv'})

MODEL_CONFIG = TimeMixerConfig

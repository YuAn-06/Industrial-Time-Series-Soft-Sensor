"""Model-specific configuration for TimeKAN."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class TimeKANConfig(BaseExpConfig):
    d_model: int = field(default=512, metadata={'help': 'Model dimension', 'prefix': 'dm', 'order': 100})
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers', 'prefix': 'el', 'order': 102})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    moving_avg: int = field(default=25, metadata={'help': 'moving average for Autoformer'})
    down_sampling_window: int = field(default=4, metadata={'help': 'Down sampling window for TimesMixer and TimeKAN ', 'prefix': 'dsw', 'order': 105})
    channel_independence: bool = field(default=False, metadata={'help': 'if channel independence for TimesNet'})
    down_sampling_layers: int = field(default=2, metadata={'help': 'Number of down sampling layers for TimesMixer', 'prefix': 'dsl', 'order': 104})
    use_norm: bool = field(default=False, metadata={'help': 'If use normalization for TimesMixer'})
    begin_order: bool = field(default=False, metadata={'help': 'If use future temporal feature for TimeKAN'})

MODEL_CONFIG = TimeKANConfig

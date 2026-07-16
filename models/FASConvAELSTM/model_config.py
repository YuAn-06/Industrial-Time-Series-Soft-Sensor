"""Model-specific configuration for FASConvAELSTM."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class FASConvAELSTMConfig(BaseExpConfig):
    model_stage: str = field(default='finetune', metadata={'help': 'Model stage for models with pretraining', 'prefix': 'stage', 'order': 106})
    e_layers: int = field(default=1, metadata={'help': 'Encoder layers', 'prefix': 'el', 'order': 105})
    dropout: float = field(default=0.05, metadata={'help': 'Dropout rate'})
    activation: str = field(default='gelu', metadata={'help': 'Activation function'})
    hidden_dim: int = field(default=10, metadata={'help': 'hidden dimension for each distributed GRU unit', 'prefix': 'hd', 'order': 104})
    fa_lags: list[int] = field(default_factory=lambda: [0, 3, 5, 9], metadata={'help': 'Lag offsets used to build FA/CNN input matrices', 'prefix': 'lags', 'order': 100})
    fa_kernel_sizes: list[int] = field(default_factory=lambda: [3, 2, 2], metadata={'help': 'Feature-aligned ConvAE kernel sizes', 'prefix': 'fk', 'order': 101})
    fa_channels: list[int] = field(default_factory=lambda: [6, 10, 1], metadata={'help': 'Feature-aligned ConvAE output channels', 'prefix': 'fc', 'order': 102})
    fa_pretrain_learning_rates: list[float] = field(default_factory=lambda: [0.005, 0.01, 0.01], metadata={'help': 'Layer-wise FA-SConvAE pretraining learning rates', 'prefix': 'flr', 'order': 103})
    freeze_tae: bool = field(default=False, metadata={'help': 'Freeze pretrained TAE branches during STDTAEm finetuning'})

MODEL_CONFIG = FASConvAELSTMConfig

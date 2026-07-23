"""Model-specific configuration for PETC-TNet."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class PETCTNetConfig(BaseExpConfig):
    patch_len: int = field(default=16, metadata={"help": "Non-overlapping patch length", "prefix": "ps", "order": 100})
    kernel_size: int = field(default=2, metadata={"help": "TCCAN causal convolution kernel size", "prefix": "ks", "order": 101})
    tcn_channels: list[int] = field(
        default_factory=lambda: [32, 32, 32],
        metadata={"help": "Output channels of the three TCCAN layers"},
    )
    reduction_ratio: int = field(default=3, metadata={"help": "Channel-attention reduction ratio", "prefix": "rr", "order": 102})
    channel_attention_heads: int = field(default=4, metadata={"help": "Number of parallel channel-attention MLP heads", "prefix": "cah", "order": 103})
    d_model: int = field(default=32, metadata={"help": "Transformer latent dimension", "prefix": "dm", "order": 104})
    n_heads: int = field(default=8, metadata={"help": "Transformer attention heads", "prefix": "nh", "order": 105})
    e_layers: int = field(default=1, metadata={"help": "Transformer encoder layers", "prefix": "el", "order": 106})
    d_layers: int = field(default=1, metadata={"help": "Transformer decoder layers", "prefix": "dl", "order": 107})
    d_ff: int = field(default=128, metadata={"help": "Transformer feed-forward dimension", "prefix": "dff", "order": 108})
    dropout: float = field(default=0.05, metadata={"help": "Dropout rate"})
    activation: str = field(default="gelu", metadata={"help": "Transformer activation"})

    def validate(self) -> None:
        super().validate()
        if self.task != "short_term_forecasting":
            raise ValueError("PETC_TNet uses the benchmark's short_term_forecasting pipeline.")
        if self.pred_len != 1:
            raise ValueError("PETC_TNet paper reproduction requires pred_len=1.")
        if self.patch_len <= 0 or self.seq_len % self.patch_len != 0:
            raise ValueError("seq_len must be exactly divisible by patch_len for non-overlapping patches.")
        if not self.tcn_channels or any(channels <= 0 for channels in self.tcn_channels):
            raise ValueError("tcn_channels must contain positive channel sizes.")
        if self.kernel_size <= 0 or self.reduction_ratio <= 0 or self.channel_attention_heads <= 0:
            raise ValueError("kernel_size, reduction_ratio, and channel_attention_heads must be positive.")
        if self.d_model <= 0 or self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads.")
        if self.e_layers <= 0 or self.d_layers <= 0 or self.d_ff <= 0:
            raise ValueError("Transformer layer counts and d_ff must be positive.")


MODEL_CONFIG = PETCTNetConfig

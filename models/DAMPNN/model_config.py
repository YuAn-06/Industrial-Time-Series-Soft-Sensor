"""Model-specific configuration for DAMPNN."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class DAMPNNConfig(BaseExpConfig):
    d_model: int = field(
        default=20,
        metadata={"help": "Node representation dimension", "prefix": "dm", "order": 100},
    )
    graph_projection_dim: int = field(
        default=40,
        metadata={"help": "Projection dimension used to learn the micro graph", "prefix": "gpd", "order": 101},
    )
    attention_dim: int = field(
        default=40,
        metadata={"help": "Query/key/value dimension in dual self-attention", "prefix": "ad", "order": 102},
    )
    message_passing_layers: int = field(
        default=2,
        metadata={"help": "Number of dynamic message-passing layers", "prefix": "mpl", "order": 103},
    )
    graph_threshold: float = field(
        default=0.8,
        metadata={"help": "Adaptive graph edge threshold", "prefix": "gt", "order": 104},
    )
    graph_beta: float = field(
        default=0.01,
        metadata={"help": "Scale beta in micro graph learning", "prefix": "gb", "order": 105},
    )
    graph_lambda: float = field(
        default=30.0,
        metadata={"help": "Lambda in the adaptive momentum factor", "prefix": "gl", "order": 106},
    )
    dropout: float = field(default=0.0, metadata={"help": "Dropout before the regression head"})

    def validate(self) -> None:
        super().validate()
        if self.task != "soft_sensor":
            raise ValueError("DAMPNN supports only the soft_sensor task.")
        if self.d_model <= 0 or self.graph_projection_dim <= 0 or self.attention_dim <= 0:
            raise ValueError("DAMPNN representation and projection dimensions must be positive.")
        if self.message_passing_layers <= 0:
            raise ValueError("message_passing_layers must be positive.")
        if not 0.0 <= self.graph_threshold <= 1.0:
            raise ValueError("graph_threshold must be in [0, 1].")
        if self.graph_beta <= 0.0 or self.graph_lambda < 0.0:
            raise ValueError("graph_beta must be positive and graph_lambda non-negative.")


MODEL_CONFIG = DAMPNNConfig

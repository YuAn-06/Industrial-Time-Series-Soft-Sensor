"""Template declaration of model-specific hyperparameters."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class TemplateModelConfig(BaseExpConfig):
    """Parameters owned by the model rather than the benchmark workflow."""

    hidden_dim: int = field(default=64, metadata={"help": "Hidden dimension of the example projection head."})


MODEL_CONFIG = TemplateModelConfig

"""Model-specific configuration for Koopa."""

from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class KoopaConfig(BaseExpConfig):
    """This model has no additional architecture parameters."""

MODEL_CONFIG = KoopaConfig

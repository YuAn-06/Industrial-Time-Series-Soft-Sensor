"""Public package interface for PETC-TNet."""

from .model_arch import Model
from .model_config import MODEL_CONFIG
from .model_spec import MODEL_SPEC

__all__ = ["Model", "MODEL_CONFIG", "MODEL_SPEC"]

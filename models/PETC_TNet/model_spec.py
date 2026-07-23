"""Benchmark capability card for PETC-TNet."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="PETC_TNet",
    module="models.PETC_TNet",
    supported_tasks=("short_term_forecasting",),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
    paper_title="Patch-Decomposition-Enhanced TCN With Transformer for Soft Sensor Modeling",
    paper_url="https://doi.org/10.1109/JSEN.2025.3615736",
)

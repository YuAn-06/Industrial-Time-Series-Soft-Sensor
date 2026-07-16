"""Benchmark capability card for Transformer."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="Transformer",
    module="models.Transformer",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

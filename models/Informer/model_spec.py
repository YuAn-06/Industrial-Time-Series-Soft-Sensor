"""Benchmark capability card for Informer."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="Informer",
    module="models.Informer",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

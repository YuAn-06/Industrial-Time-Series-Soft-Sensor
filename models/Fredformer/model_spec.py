"""Benchmark capability card for Fredformer."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="Fredformer",
    module="models.Fredformer",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

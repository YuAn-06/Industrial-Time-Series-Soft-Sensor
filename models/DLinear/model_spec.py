"""Benchmark capability card for DLinear."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="DLinear",
    module="models.DLinear",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

"""Benchmark capability card for TimeKAN."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="TimeKAN",
    module="models.TimeKAN",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

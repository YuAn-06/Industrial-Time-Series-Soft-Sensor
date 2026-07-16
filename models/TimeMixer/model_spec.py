"""Benchmark capability card for TimeMixer."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="TimeMixer",
    module="models.TimeMixer",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

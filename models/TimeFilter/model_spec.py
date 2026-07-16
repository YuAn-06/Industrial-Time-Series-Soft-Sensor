"""Benchmark capability card for TimeFilter."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="TimeFilter",
    module="models.TimeFilter",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

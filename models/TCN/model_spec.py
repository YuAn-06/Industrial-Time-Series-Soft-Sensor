"""Benchmark capability card for TCN."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="TCN",
    module="models.TCN",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

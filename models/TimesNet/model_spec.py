"""Benchmark capability card for TimesNet."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="TimesNet",
    module="models.TimesNet",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

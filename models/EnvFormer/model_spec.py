"""Benchmark capability card for EnvFormer."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="EnvFormer",
    module="models.EnvFormer",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

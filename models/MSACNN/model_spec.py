"""Benchmark capability card for MSACNN."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="MSACNN",
    module="models.MSACNN",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

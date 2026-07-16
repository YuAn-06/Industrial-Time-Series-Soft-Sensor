"""Benchmark capability card for Autoformer."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="Autoformer",
    module="models.Autoformer",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

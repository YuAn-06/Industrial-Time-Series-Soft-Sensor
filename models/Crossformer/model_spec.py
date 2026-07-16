"""Benchmark capability card for Crossformer."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="Crossformer",
    module="models.Crossformer",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

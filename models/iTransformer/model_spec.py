"""Benchmark capability card for iTransformer."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="iTransformer",
    module="models.iTransformer",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

"""Benchmark capability card for FEDformer."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="FEDformer",
    module="models.FEDformer",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

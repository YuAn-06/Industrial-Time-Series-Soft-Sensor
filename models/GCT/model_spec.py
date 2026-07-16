"""Benchmark capability card for GCT."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="GCT",
    module="models.GCT",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

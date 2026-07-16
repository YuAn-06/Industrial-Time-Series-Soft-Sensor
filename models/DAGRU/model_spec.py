"""Benchmark capability card for DAGRU."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="DAGRU",
    module="models.DAGRU",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

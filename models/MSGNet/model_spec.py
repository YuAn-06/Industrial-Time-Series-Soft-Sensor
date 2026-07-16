"""Benchmark capability card for MSGNet."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="MSGNet",
    module="models.MSGNet",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

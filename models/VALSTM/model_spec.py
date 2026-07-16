"""Benchmark capability card for VALSTM."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="VALSTM",
    module="models.VALSTM",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

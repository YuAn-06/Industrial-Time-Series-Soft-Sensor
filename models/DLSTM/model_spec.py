"""Benchmark capability card for DLSTM."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="DLSTM",
    module="models.DLSTM",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

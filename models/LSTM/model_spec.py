"""Benchmark capability card for LSTM."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="LSTM",
    module="models.LSTM",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

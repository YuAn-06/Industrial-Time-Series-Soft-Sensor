"""Benchmark capability card for Nonstationary_Transformer."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="Nonstationary_Transformer",
    module="models.Nonstationary_Transformer",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

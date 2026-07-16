"""Benchmark capability card for ARDNN."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="ARDNN",
    module="models.ARDNN",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

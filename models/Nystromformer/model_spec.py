"""Benchmark capability card for Nystromformer."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="Nystromformer",
    module="models.Nystromformer",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

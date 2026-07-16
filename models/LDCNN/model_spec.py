"""Benchmark capability card for LDCNN."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="LDCNN",
    module="models.LDCNN",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

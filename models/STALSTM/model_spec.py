"""Benchmark capability card for STALSTM."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="STALSTM",
    module="models.STALSTM",
    supported_tasks=('soft_sensor',),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

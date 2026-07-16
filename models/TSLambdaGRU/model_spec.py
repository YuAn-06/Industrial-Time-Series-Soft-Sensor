"""Benchmark capability card for TSLambdaGRU."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="TSLambdaGRU",
    module="models.TSLambdaGRU",
    supported_tasks=('soft_sensor',),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

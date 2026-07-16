"""Benchmark capability card for GraphSAGE_IMATCN."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="GraphSAGE_IMATCN",
    module="models.GraphSAGE_IMATCN",
    supported_tasks=('soft_sensor',),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

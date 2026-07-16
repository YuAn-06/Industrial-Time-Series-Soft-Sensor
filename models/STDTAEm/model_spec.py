"""Benchmark capability card for STDTAEm."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="STDTAEm",
    module="models.STDTAEm",
    supported_tasks=('soft_sensor',),
    dataset_type="standard",
    loss_type="stdtaem",
    pretrain_stages=('pretrain',),
)

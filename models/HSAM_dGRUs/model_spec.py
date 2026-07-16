"""Benchmark capability card for HSAM_dGRUs."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="HSAM_dGRUs",
    module="models.HSAM_dGRUs",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="huber",
    pretrain_stages=(),
)

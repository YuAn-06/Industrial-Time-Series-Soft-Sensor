"""Benchmark capability card for VRNN."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="VRNN",
    module="models.VRNN",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="vrnn",
    pretrain_stages=(),
)

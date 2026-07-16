"""Benchmark capability card for DMVAER."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="DMVAER",
    module="models.DMVAER",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="multimode",
    loss_type="dmvaer",
    pretrain_stages=(),
)

"""Benchmark capability card for CVAESMC."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="CVAESMC",
    module="models.CVAESMC",
    supported_tasks=('short_term_forecasting',),
    dataset_type="standard",
    loss_type="cvaesmc",
    pretrain_stages=(),
)

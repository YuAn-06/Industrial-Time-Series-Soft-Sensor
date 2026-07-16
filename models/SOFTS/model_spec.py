"""Benchmark capability card for SOFTS."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="SOFTS",
    module="models.SOFTS",
    supported_tasks=('short_term_forecasting',),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

"""Benchmark capability card for Koopa."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="Koopa",
    module="models.Koopa",
    supported_tasks=('short_term_forecasting',),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

"""Benchmark capability card for SparseTSF."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="SparseTSF",
    module="models.SparseTSF",
    supported_tasks=('short_term_forecasting',),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

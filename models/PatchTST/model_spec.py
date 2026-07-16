"""Benchmark capability card for PatchTST."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="PatchTST",
    module="models.PatchTST",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
)

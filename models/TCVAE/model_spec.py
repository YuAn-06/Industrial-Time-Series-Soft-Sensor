"""Benchmark capability card for TCVAE."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="TCVAE",
    module="models.TCVAE",
    supported_tasks=('soft_sensor', 'short_term_forecasting'),
    dataset_type="standard",
    loss_type="tcvae",
    pretrain_stages=(),
)

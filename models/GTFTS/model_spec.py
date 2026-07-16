"""Benchmark capability card for GTFTS."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="GTFTS",
    module="models.GTFTS",
    supported_tasks=('short_term_forecasting',),
    dataset_type="standard",
    loss_type="gtfts",
    pretrain_stages=(),
)

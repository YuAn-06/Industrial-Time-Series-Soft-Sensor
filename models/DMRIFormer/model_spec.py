"""Benchmark capability card for DMRIFormer."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="DMRIFormer",
    module="models.DMRIFormer",
    supported_tasks=('short_term_forecasting',),
    dataset_type="multimode",
    loss_type="mse",
    pretrain_stages=(),
)

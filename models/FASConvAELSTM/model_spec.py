"""Benchmark capability card for FASConvAELSTM."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="FASConvAELSTM",
    module="models.FASConvAELSTM",
    supported_tasks=('soft_sensor',),
    dataset_type="lagged_matrix",
    loss_type="fasconvaelstm",
    pretrain_stages=('pretrain_l1', 'pretrain_l2', 'pretrain_l3'),
)

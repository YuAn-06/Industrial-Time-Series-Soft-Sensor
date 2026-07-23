"""Benchmark capability card for DAMPNN."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="DAMPNN",
    module="models.DAMPNN",
    supported_tasks=("soft_sensor",),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
    paper_title=(
        "DAMPNN: Dynamic Adaptive Message Passing Neural Network for "
        "Industrial Soft Sensor"
    ),
    paper_url="https://doi.org/10.1109/TII.2024.3475419",
)

"""Public execution interfaces for InduTS-SS.

Most users only need the following four functions:

- ``train_test_from_yaml``: train and test from a YAML configuration.
- ``test_from_checkpoint``: load and evaluate a complete model checkpoint.
- ``run_pretrain_finetune``: run the full pretraining/finetuning pipeline.
- ``run_finetune_only``: skip pretraining and finetune existing weights.

Functions from ``builder`` are primarily shared runner internals, but are also
exported for advanced use cases.
"""

from .builder import build_args_from_yaml, build_experiment, build_stage_args
from .pretrain_finetune import run_finetune_only, run_pretrain_finetune
from .test_checkpoint import test_checkpoint, test_from_checkpoint
from .train_test import train_test, train_test_from_yaml

__all__ = [
    "build_args_from_yaml",
    "build_experiment",
    "build_stage_args",
    "train_test",
    "train_test_from_yaml",
    "test_checkpoint",
    "test_from_checkpoint",
    "run_pretrain_finetune",
    "run_finetune_only",
]

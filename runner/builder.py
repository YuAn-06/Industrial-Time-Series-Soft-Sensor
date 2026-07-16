"""Build configs, experiments, loggers, and model state for runner workflows.

This module prepares objects but does not orchestrate complete train or test
workflows. Config overrides must be applied before ``prepare_run`` so that the
generated ``setting`` matches the effective experiment configuration.
"""

import copy
import os

import torch

from exp import get_exp_by_model_and_task
from utils import Logger, load_config, prepare_run, setup_seed


def apply_overrides(args, overrides=None):
    """Apply explicit overrides before building the setting; reject unknown keys."""
    for key, value in (overrides or {}).items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown config field: {key}")
        setattr(args, key, value)
    return args


def build_args_from_yaml(yaml_path, overrides=None):
    """Load YAML, apply overrides, and prepare the setting and result directory."""
    args = load_config(yaml_path, argv=[])
    apply_overrides(args, overrides)
    return prepare_run(args)


def build_experiment(args):
    """Set the random seed and create the Experiment selected by task and model."""
    setup_seed(args.seed)
    return get_exp_by_model_and_task(args)


def build_logger(args):
    """Create the run logger under ``args.save_dir``."""
    return Logger(args.save_dir)


def build_stage_args(base_args, stage, pretrained_checkpoint=""):
    """Create an independent config for one pretraining or finetuning stage.

    The base config is copied before setting ``model_stage`` and the pretrained
    checkpoint. Stage-specific ``pretrain_*`` or ``finetune_*`` epoch and
    learning-rate overrides are then applied.
    """
    args = copy.deepcopy(base_args)
    args.model_stage = stage
    args.pretrained_ckpt = (
        "" if stage in ("pretrain", "pretrain_l1") else pretrained_checkpoint
    )

    stage_prefix = "pretrain" if stage.startswith("pretrain_l") else stage
    stage_epoch = getattr(args, f"{stage_prefix}_epoch", -1)
    stage_learning_rate = getattr(
        args, f"{stage_prefix}_learning_rate", -1.0
    )
    if stage_epoch > 0:
        args.epoch = stage_epoch
    if stage_learning_rate > 0:
        args.learning_rate = stage_learning_rate

    return prepare_run(args)


def load_model_checkpoint(exp, checkpoint_path, strict=True):
    """Load a complete trained-model checkpoint into an Experiment.

    Supports a raw state dict, dictionaries containing ``state_dict`` or
    ``model_state_dict``, and DataParallel weights prefixed with ``module.``.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=exp.device)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    try:
        exp.model.load_state_dict(checkpoint, strict=strict)
    except RuntimeError as error:
        if not isinstance(checkpoint, dict):
            raise
        if all(key.startswith("module.") for key in checkpoint):
            checkpoint = {
                key.removeprefix("module."): value
                for key, value in checkpoint.items()
            }
            exp.model.load_state_dict(checkpoint, strict=strict)
        else:
            raise error
    return exp

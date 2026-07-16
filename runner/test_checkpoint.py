"""Standalone evaluation workflow for an existing complete checkpoint.

This module never trains the model. It reconstructs the model and data config
from YAML, loads the checkpoint, and calls ``Experiment.test``. Metrics,
predictions, and figures are written to ``evaluation/`` beside the checkpoint
by default so that original training artifacts are not overwritten.

The ``test_`` filename describes the workflow. ``__test__ = False`` prevents
pytest from collecting this module as a test file.
"""

import os

__test__ = False

from .builder import (
    build_args_from_yaml,
    build_experiment,
    build_logger,
    load_model_checkpoint,
)


def test_checkpoint(args, checkpoint_path, output_dir=None, strict=True):
    """Load a complete checkpoint and evaluate it with prepared args.

    ``strict=True`` requires the model architecture to match the checkpoint.
    Disable strict loading only when partial-weight compatibility is intended.
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(checkpoint_path), "evaluation")

    args.save_dir = os.path.normpath(output_dir) + os.sep
    os.makedirs(args.save_dir, exist_ok=True)

    logger = build_logger(args)
    try:
        logger.info(f"Setting: {args.setting}")
        logger.info(f"Checkpoint: {checkpoint_path}")
        exp = build_experiment(args)
        load_model_checkpoint(exp, checkpoint_path, strict=strict)
        logger.info("Start checkpoint testing...")
        exp.test(logger)
        return exp
    finally:
        logger.remove_handles()


def test_from_checkpoint(
    yaml_path,
    checkpoint_path,
    overrides=None,
    output_dir=None,
    strict=True,
):
    """Rebuild an experiment from YAML, then load and evaluate a checkpoint."""
    args = build_args_from_yaml(yaml_path, overrides=overrides)
    return test_checkpoint(
        args,
        checkpoint_path,
        output_dir=output_dir,
        strict=strict,
    )

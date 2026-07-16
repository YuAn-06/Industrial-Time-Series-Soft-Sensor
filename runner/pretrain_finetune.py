"""Pretraining and finetuning workflow orchestration.

Two common workflows are exposed:

- ``run_pretrain_finetune`` runs all required pretraining stages followed by
  finetuning and optional testing.
- ``run_finetune_only`` skips pretraining and starts from a supplied pretrained
  checkpoint.

Pretraining stages are read from each model's ``ModelSpec.pretrain_stages``.
For example, FASConvAELSTM declares ``pretrain_l1 -> pretrain_l2 ->
pretrain_l3``, while STDTAEm declares ``pretrain``. Every stage reuses
``train_test``; this module only manages dependencies, checkpoint handoff, and
checkpoint archival.
"""

import os
import shutil

from models.registry import get_model_spec
from utils import load_config

from .builder import build_stage_args
from .train_test import train_test


VALID_STAGES = (
    "pretrain",
    "pretrain_l1",
    "pretrain_l2",
    "pretrain_l3",
    "finetune",
)

STAGE_DEPENDENCIES = {
    "pretrain_l2": ("pretrain_l1",),
    "pretrain_l3": ("pretrain_l2",),
    "finetune": ("pretrain_l3", "pretrain", "pretrain_l2", "pretrain_l1"),
}


def stage_checkpoint_path(checkpoint_dir, stage):
    """Return the standard checkpoint path for a stage in the shared folder."""
    return os.path.join(checkpoint_dir, f"checkpoint_{stage}.pth")


def resolve_stage_input(checkpoint_dir, stage, previous_checkpoint=""):
    """Use the previous output first, then search declared stage dependencies."""
    if previous_checkpoint:
        return previous_checkpoint
    for dependency in STAGE_DEPENDENCIES.get(stage, ()):
        checkpoint = stage_checkpoint_path(checkpoint_dir, dependency)
        if os.path.isfile(checkpoint):
            return checkpoint
    return ""


def archive_stage_checkpoint(source_path, checkpoint_dir, stage):
    """Archive a stage's best weights under its standard checkpoint name."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    destination = stage_checkpoint_path(checkpoint_dir, stage)
    shutil.copy2(source_path, destination)
    return destination


def run_stage_pipeline(
    yaml_path,
    stages,
    test_stages=("finetune",),
    checkpoint_dir="",
    initial_checkpoint="",
):
    """Run an explicit sequence of pretraining and/or finetuning stages.

    This advanced interface supports resuming intermediate stages or defining
    custom stage sequences. Prefer the two explicit public workflows below for
    standard full-pipeline and finetune-only runs.
    """
    base_args = load_config(yaml_path, argv=[])
    stages = tuple(stages)
    test_stages = tuple(test_stages)

    if not checkpoint_dir:
        checkpoint_dir = build_stage_args(base_args, "pretrain").save_dir
    checkpoint_dir = os.path.normpath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if initial_checkpoint:
        if not os.path.isfile(initial_checkpoint):
            raise FileNotFoundError(
                f"Initial checkpoint not found: {initial_checkpoint}"
            )
        if stages != ("finetune",):
            raise ValueError(
                "initial_checkpoint is only valid for a finetune-only run."
            )

    previous_checkpoint = initial_checkpoint
    stage_results = {}
    for stage in stages:
        if stage not in VALID_STAGES:
            raise ValueError(f"Unsupported stage: {stage}")

        stage_input = resolve_stage_input(
            checkpoint_dir,
            stage,
            previous_checkpoint=previous_checkpoint,
        )
        if stage in ("pretrain_l2", "pretrain_l3") and not stage_input:
            raise FileNotFoundError(f"{stage} requires its preceding checkpoint.")
        if stage == "finetune" and not stage_input:
            raise FileNotFoundError(
                "Finetuning requires a pretrained checkpoint."
            )

        stage_args = build_stage_args(
            base_args,
            stage,
            pretrained_checkpoint=stage_input,
        )
        train_test(
            stage_args,
            do_train=True,
            do_test=stage in test_stages,
        )

        source_checkpoint = os.path.join(stage_args.save_dir, "checkpoint.pth")
        if not os.path.isfile(source_checkpoint):
            raise FileNotFoundError(
                f"{stage} checkpoint not found: {source_checkpoint}"
            )

        checkpoint = source_checkpoint
        if stage.startswith("pretrain"):
            checkpoint = archive_stage_checkpoint(
                source_checkpoint,
                checkpoint_dir,
                stage,
            )

        stage_results[stage] = {
            "checkpoint": checkpoint,
            "source_checkpoint": source_checkpoint,
            "save_dir": stage_args.save_dir,
        }
        previous_checkpoint = checkpoint

    return stage_results


def run_pretrain_finetune(
    yaml_path,
    checkpoint_dir="",
    do_test=True,
):
    """Run all model-specific pretraining stages, then finetune and test."""
    base_args = load_config(yaml_path, argv=[])
    pretrain_stages = get_model_spec(base_args.model).pretrain_stages
    if not pretrain_stages:
        raise ValueError(
            f"Model '{base_args.model}' does not define pretraining stages."
        )
    stages = tuple(pretrain_stages) + ("finetune",)
    test_stages = ("finetune",) if do_test else ()
    return run_stage_pipeline(
        yaml_path,
        stages=stages,
        test_stages=test_stages,
        checkpoint_dir=checkpoint_dir,
    )


def run_finetune_only(
    yaml_path,
    pretrained_checkpoint,
    checkpoint_dir="",
    do_test=True,
):
    """Skip pretraining and finetune from the supplied pretrained checkpoint."""
    test_stages = ("finetune",) if do_test else ()
    return run_stage_pipeline(
        yaml_path,
        stages=("finetune",),
        test_stages=test_stages,
        checkpoint_dir=checkpoint_dir,
        initial_checkpoint=pretrained_checkpoint,
    )

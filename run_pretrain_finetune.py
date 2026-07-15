"""
Run pretraining and finetuning from one YAML file.

The runner keeps exp classes unchanged: it derives two argument objects from the
same YAML, trains pretraining stages, then injects the produced checkpoint into
model_stage='finetune'. FA-SConvAE-LSTM can use layer-wise stages:
pretrain_l1 -> pretrain_l2 -> pretrain_l3 -> finetune.
"""

import copy
import argparse
import os
import shutil
import sys
from dataclasses import asdict


VALID_STAGES = ("pretrain", "pretrain_l1", "pretrain_l2", "pretrain_l3", "finetune")

STAGE_DEPENDENCIES = {
    "pretrain_l2": ("pretrain_l1",),
    "pretrain_l3": ("pretrain_l2",),
    "finetune": ("pretrain_l3", "pretrain", "pretrain_l2", "pretrain_l1"),
}


def stage_checkpoint_path(checkpoint_dir, stage):
    """Return the shared checkpoint filename for one pipeline stage."""
    return os.path.join(checkpoint_dir, f"checkpoint_{stage}.pth")


def resolve_stage_input(checkpoint_dir, stage, previous_ckpt=""):
    """Find the checkpoint that should initialize a stage."""
    if previous_ckpt:
        return previous_ckpt

    for dependency in STAGE_DEPENDENCIES.get(stage, ()):
        checkpoint = stage_checkpoint_path(checkpoint_dir, dependency)
        if os.path.isfile(checkpoint):
            return checkpoint

    return ""


def archive_stage_checkpoint(source_path, checkpoint_dir, stage):
    """Copy a stage's best checkpoint into the shared checkpoint folder."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    destination = stage_checkpoint_path(checkpoint_dir, stage)
    shutil.copy2(source_path, destination)
    return destination



def refresh_run_fields(args):
    from utils.configs import build_setting

    setting = build_setting(args)
    save_dir = os.path.join(".", "results", args.model, setting) + os.sep
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir

    selected_keys = [
        "data_name", "model", "task", "seq_len", "label_len", "pred_len",
        "dropout", "activation", "batch_size", "learning_rate", "epoch",
        "patience", "d_model", "d_ff", "n_heads", "e_layers", "d_layers",
        "hidden_dim", "moving_avg", "individual", "kernel_size", "distil",
        "num_layers", "period_len", "model_type", "down_sampling_layers",
        "down_sampling_window", "x_embed_dim", "z_embed_dim", "z_dim",
        "node_dim", "conv_channel", "skip_channel", "propalpha", "gcn_depth",
        "model_stage", "std_window", "latent_dim",
        "fa_lags", "fa_kernel_sizes", "fa_channels", "fa_pretrain_learning_rates",
    ]
    config_dict = asdict(args)
    args.setting = ", ".join(
        f"{k}: {v}" for k, v in config_dict.items() if k in selected_keys
    )
    return args


def make_stage_args(base_args, stage, pretrained_ckpt=""):
    args = copy.deepcopy(base_args)
    args.model_stage = stage
    args.pretrained_ckpt = "" if stage in ("pretrain", "pretrain_l1") else pretrained_ckpt

    stage_prefix = "pretrain" if stage.startswith("pretrain_l") else stage
    stage_epoch = getattr(args, f"{stage_prefix}_epoch", -1)
    stage_learning_rate = getattr(args, f"{stage_prefix}_learning_rate", -1.0)
    
    if stage_epoch > 0:
        args.epoch = stage_epoch
    if stage_learning_rate > 0:
        args.learning_rate = stage_learning_rate

    return refresh_run_fields(args)


def run_stage(args, stage_name, do_test=False):
    from exp import get_exp_by_model_and_task
    from utils import Logger, print_args, setup_seed

    logger = Logger(args.save_dir)
    try:
        print(f"==== {stage_name}: {args.model} ====")
        print_args(args)
        logger.info(f"using configs: {args.setting}")
        logger.info(f"Start {stage_name} training...")

        setup_seed(args.seed)
        exp = get_exp_by_model_and_task(args)
        exp.train(logger)

        if do_test:
            logger.info(f"Start {stage_name} testing...")
            exp.test(logger)
    finally:
        logger.remove_handles()


def load_args_from_yaml(yaml_path):
    from utils import Parse_arguments

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        return Parse_arguments(yaml_path)
    finally:
        sys.argv = original_argv


def run_stage_pipeline(
    yaml_path,
    stages,
    test_stages,
    checkpoint_dir="",
    initial_ckpt="",
):
    base_args = load_args_from_yaml(yaml_path)

    if not checkpoint_dir:
        # Use the normal model_stage=pretrain setting folder as the shared
        # checkpoint folder. This keeps all layer-wise checkpoints beside the
        # standard pretraining run instead of creating a separate directory.
        checkpoint_dir = make_stage_args(base_args, "pretrain").save_dir
    checkpoint_dir = os.path.normpath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Shared checkpoint folder: {checkpoint_dir}")

    if initial_ckpt:
        if not os.path.exists(initial_ckpt):
            raise FileNotFoundError(f"initial checkpoint not found: {initial_ckpt}")
        stages = ("finetune",)

    previous_ckpt = initial_ckpt
    stage_results = {}

    for stage in stages:
        pretrained_ckpt = resolve_stage_input(
            checkpoint_dir,
            stage,
            previous_ckpt=previous_ckpt,
        )
        if stage in ("pretrain_l2", "pretrain_l3") and not pretrained_ckpt:
            expected = ", ".join(
                stage_checkpoint_path(checkpoint_dir, dependency)
                for dependency in STAGE_DEPENDENCIES[stage]
            )
            raise FileNotFoundError(
                f"{stage} requires a preceding layer checkpoint. Expected: {expected}"
            )
        if stage == "finetune" and checkpoint_dir and not pretrained_ckpt:
            print(
                f"Warning: no pretrained checkpoint found in {checkpoint_dir}; "
                "finetuning will start from randomly initialized weights."
            )

        stage_args = make_stage_args(
            base_args,
            stage,
            pretrained_ckpt=pretrained_ckpt,
        )
        run_stage(stage_args, stage, do_test=stage in test_stages)

        source_checkpoint = os.path.join(stage_args.save_dir, "checkpoint.pth")
        if not os.path.exists(source_checkpoint):
            raise FileNotFoundError(
                f"{stage} checkpoint not found: {source_checkpoint}"
            )
        if stage.startswith("pretrain"):
            checkpoint = archive_stage_checkpoint(
                source_checkpoint,
                checkpoint_dir,
                stage,
            )
        else:
            # Fine-tuning is a separate experiment. Keep its best model in the
            # normal stagefinetune result folder instead of mixing it with the
            # shared pretraining checkpoints.
            checkpoint = source_checkpoint

        stage_results[stage] = {
            "checkpoint": checkpoint,
            "source_checkpoint": source_checkpoint,
            "save_dir": stage_args.save_dir,
        }
        previous_ckpt = checkpoint

    return stage_results


def parse_pipeline_args():
    parser = argparse.ArgumentParser(
        description="Run pretraining and finetuning from one YAML file."
    )
    parser.add_argument(
        "--yaml",
        "--yaml_path",
        dest="yaml_path",
        default="./scripts/SS_task/SRU_scripts/yaml/FASConvAELSTM.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        # default=("pretrain_l1", "pretrain_l2", "pretrain_l3", "finetune"),
        default=("finetune",),
        choices=VALID_STAGES,
        help="Stages to run, for example: --stages pretrain_l1 pretrain_l2 pretrain_l3 finetune",
    )
    parser.add_argument(
        "--test_stages",
        nargs="+",
        default=("finetune",),
        choices=VALID_STAGES,
        help="Stages to test after training, for example: --test_stages finetune",
    )
    parser.add_argument(
        "--checkpoint_dir",
        # default="",
        default="./results/FASConvAELSTM/SRU_FASConvAELSTM_soft_sensor_sl4_ll4_pl6_bt100_lr0p01_ep200_pat30_lags0-3-5-9_fk3-2-2_fc6-10-1_flr0p005-0p01-0p01_hd64_el1_stagepretrain/",
        help=(
            "Shared folder used to save and auto-load stage checkpoints. "
            "Files are named checkpoint_<stage>.pth. When omitted, the normal "
            "model_stage=pretrain result folder is used."
        ),
    )
    parser.add_argument(
        "--initial_ckpt",
        default="",
        help=(
            "Legacy option: one explicit checkpoint for direct finetuning. "
            "Prefer --checkpoint_dir."
        ),
    )
    return parser.parse_args()


def main():
    cli_args = parse_pipeline_args()
    results = run_stage_pipeline(
        cli_args.yaml_path,
        stages=tuple(cli_args.stages),
        test_stages=tuple(cli_args.test_stages),
        checkpoint_dir=cli_args.checkpoint_dir,
        initial_ckpt=cli_args.initial_ckpt,
    )

    print("==== done ====")
    pretrain_checkpoints = [
        result["checkpoint"]
        for stage, result in results.items()
        if stage.startswith("pretrain")
    ]
    if pretrain_checkpoints:
        print(
            "pretrain_checkpoint_dir: "
            f"{os.path.dirname(pretrain_checkpoints[0])}"
        )
    for stage, result in results.items():
        print(f"{stage}_checkpoint: {result['checkpoint']}")


if __name__ == "__main__":
    main()

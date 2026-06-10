"""
Run pretraining and finetuning from one YAML file.

The runner keeps exp classes unchanged: it derives two argument objects from the
same YAML, trains model_stage='pretrain', then injects the produced checkpoint
into model_stage='finetune'.
"""

import copy
import argparse
import os
import sys
from dataclasses import asdict


VALID_STAGES = ("pretrain", "finetune")



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
    ]
    config_dict = asdict(args)
    args.setting = ", ".join(
        f"{k}: {v}" for k, v in config_dict.items() if k in selected_keys
    )
    return args


def make_stage_args(base_args, stage, pretrained_ckpt=""):
    args = copy.deepcopy(base_args)
    args.model_stage = stage
    args.pretrained_ckpt = "" if stage == "pretrain" else pretrained_ckpt

    stage_epoch = getattr(args, f"{stage}_epoch", -1)
    stage_learning_rate = getattr(args, f"{stage}_learning_rate", -1.0)
    
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


def run_stage_pipeline(yaml_path, stages, test_stages, initial_ckpt=""):
    base_args = load_args_from_yaml(yaml_path)
    if initial_ckpt:
        if not os.path.exists(initial_ckpt):
            raise FileNotFoundError(f"initial checkpoint not found: {initial_ckpt}")
        stages = ("finetune",)

    previous_ckpt = initial_ckpt
    stage_results = {}

    for stage in stages:
        stage_args = make_stage_args(base_args, stage, pretrained_ckpt=previous_ckpt)
        run_stage(stage_args, stage, do_test=stage in test_stages)

        checkpoint = os.path.join(stage_args.save_dir, "checkpoint.pth")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"{stage} checkpoint not found: {checkpoint}")

        stage_results[stage] = {
            "checkpoint": checkpoint,
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
        default="./scripts/SS_task/SRU_scripts/yaml/STDTAEm.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=("pretrain", "finetune"), # ('pretrain', 'finetune') ('pretrain',)  ('finetune',)
        choices=VALID_STAGES,
        help="Stages to run, for example: --stages pretrain finetune",
    )
    parser.add_argument(
        "--test_stages",
        nargs="+",
        default=("finetune",),
        choices=VALID_STAGES,
        help="Stages to test after training, for example: --test_stages finetune",
    )
    parser.add_argument(
        "--initial_ckpt",
        default="",
        help="Existing pretrain checkpoint for direct finetuning.",
    ) # if you have a initial checkpoint, please provide the path as "./results/STDTAEm/{results_name}/checkpoint.pth" and set stages = ("finetune",)
    return parser.parse_args()


def main():
    cli_args = parse_pipeline_args()
    results = run_stage_pipeline(
        cli_args.yaml_path,
        stages=tuple(cli_args.stages),
        test_stages=tuple(cli_args.test_stages),
        initial_ckpt=cli_args.initial_ckpt,
    )

    print("==== done ====")
    for stage, result in results.items():
        print(f"{stage}_checkpoint: {result['checkpoint']}")
        print(f"{stage}_save_dir: {result['save_dir']}")


if __name__ == "__main__":
    main()

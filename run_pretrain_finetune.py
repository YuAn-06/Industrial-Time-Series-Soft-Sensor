"""Convenient entry point for full pretrain/finetune or finetune-only runs."""

import argparse

from runner import run_finetune_only, run_pretrain_finetune


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pretraining/finetuning from one YAML file."
    )
    parser.add_argument(
        "--yaml",
        default="./scripts/SS_task/SRU_scripts/yaml/FASConvAELSTM.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "finetune"),
        default="full",
        help="Run all pretraining stages or finetune from an existing checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Required when --mode finetune.",
    )
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--no_test", action="store_true")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    if cli_args.mode == "finetune":
        if not cli_args.checkpoint:
            raise ValueError("--checkpoint is required when --mode finetune.")
        results = run_finetune_only(
            cli_args.yaml,
            pretrained_checkpoint=cli_args.checkpoint,
            checkpoint_dir=cli_args.checkpoint_dir,
            do_test=not cli_args.no_test,
        )
    else:
        results = run_pretrain_finetune(
            cli_args.yaml,
            checkpoint_dir=cli_args.checkpoint_dir,
            do_test=not cli_args.no_test,
        )

    print("==== done ====")
    for stage, result in results.items():
        print(f"{stage}_checkpoint: {result['checkpoint']}")


if __name__ == "__main__":
    main()

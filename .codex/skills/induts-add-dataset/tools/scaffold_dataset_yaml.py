#!/usr/bin/env python
"""Create a minimal InduTS-SS YAML for a newly onboarded dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


TASK_DIR = {
    "soft_sensor": "SS_task",
    "short_term_forecasting": "LSF_task",
}


def dropped_quality_column(data_name: str, target: str) -> str | None:
    """Mirror utils.tools.del_columns without importing project modules."""
    if data_name == "PPGAS":
        return "NOX" if target == "CO" else "CO"
    if data_name == "SRU":
        return "H2S" if target == "SO2" else "SO2"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaffold an InduTS-SS dataset YAML.")
    parser.add_argument("--dataset", required=True, help="Dataset name used as data_name.")
    parser.add_argument("--csv", required=True, help="Dataset CSV path.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--task", required=True, choices=sorted(TASK_DIR), help="Task type.")
    parser.add_argument("--model", default="DLinear", help="Model name for the YAML.")
    parser.add_argument("--seq-len", type=int, default=16, help="Input sequence length.")
    parser.add_argument("--label-len", type=int, default=8, help="Decoder label length.")
    parser.add_argument("--pred-len", type=int, default=6, help="Prediction length.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epoch", type=int, default=3, help="Epoch count for smoke YAML.")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=2021, help="Random seed.")
    parser.add_argument("--use-cuda", action="store_true", help="Enable CUDA in generated YAML.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing YAML.")
    parser.add_argument("--out", default="", help="Optional output YAML path.")
    return parser.parse_args()


def infer_dimensions(csv_path: Path, data_name: str, target: str, task: str) -> tuple[int, int]:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column not found in CSV: {target}")

    x_columns = [col for col in df.columns if col.startswith("x_")]
    drop_col = dropped_quality_column(data_name, target)
    if task == "soft_sensor":
        if x_columns:
            features = x_columns
        else:
            features = [col for col in df.columns if col not in {"date", "mode", target, drop_col}]
    else:
        if x_columns:
            features = x_columns + ([target] if target not in x_columns else [])
        else:
            features = [col for col in df.columns if col not in {"date", "mode", drop_col}]

    return len(features), 1


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    c_in, c_out = infer_dimensions(csv_path, args.dataset, args.target, args.task)
    task_dir = TASK_DIR[args.task]
    out_path = (
        Path(args.out)
        if args.out
        else Path("scripts") / task_dir / f"{args.dataset}_scripts" / "yaml" / f"{args.model}.yaml"
    )
    if out_path.exists() and not args.force:
        raise FileExistsError(f"YAML already exists: {out_path}. Use --force to overwrite.")

    params = {
        "model": args.model,
        "task": args.task,
        "data_name": args.dataset,
        "data_path": str(csv_path).replace("\\", "/"),
        "target": args.target,
        "enc_in": c_in,
        "dec_in": c_in,
        "C_in": c_in,
        "C_out": c_out,
        "seq_len": args.seq_len,
        "label_len": 1 if args.task == "soft_sensor" else args.label_len,
        "pred_len": 1 if args.task == "soft_sensor" else args.pred_len,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epoch": args.epoch,
        "patience": args.patience,
        "seed": args.seed,
        "use_cuda": bool(args.use_cuda),
        "gpu": 0,
        "use_tensorboard": False,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"params": params}, handle, sort_keys=False, allow_unicode=True)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

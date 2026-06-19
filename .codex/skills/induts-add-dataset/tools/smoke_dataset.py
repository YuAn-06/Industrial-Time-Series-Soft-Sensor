#!/usr/bin/env python
"""Prepare or run a small smoke test for an InduTS-SS dataset YAML."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test an InduTS-SS dataset YAML.")
    parser.add_argument("--yaml", required=True, help="YAML file to validate or run.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    parser.add_argument("--epoch", type=int, default=1, help="Temporary epoch override.")
    parser.add_argument("--patience", type=int, default=1, help="Temporary patience override.")
    parser.add_argument("--batch-size", type=int, default=8, help="Temporary batch-size override.")
    parser.add_argument(
        "--runner",
        default="run_with_yaml.py",
        help="Runner to execute. This checkout's run_with_yaml.py may need manual YAML selection.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("params"), dict):
        raise ValueError("Expected YAML with top-level 'params' mapping.")

    required = ["model", "task", "data_name", "data_path", "target", "seq_len", "batch_size"]
    missing = [key for key in required if key not in data["params"]]
    if missing:
        raise ValueError(f"Missing required params: {', '.join(missing)}")
    return data


def write_temp_yaml(source: Path, data: dict, epoch: int, patience: int, batch_size: int) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="induts_smoke_"))
    temp_path = temp_dir / source.name
    smoke_data = dict(data)
    smoke_params = dict(data["params"])
    smoke_params["epoch"] = epoch
    smoke_params["patience"] = patience
    smoke_params["batch_size"] = batch_size
    smoke_params["use_tensorboard"] = False
    smoke_data["params"] = smoke_params
    with temp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(smoke_data, handle, sort_keys=False, allow_unicode=True)
    return temp_path


def main() -> int:
    args = parse_args()
    yaml_path = Path(args.yaml)
    data = load_yaml(yaml_path)
    temp_yaml = write_temp_yaml(yaml_path, data, args.epoch, args.patience, args.batch_size)

    runner = Path(args.runner)
    command = [sys.executable, str(runner), "--yaml", str(temp_yaml)]

    print(f"Validated YAML: {yaml_path}")
    print(f"Temporary smoke YAML: {temp_yaml}")
    print("Command:", " ".join(command))
    print(
        "Note: if run_with_yaml.py does not accept --yaml in this checkout, "
        "set yaml_name inside run_with_yaml.py or use run_pretrain_finetune.py for staged models."
    )

    if args.dry_run:
        shutil.rmtree(temp_yaml.parent, ignore_errors=True)
        return 0
    if not runner.exists():
        raise FileNotFoundError(f"Runner not found: {runner}")

    completed = subprocess.run(command, check=False)
    shutil.rmtree(temp_yaml.parent, ignore_errors=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

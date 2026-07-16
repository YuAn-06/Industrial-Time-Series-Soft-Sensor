#!/usr/bin/env python
"""Run one-epoch CPU smoke tests for InduTS-SS YAML configs."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_OVERRIDES = {
    "use_cuda": False,
    "device": "cpu",
    "use_multi_gpu": False,
    "use_amp": False,
    "use_tensorboard": False,
    "num_workers": 0,
    "epoch": 1,
    "patience": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU smoke tests for InduTS-SS YAML configs.")
    parser.add_argument("--yaml", nargs="+", required=True, help="One or more YAML configs to smoke.")
    parser.add_argument("--batch-size", type=int, default=8, help="Temporary smoke batch size.")
    parser.add_argument("--max-train-batches", type=int, default=2, help="Maximum train batches in the smoke epoch.")
    parser.add_argument("--max-test-batches", type=int, default=2, help="Maximum test batches for output shape checks.")
    parser.add_argument("--output-dir", default="smoke_runs", help="Directory for smoke artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Create smoke YAMLs and report commands only.")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary YAMLs after successful runs.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict) or "params" not in data:
        raise ValueError(f"YAML must contain a top-level 'params' object: {path}")
    return data


def make_smoke_yaml(src: Path, tmp_dir: Path, batch_size: int) -> Path:
    data = load_yaml(src)
    params = dict(data["params"])
    params.update(DEFAULT_OVERRIDES)
    if batch_size > 0:
        params["batch_size"] = batch_size

    data["params"] = params
    tmp_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(src.with_suffix("")))
    tmp_path = tmp_dir / f"{safe_name}_smoke.yaml"
    with tmp_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
    return tmp_path


def import_project_modules():
    os.environ.setdefault("MPLBACKEND", "Agg")
    repo_root = str(Path.cwd())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from exp.exp_factory import get_exp_by_model_and_task
    from utils.configs import Parse_arguments
    from utils.tools import setup_seed

    return Parse_arguments, get_exp_by_model_and_task, setup_seed


def tensor_shape(value: Any) -> list[int] | None:
    if hasattr(value, "shape"):
        return [int(dim) for dim in value.shape]
    return None


def forward_probe(exp: Any) -> dict[str, Any]:
    _, loader = exp._get_data(flag="test")
    batch = next(iter(loader))
    batch = {key: value.to(exp.device) for key, value in batch.items()}

    exp.model.eval()
    import torch

    with torch.no_grad():
        try:
            raw_outputs = exp.model(**batch, flag="test")
        except TypeError:
            raw_outputs = exp.model(**batch)
        pred = exp._select_pred(raw_outputs, flag="test")
        gt = exp._select_gt(**batch, flag="test")

    pred_shape = tensor_shape(pred)
    gt_shape = tensor_shape(gt)
    if pred_shape != gt_shape:
        raise AssertionError(f"Prediction shape {pred_shape} does not match ground-truth shape {gt_shape}.")
    return {
        "prediction_shape": pred_shape,
        "ground_truth_shape": gt_shape,
        "batch_keys": sorted(batch.keys()),
        "raw_output_type": type(raw_outputs).__name__,
    }


def smoke_train_and_test(exp: Any, max_train_batches: int, max_test_batches: int) -> dict[str, Any]:
    import numpy as np
    import torch

    _, train_loader = exp._get_data(flag="train")
    _, test_loader = exp._get_data(flag="test")
    optimizer = exp._select_optimizer()

    train_losses = []
    exp.model.train()
    for batch_index, batch in enumerate(train_loader):
        if batch_index >= max_train_batches:
            break
        batch = {key: value.to(exp.device) for key, value in batch.items()}
        optimizer.zero_grad()
        outputs = exp.model(**batch)
        loss_outputs = exp._select_pred(outputs, flag="train") if exp.args.task == "short_term_forecasting" else outputs
        trues = exp._select_gt(**batch, flag="train")
        loss = exp.loss.calculate_loss(loss_outputs, trues, flag="train")
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

    if not train_losses:
        raise RuntimeError("No train batches were available for smoke training.")

    test_shapes = []
    exp.model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            if batch_index >= max_test_batches:
                break
            batch = {key: value.to(exp.device) for key, value in batch.items()}
            try:
                raw_outputs = exp.model(**batch, flag="test")
            except TypeError:
                raw_outputs = exp.model(**batch)
            pred = exp._select_pred(raw_outputs, flag="test")
            gt = exp._select_gt(**batch, flag="test")
            pred_shape = tensor_shape(pred)
            gt_shape = tensor_shape(gt)
            if pred_shape != gt_shape:
                raise AssertionError(f"Prediction shape {pred_shape} does not match ground-truth shape {gt_shape}.")
            test_shapes.append({"prediction_shape": pred_shape, "ground_truth_shape": gt_shape})

    if not test_shapes:
        raise RuntimeError("No test batches were available for smoke testing.")

    return {
        "train_batches": len(train_losses),
        "test_batches": len(test_shapes),
        "mean_train_loss": float(np.mean(train_losses)),
        "test_shapes": test_shapes,
    }


def run_one(src_yaml: Path, smoke_yaml: Path, output_dir: Path, max_train_batches: int, max_test_batches: int) -> dict[str, Any]:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{smoke_yaml.stem}.log"

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    result: dict[str, Any] = {
        "source_yaml": str(src_yaml),
        "smoke_yaml": str(smoke_yaml),
        "log_path": str(log_path),
        "status": "failed",
    }

    try:
        Parse_arguments, get_exp_by_model_and_task, setup_seed = import_project_modules()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            old_argv = sys.argv[:]
            try:
                sys.argv = [old_argv[0]]
                args = Parse_arguments(str(smoke_yaml))
            finally:
                sys.argv = old_argv
            args_dict = asdict(args)
            result.update(
                {
                    "model": args.model,
                    "task": args.task,
                    "data_name": args.data_name,
                    "target": args.target,
                    "epoch": args.epoch,
                    "batch_size": args.batch_size,
                    "device": args.device,
                    "use_cuda": args.use_cuda,
                    "expected_output": (
                        [args.batch_size, args.C_out]
                        if args.task == "soft_sensor"
                        else [args.batch_size, args.pred_len, args.C_out]
                    ),
                    "config": args_dict,
                }
            )

            setup_seed(args.seed)
            exp = get_exp_by_model_and_task(args)
            result["forward_probe"] = forward_probe(exp)
            result["smoke_run"] = smoke_train_and_test(exp, max_train_batches, max_test_batches)

        combined_output = stdout_buffer.getvalue() + stderr_buffer.getvalue()
        shape_match = re.search(r"test shape:\s*(.*?)\s*trues shape:\s*(.*)", combined_output)
        if shape_match:
            result["test_shape_line"] = shape_match.group(0)
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        log_path.write_text(stdout_buffer.getvalue() + stderr_buffer.getvalue(), encoding="utf-8")

    return result


def write_report(path: Path, results: list[dict[str, Any]]) -> None:
    lines = ["# InduTS Smoke Report", ""]
    for item in results:
        lines.extend(
            [
                f"## {Path(item['source_yaml']).name}",
                "",
                f"- Status: `{item['status']}`",
                f"- Source YAML: `{item['source_yaml']}`",
                f"- Smoke YAML: `{item['smoke_yaml']}`",
                f"- Log: `{item['log_path']}`",
            ]
        )
        for key in ["model", "task", "data_name", "target", "epoch", "batch_size", "device", "use_cuda"]:
            if key in item:
                lines.append(f"- {key}: `{item[key]}`")
        if "forward_probe" in item:
            probe = item["forward_probe"]
            lines.append(f"- forward prediction shape: `{probe['prediction_shape']}`")
            lines.append(f"- forward ground-truth shape: `{probe['ground_truth_shape']}`")
        if "smoke_run" in item:
            smoke = item["smoke_run"]
            lines.append(f"- train batches: `{smoke['train_batches']}`")
            lines.append(f"- test batches: `{smoke['test_batches']}`")
            lines.append(f"- mean train loss: `{smoke['mean_train_loss']:.6g}`")
            lines.append(f"- test batch shapes: `{smoke['test_shapes']}`")
        if "test_shape_line" in item:
            lines.append(f"- {item['test_shape_line']}")
        if "error" in item:
            lines.append(f"- Error: `{item['error']}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tmp_dir = output_dir / "tmp_yaml"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for yaml_arg in args.yaml:
        src = Path(yaml_arg)
        if not src.exists():
            results.append({"source_yaml": str(src), "smoke_yaml": "", "log_path": "", "status": "failed", "error": "YAML not found"})
            continue
        smoke_yaml = make_smoke_yaml(src, tmp_dir, args.batch_size)
        if args.dry_run:
            results.append(
                {
                    "source_yaml": str(src),
                    "smoke_yaml": str(smoke_yaml),
                    "log_path": "",
                    "status": "planned",
                    "command": f"{sys.executable} .codex/skills/induts-smoke/scripts/smoke_models.py --yaml {src}",
                }
            )
        else:
            results.append(run_one(src, smoke_yaml, output_dir, args.max_train_batches, args.max_test_batches))

    report_path = output_dir / "smoke_report.md"
    json_path = output_dir / "smoke_report.json"
    write_report(report_path, results)
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.keep_tmp and not args.dry_run and all(item.get("status") == "passed" for item in results):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Saved smoke report: {report_path}")
    print(f"Saved smoke JSON: {json_path}")
    for item in results:
        print(f"{item.get('status', 'unknown').upper()}: {item.get('source_yaml')}")
        if item.get("error"):
            print(f"  {item['error']}")

    return 0 if all(item.get("status") in {"passed", "planned"} for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

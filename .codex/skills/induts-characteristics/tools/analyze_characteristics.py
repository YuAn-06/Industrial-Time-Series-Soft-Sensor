#!/usr/bin/env python
"""Run data/data_visualization.py for the InduTS characteristics skill."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thin wrapper around data/data_visualization.py."
    )
    parser.add_argument("--csv", default=None, help="Optional CSV path passed to data_visualization.py.")
    parser.add_argument("--target", default=None, help="Optional target column passed to data_visualization.py.")
    parser.add_argument("--dataset", "--data-name", default="MP", dest="data_name")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--acf-lags", type=int, default=100)
    parser.add_argument("--tsne-sample-rows", type=int, default=3000)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [
        sys.executable,
        "data/data_visualization.py",
        "--data-name",
        args.data_name,
        "--plot-format",
        args.plot_format,
        "--acf-lags",
        str(args.acf_lags),
        "--tsne-sample-rows",
        str(args.tsne_sample_rows),
    ]
    if args.csv:
        command.extend(["--csv", args.csv])
    if args.target:
        command.extend(["--target", args.target])
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])
    if args.show:
        command.append("--show")
    if args.no_plots:
        command.append("--no-plots")

    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

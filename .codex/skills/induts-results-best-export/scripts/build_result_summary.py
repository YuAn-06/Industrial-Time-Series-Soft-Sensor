#!/usr/bin/env python
"""Build compact cross-dataset summaries from exported InduTS best-result CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_COLUMNS = [
    "dataset",
    "model",
    "mae",
    "mse",
    "rmse",
    "mape",
    "mspe",
    "wape",
    "corr_or_r2",
    "setting",
    "run_dir",
    "artifact_dir",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-dataset best_per_model.csv exports into summary CSV/Markdown files."
    )
    parser.add_argument(
        "--input-root",
        default="report",
        help="Exporter output root containing <task>/<dataset>/best_per_model.csv files.",
    )
    parser.add_argument("--output-dir", default="", help="Directory for summary files. Defaults to input-root.")
    parser.add_argument("--task", default="", help="Optional task directory filter, such as soft_sensor.")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset to include. Repeat for multiple.")
    parser.add_argument("--model", action="append", default=[], help="Model to include. Repeat for multiple.")
    parser.add_argument("--metric", default="mse", help="Metric column used for sorting and matrix values.")
    parser.add_argument(
        "--prefix",
        default="best_summary",
        help="Output file prefix, e.g. selected_soft_sensor_best_summary.",
    )
    parser.add_argument(
        "--excel",
        action="store_true",
        help="Also write an .xlsx workbook when openpyxl or another Excel writer is installed.",
    )
    return parser.parse_args()


def fmt(value: object) -> str:
    try:
        return f"{float(value):.6g}"
    except Exception:
        return ""


def discover_csvs(input_root: Path, task: str) -> list[Path]:
    if task:
        search_root = input_root / task
        return sorted(search_root.glob("*/best_per_model.csv"))
    return sorted(input_root.glob("*/*/best_per_model.csv"))


def load_combined(args: argparse.Namespace) -> pd.DataFrame:
    input_root = Path(args.input_root)
    frames = []
    for csv_path in discover_csvs(input_root, args.task):
        df = pd.read_csv(csv_path)
        if not df.empty:
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        return combined

    if args.dataset:
        allowed = {item.lower() for item in args.dataset}
        combined = combined[combined["dataset"].astype(str).str.lower().isin(allowed)]
    if args.model:
        allowed = set(args.model)
        combined = combined[combined["model"].isin(allowed)]

    columns = [col for col in DEFAULT_COLUMNS if col in combined.columns]
    for extra in ("rank_metric", "rank_direction", "metric_value"):
        if extra in combined.columns and extra not in columns:
            columns.append(extra)
    combined = combined[columns]
    if args.metric in combined.columns:
        combined = combined.sort_values(["dataset", args.metric, "model"])
    else:
        combined = combined.sort_values(["dataset", "model"])
    return combined


def build_matrix(combined: pd.DataFrame, metric: str, models: list[str]) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame(columns=models)
    model_order = models or sorted(combined["model"].dropna().unique())
    datasets = sorted(combined["dataset"].dropna().unique())
    matrix = pd.DataFrame(index=datasets, columns=model_order)
    if metric not in combined.columns:
        return matrix
    for _, row in combined.iterrows():
        if row["model"] in matrix.columns:
            matrix.loc[row["dataset"], row["model"]] = row[metric]
    return matrix


def write_markdown(path: Path, combined: pd.DataFrame, matrix: pd.DataFrame, args: argparse.Namespace, excel_written: bool) -> None:
    metric_label = args.metric
    task_note = f" for `{args.task}`" if args.task else ""
    lines = [
        "# InduTS Result Summary",
        "",
        f"Combined best-per-model exports{task_note}. Matrix metric: `{metric_label}`.",
        "",
        "## Best Results",
        "",
        "| Dataset | Model | MAE | MSE | RMSE | MAPE | WAPE | Corr/R2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in combined.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("model", "")),
                    fmt(row.get("mae", "")),
                    fmt(row.get("mse", "")),
                    fmt(row.get("rmse", "")),
                    fmt(row.get("mape", "")),
                    fmt(row.get("wape", "")),
                    fmt(row.get("corr_or_r2", "")),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Coverage",
        "",
    ]
    if matrix.empty:
        lines.append("No matching result rows were found.")
    else:
        headers = ["Dataset", *[str(col) for col in matrix.columns]]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for dataset, row in matrix.iterrows():
            coverage = ["yes" if pd.notna(value) and value != "" else "-" for value in row]
            lines.append("| " + " | ".join([str(dataset), *coverage]) + " |")

    lines += [
        "",
        "## Files",
        "",
        f"- Summary CSV: `{args.prefix}.csv`",
        f"- {metric_label} matrix CSV: `{args.prefix}_{metric_label}_matrix.csv`",
    ]
    if args.excel:
        if excel_written:
            lines.append(f"- Excel workbook: `{args.prefix}.xlsx`")
        else:
            lines.append("- Excel workbook was requested but not written because no Excel writer is installed.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = load_combined(args)
    matrix = build_matrix(combined, args.metric, args.model)

    csv_out = output_dir / f"{args.prefix}.csv"
    matrix_out = output_dir / f"{args.prefix}_{args.metric}_matrix.csv"
    md_out = output_dir / f"{args.prefix}.md"
    xlsx_out = output_dir / f"{args.prefix}.xlsx"

    combined.to_csv(csv_out, index=False)
    matrix.to_csv(matrix_out)

    excel_written = False
    if args.excel:
        try:
            with pd.ExcelWriter(xlsx_out) as writer:
                combined.to_excel(writer, index=False, sheet_name="best_by_dataset_model")
                matrix.to_excel(writer, sheet_name=f"{args.metric}_matrix")
            excel_written = True
        except ModuleNotFoundError:
            excel_written = False

    write_markdown(md_out, combined, matrix, args, excel_written)

    print(f"Rows: {len(combined)}")
    print(f"Summary CSV: {csv_out}")
    print(f"Matrix CSV: {matrix_out}")
    if args.excel:
        print(f"Excel: {xlsx_out if excel_written else 'not written'}")
    print(f"Markdown: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

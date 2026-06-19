#!/usr/bin/env python
"""Inspect a CSV before onboarding it as an InduTS-SS dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def dropped_quality_column(data_name: str, target: str) -> str | None:
    """Mirror utils.tools.del_columns without importing project modules."""
    if data_name == "PPGAS":
        return "NOX" if target == "CO" else "CO"
    if data_name == "SRU":
        return "H2S" if target == "SO2" else "SO2"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an InduTS-SS dataset CSV.")
    parser.add_argument("--csv", required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--target", default="", help="Target column name.")
    parser.add_argument("--data-name", default="", help="Optional InduTS data_name for dataset-specific drops.")
    parser.add_argument("--sample-rows", type=int, default=5, help="Preview rows to include.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    target = args.target
    columns = list(df.columns)
    x_columns = [col for col in columns if col.startswith("x_")]
    has_date = "date" in columns
    has_mode = "mode" in columns
    target_exists = bool(target) and target in columns
    data_name = args.data_name or csv_path.parent.name
    drop_col = dropped_quality_column(data_name, target)

    if x_columns:
        soft_sensor_features = x_columns
        forecasting_features = x_columns + ([target] if target_exists and target not in x_columns else [])
    else:
        excluded_soft = {"date", "mode", drop_col}
        if target_exists:
            excluded_soft.add(target)
        soft_sensor_features = [col for col in columns if col not in excluded_soft]
        forecasting_features = [col for col in columns if col not in {"date", "mode", drop_col}]

    missing = df.isna().sum()
    report = {
        "csv": str(csv_path),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": columns,
        "numeric_columns": list(df.select_dtypes(include="number").columns),
        "has_date": has_date,
        "has_mode": has_mode,
        "target": target,
        "target_exists": target_exists,
        "data_name": data_name,
        "dataset_specific_drop": drop_col,
        "x_prefixed_columns": x_columns,
        "feature_candidates": {
            "soft_sensor": soft_sensor_features,
            "short_term_forecasting": forecasting_features,
        },
        "suggested_dimensions": {
            "soft_sensor": {"C_in": len(soft_sensor_features), "C_out": 1 if target_exists else 0},
            "short_term_forecasting": {"C_in": len(forecasting_features), "C_out": 1 if target_exists else 0},
        },
        "missing_values": {col: int(value) for col, value in missing.items() if int(value) > 0},
        "preview": df.head(args.sample_rows).to_dict(orient="records"),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

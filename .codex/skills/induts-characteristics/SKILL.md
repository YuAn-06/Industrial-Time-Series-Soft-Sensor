---
name: induts-characteristics
description: Use when inspecting, visualizing, or reporting InduTS-SS dataset characteristics, including stationarity, autocorrelation, outliers, missing values, split drift, target correlations, mode distributions, and data patterns before benchmark runs or dataset onboarding.
---

# InduTS Characteristics

## Purpose

Use this skill to summarize dataset behavior before running or extending the InduTS-SS benchmark. The skill runs `data/data_visualization.py`, which is the source of the plots and report.

## Quick Start

Run the characteristics report for a known dataset CSV:

```bash
python .codex/skills/induts-characteristics/tools/analyze_characteristics.py --csv data/DC/debutanizer_column.csv --target y_1 --dataset DC
```

Equivalent direct command:

```bash
python data/data_visualization.py --data-name DC --csv data/DC/debutanizer_column.csv --target y_1
```

For a faster text-only pass:

```bash
python .codex/skills/induts-characteristics/tools/analyze_characteristics.py --csv data/SRU/SRU_data.csv --target SO2 --dataset SRU --no-plots
```

Artifacts are written to:

```text
data/<dataset>/characteristics report/
```

## Workflow

1. Identify the dataset, CSV path, and target column from the user request, YAML, or `data/data_provider.py`.
2. Inspect the CSV by running `data/data_visualization.py` directly or through the thin skill wrapper. Pass `--target` whenever the task mentions a target.
3. Treat input features as all CSV columns except `date`, `mode`, and the selected target. Preserve CSV column order.
4. Use `--max-features` when the dataset has many columns and plots would become unreadable.
5. Report the generated Markdown path, key stationarity/outlier/correlation findings, embedded plot names, and any caveats.

## Script Capabilities

`data/data_visualization.py` computes and saves:

- CSV shape, columns, date/mode/target presence, missing-value counts.
- Feature candidates using the InduTS-SS feature rule: exclude `date`, `mode`, and `target`.
- Descriptive statistics for numeric columns.
- ADF stationarity test when `statsmodels` is installed.
- Ljung-Box autocorrelation test when `statsmodels` is installed.
- IQR outlier percentage.
- Spearman correlations among numeric columns and top absolute correlations with the target.
- Standard 70/10/20 split summary and train/test mean drift.
- Saved plots embedded in the Markdown report: time series grid, ACF grid, Spearman heatmap, and train/test 2D projection.

## Plot Guidance

- Plots are saved by default and embedded in `*_characteristics.md`; use `--no-plots` for CI, remote shells, or quick diagnosis.
- Use `--plot-format png` for readable local artifacts.
- Use `--tsne` only for small or sampled datasets; by default the script uses PCA for the 2D train/test projection.
- Use `--sample-rows` to cap expensive visualizations while keeping the report deterministic.

## Interpretation Rules

- ADF `p_value < 0.05` suggests stationarity; otherwise call it non-stationary, not broken.
- Large Ljung-Box significance means autocorrelation is present, which is expected for time-series data.
- High IQR outlier percentages are prompts to inspect units, operating modes, and abnormal periods before changing loaders.
- Correlation with the target is descriptive only; do not use it as the sole basis for feature removal in benchmark comparisons.
- Keep fairness visible: dataset characterization may justify documentation or preprocessing notes, but should not silently change split, scaler, target, or feature rules.

## Relation To Existing Code

`tools/analyze_characteristics.py` is only a compatibility wrapper. The authoritative implementation is `data/data_visualization.py`.

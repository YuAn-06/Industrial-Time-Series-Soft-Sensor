---
name: induts-results-best-export
description: Use when analyzing InduTS-SS benchmark results under results/ to find the best run for each model by prediction task and dataset, export grouped reports under report/, collect each best model's checkpoint, setting evidence, metrics, predictions, and ground truth files, or build cross-dataset/model summary CSV/Markdown/Excel tables from exported best_per_model.csv files.
---

# InduTS Results Best Export

## Purpose

Use this skill to turn raw InduTS-SS `results/<Model>/<setting>/` folders into reproducible best-result packages. Reports are grouped by task and dataset under `report/<task>/<dataset>/`. After per-dataset export, optionally build compact cross-dataset/model summaries from the generated `best_per_model.csv` files.

## Workflow

1. Clarify or infer the task filter: `SS_task`/`soft_sensor` or `LSF_task`/`short_term_forecasting`, dataset, optional prediction setting filters, target, and ranking metric.
2. Inspect local `results/` before reporting. Do not assume every run has `metrics.npy`, `pred.npy`, or `checkpoint.pth`.
3. Run the bundled exporter from the repository root.
4. Check the console summary and open the generated workbook, CSV, or manifest if the result count looks surprising.
5. If the user asks to "整理", "汇总", compare selected models, or combine datasets, run the summary builder on the exporter output.
6. Report the `report/` root, grouped workbook/CSV paths, summary paths, ranking metric/direction, number of discovered runs, number of best model rows, and any missing artifact warnings.

## Export Command

Default ranking uses `mse` with lower values better:

```bash
python .codex/skills/induts-results-best-export/scripts/export_best_results.py --task soft_sensor --dataset DC
```

This writes:

```text
report/<task>/<dataset>/best_results.xlsx
report/<task>/<dataset>/best_artifacts/<Model>/
report/<task>/<dataset>/manifest.json
report/index.json
```

Common examples:

```bash
python .codex/skills/induts-results-best-export/scripts/export_best_results.py --task short_term_forecasting --dataset DC --pred-len 5 --metric mse
python .codex/skills/induts-results-best-export/scripts/export_best_results.py --task SS_task --dataset SRU --seq-len 16 --metric rmse --output-dir report
python .codex/skills/induts-results-best-export/scripts/export_best_results.py --task LSF_task --dataset DC --pred-len 5 --metric corr_or_r2 --direction max
```

Use `--no-checkpoints` when the user only wants lightweight prediction/metric files and not model parameter files.

## Cross-Dataset Summary Command

After exporting one or more `best_per_model.csv` files, combine them into a compact summary:

```bash
python .codex/skills/induts-results-best-export/scripts/build_result_summary.py --input-root report --task soft_sensor --metric mse --prefix soft_sensor_best_summary
```

This writes:

```text
report/soft_sensor_best_summary.csv
report/soft_sensor_best_summary_mse_matrix.csv
report/soft_sensor_best_summary.md
```

Use repeated `--model` and `--dataset` filters for selected-model reports:

```bash
python .codex/skills/induts-results-best-export/scripts/build_result_summary.py ^
  --input-root report/selected_soft_sensor_results ^
  --output-dir report/selected_soft_sensor_results ^
  --task soft_sensor ^
  --model PatchTST --model GraphSAGE_IMATCN --model HSAM_dGRUs --model iTransformer ^
  --metric mse ^
  --prefix selected_soft_sensor_best_summary
```

Use `--excel` when an Excel writer such as `openpyxl` is installed. Without it, the summary builder still writes CSV and Markdown.

## Ranking Rules

- Select the best run independently within each model directory.
- Use `metrics.npy` when available. The repository stores metrics as `mae, mse, rmse, mape, mspe, wape, corr_or_r2`.
- Prefer lower values for error metrics: `mae`, `mse`, `rmse`, `mape`, `mspe`, `wape`.
- Prefer higher values for `corr_or_r2`, `corr`, or `r2`.
- Mark runs without parseable metrics as `missing_metrics`; do not select them as best unless the user explicitly asks for an inventory only.

## Artifact Policy

For each best run, collect:

- Model parameters: `checkpoint*.pth`, `*.ckpt`, `*.pt` unless `--no-checkpoints` is set.
- Prediction outputs: `pred.*`, `true.*`, `metrics.*`, `test.*`.
- Evidence/config files: `log.log`, `*.log`, `*.yaml`, `*.yml`, `args.*`, `config.*`, `*.json`.
- Matching source YAML from `scripts/SS_task/<Dataset>_scripts/yaml/<Model>.yaml` or `scripts/LSF_task/<Dataset>_scripts/yaml/<Model>.yaml` when it exists.

Keep copied artifacts under `best_artifacts/<Model>/` inside the chosen output directory.

## Excel Columns

Keep the workbook compact. Include model, task, dataset, ranking metric, metrics, status, `setting`, run directory, artifact directory, and copied-file summary. Do not expand every parsed hyperparameter into separate Excel columns; `setting` is the parameter record.

## Summary Columns

The summary builder keeps a compact row set: dataset, model, `mae`, `mse`, `rmse`, `mape`, `mspe`, `wape`, `corr_or_r2`, `setting`, run directory, artifact directory, and status. The matrix CSV uses datasets as rows and selected models as columns, with values from the chosen `--metric`.

## Validation

After running the exporter:

- Confirm each expected `report/<task>/<dataset>/best_results.xlsx` exists. If Excel dependencies are unavailable, the script writes CSV fallbacks and prints that limitation.
- Confirm each group `manifest.json` and the root `report/index.json` exist.
- Check that each best row has a `metric_value`, `run_dir`, and `artifact_dir`.
- When using `build_result_summary.py`, confirm the summary CSV, metric matrix CSV, and Markdown files exist.
- If important artifacts are missing, say exactly which model/run is incomplete instead of hiding the gap.

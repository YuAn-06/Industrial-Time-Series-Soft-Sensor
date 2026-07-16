---
name: induts-add-dataset
description: Use when adding, inspecting, scaffolding, validating, or documenting a new dataset for the InduTS-SS benchmark, including CSV schema checks, target selection, SS_task or LSF_task YAML generation, data_provider mapping decisions, and smoke tests for new dataset onboarding.
---

# InduTS Add Dataset

## Purpose

Use this skill to onboard a new dataset into the InduTS-SS benchmark with the smallest reliable change set. Prefer reusing existing dataset loaders and YAML patterns before editing core code.

## Workflow

1. Inspect the CSV:

```bash
python .codex/skills/induts-add-dataset/tools/inspect_dataset.py --csv ./data/NewDataset/data.csv --target Y
```

2. Decide whether existing loaders are enough:

- Use `Dataset_Custom` for `task: short_term_forecasting`.
- Use `Dataset_Custom_4_Soft_Sensor` for `task: soft_sensor`.
- Use multimode loaders only when the CSV has a `mode` column or an external mode-label file.
- Edit `data/data_provider.py` only when the new `data_name` is not already mapped.
- Edit `data/data_loader.py` only for genuinely special preprocessing, split, target, missing-value, or multimode behavior.

3. Place the data under:

```text
data/<Dataset>/<file>.csv
```

4. Generate a minimal YAML:

```bash
python .codex/skills/induts-add-dataset/tools/scaffold_dataset_yaml.py --dataset NewDataset --csv ./data/NewDataset/data.csv --target Y --task soft_sensor --model DLinear
python .codex/skills/induts-add-dataset/tools/scaffold_dataset_yaml.py --dataset NewDataset --csv ./data/NewDataset/data.csv --target Y --task short_term_forecasting --model DLinear
```

5. Run a smoke check:

```bash
python .codex/skills/induts-add-dataset/tools/smoke_dataset.py --yaml ./scripts/SS_task/NewDataset_scripts/yaml/DLinear.yaml --dry-run
```

Remove `--dry-run` only when dependencies are installed and the run should actually start.

6. Update docs when the dataset is public benchmark coverage:

- `readme.md`
- `.codex/skills/induts-benchmark-workflow/references/dataset-catalog.md`
- Result/report files that list supported datasets

## Dataset Contract

- CSV must be readable by `pandas.read_csv`.
- `target` must be a CSV column.
- The target column generally refers to the column containing the quality variable.
- Optional `date` enables time features.
- Optional `mode` enables multi-mode logic.
- If process-variable columns are prefixed with `x_`, the loader uses those as features.
- The input features should include all features except for the Target column, as well as the mode and date columns.
- Standard split is 70% train, 10% validation, 20% test via `data/data_loader.py::_get_borders`.

## Fairness Rules

- Keep the same seed, split rule, scaler policy, metrics, and train/validation/test protocol unless the user explicitly asks to change benchmark policy.
- For comparisons, align `seq_len`, `label_len`, `pred_len`, `batch_size`, `epoch`, `patience`, and optimizer settings across models.
- Do not silently tune the new dataset differently from existing datasets.
- Start with a small smoke YAML before generating a full model matrix.

## Tool Notes

- `tools/inspect_dataset.py` prints JSON describing columns, shape, missing values, date/mode presence, target validity, feature candidates, and suggested dimensions.
- `tools/scaffold_dataset_yaml.py` creates one minimal YAML under the standard `scripts/<task>/<dataset>_scripts/yaml/` directory.
- `tools/smoke_dataset.py` validates YAML structure and can create a temporary low-epoch YAML; with `--dry-run` it prints the command instead of running it.

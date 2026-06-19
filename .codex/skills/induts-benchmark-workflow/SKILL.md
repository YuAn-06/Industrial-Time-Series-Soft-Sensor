---
name: induts-benchmark-workflow
description: "Use when working with the InduTS-SS benchmark for industrial time-series soft sensors: understand the repository, run SS_task or LSF_task experiments, choose datasets/models/configs, reproduce results, troubleshoot YAML/CUDA/data issues, summarize outputs, or maintain benchmark fairness by adding datasets, scripts, smoke tests, and result reports."
---

# InduTS-SS Benchmark Global Workflow

## Role

Use this skill as the top-level operating manual for the InduTS-SS benchmark repository. It should give the agent a global map first, then route into smaller `.codex/skills/induts-*` skills only when needed.

The benchmark covers industrial time-series soft sensors with two main task families:

- `SS_task`: soft sensor regression / sequential estimation, where historical process variables estimate the current hard-to-measure quality variable.
- `LSF_task`: long-sequence or short-term forecasting, where historical process and/or quality variables forecast future quality variables.

Keep fairness, reproducibility, and task consistency visible in every recommendation.

## First Response Checklist

1. Classify the request: run/use, explain/compare, troubleshoot, summarize results, or maintain/extend.
2. Inspect local files before assuming details. Prefer `rg --files`, then targeted reads.
3. Identify the task family (`SS_task` or `LSF_task`), dataset, model, YAML/script path, and hardware setting.
4. Route to the smallest relevant InduTS skill listed in `Agent Skills`.
5. Give commands from the repository root and state whether they are Windows-friendly or Bash/WSL/Linux-oriented.

## Repository Map

Important root files:

- `readme.md`: user-facing benchmark overview, supported models/datasets, installation, and project structure.
- `run_with_yaml.py`: recommended reproducible single-experiment entry point for YAML configs.
- `run_pretrain_finetune.py`: two-stage pretrain/finetune runner for models such as `STDTAEm`.
- `run.py`: command-line runner used by shell scripts and batch workflows.
- `requirements.txt`: Python dependencies.

Core packages:

- `data/`: dataset classes, preprocessing, split/scaling logic, and CSV data folders.
- `exp/`: experiment lifecycle. `exp_factory.py` selects `Exp_Soft_Sensor` or `Exp_Short_Term_Forecasting`.
- `models/`: model implementations. Each model is usually a single `models/<Model>.py` file.
- `layers/`: shared neural network blocks used by model files.
- `utils/`: config parsing, dataclass config surface, logging, metrics, scalers, seeds, and training helpers.
- `scripts/`: benchmark YAML and shell matrices, grouped by task and dataset.
- `results/`: experiment outputs, checkpoints, logs, metrics, predictions, and TensorBoard events.

## Execution Pipeline

YAML-driven flow:

1. `run_with_yaml.py` chooses a YAML under `scripts/`.
2. `utils.configs.Parse_arguments(yaml_path)` builds defaults from `Init_parser()`, overlays YAML `params`, validates with `ExpConfigs`, builds `args.setting`, and creates `results/<model>/<setting>/`.
3. `exp.exp_factory.get_exp_by_model_and_task(args)` selects:
   - `Exp_Soft_Sensor` for `task: soft_sensor`
   - `Exp_Short_Term_Forecasting` for `task: short_term_forecasting`
4. The experiment builds dataloaders through `data.data_provider.data_provider`.
5. The dataloader chooses a dataset class from `data_dict` using `data_name`, task, multimode settings, and selected model.
6. The experiment trains, validates, tests, logs metrics, and saves outputs under `results/`.

## Config And Script Layout

YAML files live below:

- `scripts/SS_task/<Dataset>_scripts/yaml/<Model>.yaml`
- `scripts/LSF_task/<Dataset>_scripts/yaml/<Model>.yaml`
- Some LSF shell batches also live in horizon folders such as `scripts/LSF_task/DC_scripts/sl16_pl6/`.

Common YAML fields are under a top-level `params` object. Important keys include:

- Data/task: `task`, `data_name`, `data_path`, `target`, `C_in`, `C_out`, `seq_len`, `label_len`, `pred_len`
- Training: `batch_size`, `learning_rate`, `epoch`, `patience`, `seed`, `lradj`, `weight_decay`
- Hardware: `use_cuda`, `device`, `gpu`, `device_ids`, `use_multi_gpu`
- Model: `model`, `d_model`, `d_ff`, `n_heads`, `e_layers`, `d_layers`, plus model-specific keys in `utils/configs.py`
- Two-stage: `model_stage`, `pretrained_ckpt`, `pretrain_epoch`, `finetune_epoch`, `pretrain_learning_rate`, `finetune_learning_rate`

When changing configs, keep identical dataset split, seed, sequence length, prediction length, training budget, scaler behavior, and metric definitions for fair model comparisons.

## Datasets

Currently documented benchmark datasets:

| Dataset | Local data | Main use |
| --- | --- | --- |
| `DC` | `data/DC/debutanizer_column.csv` | Debutanizer column soft sensor; optional multimode labels in `mode_labels.txt` |
| `SRU` | `data/SRU/SRU_data.csv` | Sulfur recovery unit soft sensor |
| `Ironmaking` | `data/Ironmaking/Ironmaking.csv` | Silicon content soft sensor / forecasting |
| `MP` | `data/MP/MP_data.csv` | Mining process concentrate quality |
| `PPGAS` | `data/PPGAS/gt_2012.csv` | Gas turbine NOx/CO soft sensor / forecasting |

Dataset selection is centralized in `data/data_provider.py`. Core dataset classes are in `data/data_loader.py`:

- `Dataset_Custom`: forecasting mode.
- `Dataset_Custom_4_Soft_Sensor`: soft sensor regression / sequential estimation.
- `Dataset_MultiMode`: forecasting with mode labels.
- `Dataset_MultiMode_4_Soft_Sensor`: soft sensor mode with mode labels.

The standard split is 70% train, 10% validation, 20% test through `_get_borders`.

## Models

Model implementations live in `models/`. Examples include classical soft-sensor models, time-series foundation models, transformer variants, recurrent models, CNN models, VAE models, GNN-style models, and newer architectures.

Before adding or editing a model:

- Check an adjacent model with similar call signature.
- Check `Exp_*` model invocation and loss handling.
- Add model-specific config defaults in `utils/configs.py` and `utils/ExpConfigs.py` if needed.
- Add YAMLs under both task families only when the model genuinely supports those tasks.
- Run a small YAML smoke run before expanding the benchmark matrix.

## Outputs And Reporting

Outputs are normally saved under:

```text
results/<Model>/<setting>/
```

The setting string is built in `utils.configs.build_setting` from dataset, model, task, common training fields, and selected model-specific fields.

When reporting results, include:

- Task family and exact YAML/script path.
- Dataset, target, sequence length, label length, prediction length, seed.
- Model and model-specific hyperparameters that affect capacity or input shape.
- Metrics and whether inverse scaling was used.
- Hardware notes, CUDA/device settings, and whether AMP/multi-GPU was enabled.

## Agent Skills

| Skill | Use When | Wraps / Key Entry Points |
| --- | --- | --- |
| `induts-create-env` | Set up or troubleshoot the benchmark Python environment, PyTorch/CUDA wheels, conda, pip, GPU driver compatibility, or `torch.cuda.is_available()` issues. | `.codex/skills/induts-create-env/scripts/create_env.py`; reads `requirements.txt`; writes setup reports under `setup_reports/`. |
| `induts-characteristics` | Inspect a dataset before benchmark runs or onboarding: stationarity, autocorrelation, missing values, outliers, split drift, target correlations, modes, and plots. | `.codex/skills/induts-characteristics/tools/analyze_characteristics.py`; wraps `data/data_visualization.py`; writes reports under `data/<Dataset>/characteristics report/`. |
| `induts-add-dataset` | Add, inspect, scaffold, validate, or document a new dataset for `SS_task` or `LSF_task`. | `.codex/skills/induts-add-dataset/tools/inspect_dataset.py`; `.codex/skills/induts-add-dataset/tools/scaffold_dataset_yaml.py`; `.codex/skills/induts-add-dataset/tools/smoke_dataset.py`; may update `data/`, `scripts/`, and dataset docs. |
| `induts-smoke` | Smoke-test a new or edited model/YAML before large experiments: config parsing, model registration, CPU forward pass, tiny train/test loop, and prediction shapes. | `.codex/skills/induts-smoke/scripts/smoke_models.py`; writes temporary YAMLs and logs under `smoke_runs/`. |
| `induts-results-best-export` | Analyze `results/` to find the best run per model for each task/dataset, export grouped Excel reports, and collect best checkpoints plus prediction artifacts. | `.codex/skills/induts-results-best-export/scripts/export_best_results.py`; writes grouped reports under `report/<task>/<dataset>/`. |

## Maintenance Rules

- Preserve current directory conventions under `models/`, `data/`, `exp/`, `utils/`, `scripts/`, and `results/`.
- Do not silently change benchmark settings that would invalidate comparisons.
- Treat YAML matrices and shell scripts as public reproducibility artifacts.
- Keep Windows users in mind: prefer YAML/Python commands unless the task explicitly uses shell scripts.
- If a fix changes behavior, update the matching YAML/reference/docs so future runs are explainable.
- For generated scripts or result summaries, prefer existing helper scripts in the matching InduTS skill over one-off rewrites.

## Response Style

- Answer in the user's language when possible.
- Start with the most useful command, file, or diagnosis.
- Give exact paths and runnable commands.
- State expected outputs and where to inspect them.
- When something is uncertain, name the local file to verify next instead of guessing.

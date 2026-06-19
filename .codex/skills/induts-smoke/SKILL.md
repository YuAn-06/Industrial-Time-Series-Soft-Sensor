---
name: induts-smoke
description: Use when a new InduTS-SS model is added, scaffolded, registered, edited, or ported, or when an existing model needs a small smoke test to confirm YAML loading, model registration, CPU forward/training/test execution, and prediction output shape before larger experiments.
---

# InduTS Smoke

## Purpose

Use this skill for small smoke tests of new or existing InduTS-SS models. The goal is to confirm that a model can be registered, loaded from YAML, run through a CPU forward probe, complete a tiny training loop, test a few batches, and emit the expected prediction shape.

## Quick Start

Smoke one YAML:

```bash
conda run -n time_series_v2 python .codex/skills/induts-smoke/scripts/smoke_models.py --yaml scripts/SS_task/DC_scripts/yaml/DLinear.yaml
```

Smoke several YAMLs:

```bash
conda run -n time_series_v2 python .codex/skills/induts-smoke/scripts/smoke_models.py --yaml scripts/SS_task/DC_scripts/yaml/DLinear.yaml scripts/LSF_task/DC_scripts/yaml/DLinear.yaml
```

List the command plan without running training:

```bash
conda run -n time_series_v2 python .codex/skills/induts-smoke/scripts/smoke_models.py --yaml scripts/SS_task/DC_scripts/yaml/DLinear.yaml --dry-run
```

Run a slightly larger smoke:

```bash
conda run -n time_series_v2 python .codex/skills/induts-smoke/scripts/smoke_models.py --yaml scripts/SS_task/DC_scripts/yaml/DLinear.yaml --max-train-batches 5 --max-test-batches 5
```

## Workflow

1. Run smoke after adding a model file, editing `MODEL_REGISTRY`, changing a model forward signature, porting a model, or generating/editing YAMLs.
2. Prefer small public YAMLs such as DC/DLinear first, then smoke the changed model's real YAML.
3. Let the script create temporary YAMLs under `smoke_runs/tmp_yaml/`; do not edit benchmark YAMLs for smoke settings.
4. Confirm the report shows `status: passed`, forward probe shapes, train/test batch counts, and `test batch shapes`.
5. If smoke fails, inspect the generated log path in `smoke_runs/logs/`.

## What The Script Changes In Temporary YAMLs

The original YAML is preserved. The smoke YAML overrides:

- `use_cuda: False`
- `device: cpu`
- `use_multi_gpu: False`
- `use_amp: False`
- `use_tensorboard: False`
- `num_workers: 0`
- `epoch: 1`
- `patience: 1`
- `batch_size: <--batch-size>` unless omitted

By default the runner executes only `--max-train-batches 2` and `--max-test-batches 2`, which is intentional for a fast smoke test.

## Pass Criteria

A smoke run passes only when:

- The YAML parses through `utils.configs.Parse_arguments`.
- The model resolves through `exp.exp_basic.MODEL_REGISTRY`.
- The experiment and dataloaders instantiate on CPU.
- A forward probe produces a tensor selected by the experiment's normal prediction selector.
- The selected prediction shape matches the selected ground-truth shape.
- The tiny CPU smoke training loop and test-batch shape checks complete without exceptions.

## Notes

- This is a correctness smoke test, not a performance or metric benchmark.
- Do not compare smoke metrics across models; smoke settings intentionally change training budget and hardware.
- For very slow models, pass `--dry-run` first and then smoke a smaller YAML or dataset.

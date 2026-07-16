# 🎉 Getting Started with InduTS-SS

This guide is for first-time users and contributors who want to run or extend InduTS-SS. It covers environment setup, data preparation, experiment execution, and integration of new models, losses, and datasets. Unless stated otherwise, run all commands from the repository root.

InduTS-SS (Industrial Time-Series Soft Sensor) is a unified benchmark for dynamic soft sensing on industrial time series. It organizes data loading and preprocessing, model definitions, training, validation, testing, and result storage into a reproducible workflow. It supports two task families:

- `SS_task`: soft-sensor regression, which estimates a currently hard-to-measure quality variable from a historical window of process variables.
- `LSF_task`: time-series forecasting, which predicts quality variables over future time steps from historical process and quality variables.

> For fair model comparisons, keep the data split, scaling policy, seed, sequence length, prediction length, training budget, and evaluation metrics unchanged across models.

## Contents

1. [Install the Environment](#1-install-the-environment)
2. [Prepare Datasets](#2-prepare-datasets)
3. [Run Experiments](#3-run-experiments)
4. [Model Module](#4-model-module)
5. [Loss Functions](#5-loss-functions)
6. [Data Module](#6-data-module)
7. [Experiment Module](#7-experiment-module)

## 1. Install the Environment

### 1.1 Clone the repository

```bash
git clone <repository-url>
cd Industrial-Time-Series-Soft-Sensor
```

Python 3.10 and an isolated conda environment are recommended. The current `requirements.txt` pins `torch==2.7.1+cu128`, which requires an NVIDIA driver compatible with CUDA 12.8. Do not install this exact PyTorch wheel unchanged on a CPU-only machine or a machine with an older driver.

### 1.2 Recommended: detect hardware and create the environment

Start with the read-only hardware check. It inspects the operating system, CPU, memory, conda, GPU, driver, and CUDA support, then writes a report under `setup_reports/`:

```bash
python .codex/skills/induts-create-env/scripts/create_env.py --dry-run
```

After reviewing the proposed installation, create the default `induts-ss` environment and install the dependencies:

```bash
python .codex/skills/induts-create-env/scripts/create_env.py --create
```

To use a custom environment name:

```bash
python .codex/skills/induts-create-env/scripts/create_env.py --env-name induts-ss-cu128 --create
```

### 1.3 Manual installation

If the installed driver supports CUDA 12.8, use the repository-pinned dependencies:

```bash
conda create -n induts-ss python=3.10 -y
conda run -n induts-ss python -m pip install -r requirements.txt
```

Verify PyTorch and CUDA after installation:

```bash
conda run -n induts-ss python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

On a CPU-only machine, install the appropriate CPU build of PyTorch before installing the remaining packages. Do not use the `+cu128` wheel from `requirements.txt`. Set the experiment YAML as follows:

```yaml
use_cuda: false
device: cpu
use_multi_gpu: false
use_amp: false
```

## 2. Prepare Datasets

Datasets are normally stored under `data/<Dataset>/`. The framework currently maps these datasets:

| Name | Default file | Description |
| --- | --- | --- |
| `DC` | `data/DC/debutanizer_column.csv` | Debutanizer column |
| `SRU` | `data/SRU/SRU_data.csv` | Sulfur recovery unit |
| `Ironmaking` | `data/Ironmaking/Ironmaking.csv` | Ironmaking; obtain the source data as described by the project |
| `MP` | `data/MP/MP_data.csv` | Mining process |
| `PPGAS` | `data/PPGAS/gt_2012.csv` | Power-plant gas turbine; `PPGAS2012` is a compatibility alias |

After downloading any missing data, make sure `data_name`, `data_path`, and `target` in the YAML match the actual file. A CSV should satisfy the following contract:

- It can be read with `pandas.read_csv`.
- The column named by `target` exists.
- An optional `date` column enables time features.
- An optional `mode` column enables multimode processing.
- If process-variable columns start with `x_`, the loader prioritizes them as input features.

Inspect a local dataset before running it:

```bash
python .codex/skills/induts-add-dataset/tools/inspect_dataset.py --csv ./data/SRU/SRU_data.csv --target SO2
```

## 3. Run Experiments

### 3.1 Train and test from YAML (recommended)

Experiment configurations are organized as follows:

```text
scripts/SS_task/<Dataset>_scripts/yaml/<Model>.yaml
scripts/LSF_task/<Dataset>_scripts/yaml/<Model>.yaml
```

Set `yaml_name` in `run_with_yaml.py` to a path relative to `scripts/`:

```python
yaml_name = "SS_task/SRU_scripts/yaml/TSLambdaGRU.yaml"
```

Then run:

```bash
python run_with_yaml.py
```

A typical YAML has this structure:

```yaml
params:
  # Data and task
  model: DLinear
  task: soft_sensor
  data_name: SRU
  data_path: ./data/SRU/SRU_data.csv
  target: SO2

  # Input and output
  C_in: 20
  C_out: 1
  seq_len: 4
  label_len: 4
  pred_len: 1

  # Training
  batch_size: 64
  learning_rate: 0.001
  epoch: 300
  patience: 10
  seed: 2021

  # Evaluation and hardware
  inverse: true
  use_cuda: true
  device: cuda
  gpu: 0
  device_ids: [0]
  use_multi_gpu: false
```

Model-specific fields must be declared in `models/<Model>/model_config.py`. Unknown YAML fields cause configuration loading to fail so that misspellings are not silently ignored.

### 3.2 Run Shell scripts

The repository retains `run.py`-based Shell scripts, for example:

```bash
bash scripts/SS_task/SRU_scripts/sh/DLSTM.sh
```

They are suitable for Linux, WSL, or Git Bash and are useful for batch experiments and command-line overrides. Windows PowerShell users should prefer YAML with `python run_with_yaml.py`. `run.py` is a backward-compatible entry point; YAML is recommended for new reproducible experiments.

### 3.3 Test an existing checkpoint only

Use the original training YAML to rebuild the model and data configuration, then load the complete checkpoint. This command does not train. By default, it writes results to an `evaluation/` directory beside the checkpoint without overwriting the original artifacts:

```bash
python -c "from runner import test_from_checkpoint; test_from_checkpoint(r'./scripts/SS_task/SRU_scripts/yaml/TSLambdaGRU.yaml', r'./results/TSLambdaGRU/<setting>/checkpoint.pth')"
```

The checkpoint must match the model structure and relevant hyperparameters in the YAML. Strict weight loading is enabled by default and should only be disabled when partial compatibility is intentional.

### 3.4 Run the complete pretraining and finetuning workflow

Only models that declare stages in `MODEL_SPEC.pretrain_stages` support this workflow. For example, FA-SConvAE-LSTM uses layer-wise pretraining:

```bash
python run_pretrain_finetune.py --yaml ./scripts/SS_task/SRU_scripts/yaml/FASConvAELSTM.yaml --mode full
```

The runner executes the declared stages in order, passes each checkpoint to the next stage, then performs finetuning and testing. To train without the final test:

```bash
python run_pretrain_finetune.py --yaml ./scripts/SS_task/SRU_scripts/yaml/FASConvAELSTM.yaml --mode full --no_test
```

Stage-specific budgets can be overridden in YAML:

```yaml
pretrain_epoch: 200
finetune_epoch: 300
pretrain_learning_rate: 0.01
finetune_learning_rate: 0.001
```

### 3.5 Finetune an existing pretrained checkpoint

Skip pretraining and start from supplied weights:

```bash
python run_pretrain_finetune.py --yaml ./scripts/SS_task/SRU_scripts/yaml/FASConvAELSTM.yaml --mode finetune --checkpoint ./results/FASConvAELSTM/<pretrain-setting>/checkpoint_pretrain_l3.pth
```

Use `--checkpoint_dir <dir>` to select the shared stage-checkpoint archive directory.

### 3.6 Inspect results

Standard outputs are stored under:

```text
results/<Model>/<setting>/
```

Common artifacts include:

- `checkpoint.pth`: weights with the best validation performance;
- `metrics.npy`: test metrics;
- `pred.npy`: predictions;
- `true.npy`: ground truth;
- logs, figures, and optional TensorBoard event files.

The `setting` name is generated from the dataset, task, common training fields, and selected model fields. In dataclass field metadata, `prefix` is the abbreviation used in the setting name, while `order` controls the field's position. For example, a setting may contain `sl4_ll4_pl1_bt64_lr0.001_ep300_pat10`, and the full result path is `results/<Model>/<setting>/`.

When reporting results, record the YAML path, target, `seq_len`, `pred_len`, seed, inverse-scaling choice, and hardware configuration.

## 4. Model Module

### 4.1 Model package structure

Each model lives under `models/<Model>/`. Start from `models/_template/`:

```text
models/<Model>/
├── __init__.py
├── model_arch.py
├── model_config.py
└── model_spec.py
```

| File | Responsibility | Should not contain |
| --- | --- | --- |
| `model_arch.py` | Network architecture and `forward` | Data paths or training budgets |
| `model_config.py` | Model-owned hyperparameters and defaults | Stable framework capabilities |
| `model_spec.py` | Tasks, data representation, loss, and pretraining stages | Tunable learning rates or dimensions |

`__init__.py` provides the public exports:

```python
from .model_arch import Model
from .model_config import MODEL_CONFIG
from .model_spec import MODEL_SPEC

__all__ = ["Model", "MODEL_CONFIG", "MODEL_SPEC"]
```

### 4.2 `model_arch.py`: architecture

The entry class must be named `Model`, and its constructor receives the parsed `configs`. The standard data adapter normally passes `x_enc` and other batch fields to `forward`:

```python
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.projection = nn.Linear(configs.C_in, configs.C_out)

    def forward(self, x_enc: torch.Tensor, **batch):
        del batch
        return self.projection(x_enc[:, -self.pred_len:, :])
```

The conventional output shape is `[batch_size, pred_len, C_out]`. Models with dictionary outputs or specialized losses must follow the prediction-selection contract of the experiment class.

### 4.3 `model_config.py`: hyperparameters

The model config inherits `BaseExpConfig` and adds only model-owned fields:

```python
from dataclasses import dataclass, field
from models.base import BaseExpConfig


@dataclass
class MyModelConfig(BaseExpConfig):
    hidden_dim: int = field(
        default=64,
        metadata={"help": "Hidden dimension.", "prefix": "hd", "order": 20},
    )


MODEL_CONFIG = MyModelConfig
```

Common fields such as data path, task, dimensions, epochs, and device are already provided by `BaseExpConfig`. The configuration class validates YAML fields and basic values. In the field metadata, `prefix` controls the abbreviation included in the generated `setting`, and `order` determines its position.

### 4.4 `model_spec.py`: capability card

`MODEL_SPEC` describes the stable integration contract between the model and framework:

```python
from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    name="MyModel",
    module="models.MyModel",
    supported_tasks=("soft_sensor", "short_term_forecasting"),
    dataset_type="standard",
    loss_type="mse",
    pretrain_stages=(),
    paper_title="",
    paper_url="",
    source_url="",
)
```

- `supported_tasks` lists only tasks genuinely supported by the architecture.
- `dataset_type` commonly uses `standard`, `multimode`, or `lagged_matrix`.
- `loss_type` selects the loss protocol.
- `pretrain_stages` is empty for ordinary models and ordered for staged models.
- Paper metadata is documentation-only and must not affect runtime behavior.

### 4.5 Register and validate a model

1. Copy and complete the template package.
2. Add its canonical name and package path to `MODEL_REGISTRY` in `models/registry.py`.
3. Add YAML only for genuinely supported tasks.
4. Inspect the smoke-test plan:

```bash
python .codex/skills/induts-smoke/scripts/smoke_models.py --yaml scripts/SS_task/DC_scripts/yaml/MyModel.yaml --dry-run
```

5. Run the small CPU smoke test:

```bash
python .codex/skills/induts-smoke/scripts/smoke_models.py --yaml scripts/SS_task/DC_scripts/yaml/MyModel.yaml
```

The smoke runner writes temporary YAMLs under `smoke_runs/tmp_yaml/`, reduces training to a few batches, and disables CUDA without modifying benchmark YAMLs. Smoke metrics are not performance results.

## 5. Loss Functions

### 5.1 Loss selection

Loss functions are defined in `exp/losses.py`. Experiment classes do not select them with model-name conditionals or YAML fields. Selection follows this chain:

```text
MODEL_SPEC.loss_type
        ↓
exp.losses.LOSS_REGISTRY
        ↓
concrete BaseLoss subclass
        ↓
Losses.calculate_loss(outputs, trues, flag)
```

For a standard regression model:

```python
MODEL_SPEC = ModelSpec(
    name="MyModel",
    module="models.MyModel",
    supported_tasks=("soft_sensor",),
    dataset_type="standard",
    loss_type="mse",
)
```

An unregistered name raises an error listing the supported types instead of silently falling back.

### 5.2 Registered loss types

| `loss_type` | Class | Purpose |
| --- | --- | --- |
| `mse` | `MSE_Loss` | Mean squared error for standard regression and forecasting |
| `huber` | `HuberLoss` | Robust regression loss, currently with `delta=0.8` |
| `cvaesmc` | `CVAESMC_Loss` | Reconstruction and KL terms for CVAE-SMC |
| `dmvaer` | `DMVAER_Loss` | Input/target reconstruction and multiple KL terms |
| `vrnn` | `VRNN_Loss` | Input/target reconstruction and KL terms |
| `tcvae` | `TCVAE_Loss` | Prediction and KL terms |
| `gtfts` | `GTFTS_Loss` | Prediction and MRMC regularization terms |
| `stdtaem` | `STDTAEm_Loss` | Pretraining reconstruction/triplet loss and finetuning regression loss |
| `fasconvaelstm` | `FASConvAELSTM_Loss` | Layer-wise reconstruction loss and finetuning MSE |

Models that return a single prediction tensor should normally reuse `mse` or `huber`. Add a dedicated loss only for dictionary outputs, reconstruction branches, latent variables, or stage-dependent objectives.

### 5.3 `BaseLoss` and logging

All loss classes inherit `BaseLoss`. Its `_loss_lists` stores epoch-level total and component losses for consistent logging:

```python
import numpy as np
from torch import nn


class MyModelLoss(BaseLoss):
    def __init__(self, args):
        super().__init__(args)
        self.mse = nn.MSELoss(reduction="mean")
        self._register_loss_list("loss")
        self._register_loss_list("aux_loss")

    def forward(self, preds, trues, flag="train"):
        pred_loss = self.mse(preds["y_pred"], trues)
        aux_loss = preds["aux_loss"]
        loss = pred_loss + self.args.aux_weight * aux_loss
        if flag == "train":
            self._append_loss("loss", loss.item())
            self._append_loss("aux_loss", aux_loss.item())
        return loss

    def print_loss_details(self):
        aux = np.mean(self._loss_lists["aux_loss"])
        print(f"Aux Loss: {aux:.4f}")
```

Every loss must register a total list named `loss`, because common logging reads it through `mean_total_loss`. Record list values only during training; validation should return the scalar loss without adding entries. The framework clears the lists after each epoch log.

### 5.4 Add a custom loss

1. Add a `BaseLoss` subclass to `exp/losses.py`.
2. Implement `forward(preds, trues, flag="train")` and return a differentiable scalar tensor.
3. Register `loss` and any component lists; optionally implement `print_loss_details()`.
4. Add the class to `LOSS_REGISTRY`:

```python
LOSS_REGISTRY = {
    # Existing losses...
    "mymodel": MyModelLoss,
}
```

5. Set `loss_type="mymodel"` in `models/MyModel/model_spec.py`.
6. Put tunable loss weights in `model_config.py` and YAML, not in `MODEL_SPEC`.
7. Run a CPU smoke test and confirm finite training/validation losses and matching prediction/target shapes.

### 5.5 Output and target contract

- Ordinary models return a prediction tensor and use `MSE_Loss` or `HuberLoss`.
- Multitask, generative, or pretrained models may return dictionaries, but their keys must exactly match those consumed by the specialized loss.
- Ground truth may be a target tensor or a dictionary containing values such as `x_true`, `y_true`, and `c_true`.
- The returned loss must be a scalar Tensor. Do not call `.item()` on the returned value, because that breaks gradient propagation.
- Validate batch, time, and output-channel dimensions instead of relying on implicit broadcasting.

The loss controls optimization, whereas test metrics evaluate final predictions. Changing loss terms or weights changes the training objective and must be documented in the YAML while the evaluation protocol remains consistent.

## 6. Data Module

### 6.1 Dataset classes

Core implementations live in `data/data_loader.py`. `data/data_provider.py` selects a class from the task and model-required representation:

| Task | `dataset_type` | Dataset class |
| --- | --- | --- |
| `short_term_forecasting` | `standard` | `Dataset_Custom` |
| `soft_sensor` | `standard` | `Dataset_Custom_4_Soft_Sensor` |
| `short_term_forecasting` | `multimode` | `Dataset_MultiMode` |
| `soft_sensor` | `multimode` | `Dataset_MultiMode_4_Soft_Sensor` |
| `soft_sensor` | `lagged_matrix` | `Dataset_LaggedMatrix_4_Soft_Sensor` |

When `use_condition_label: true`, multimode representation takes precedence. Otherwise, the framework reads `MODEL_SPEC.dataset_type`.

### 6.2 Preprocessing pipeline

1. Read `data_path` with `pandas.read_csv`.
2. Validate the target and determine process-variable columns.
3. Split chronologically into 70% training, 10% validation, and 20% testing.
4. Fit scalers on the training interval only, then transform validation and test data.
5. Build sliding windows from `seq_len`, `label_len`, and `pred_len`.
6. Return batch fields such as `x_enc`, `x_dec`, time marks, and `batch_y`.
7. When `inverse: true`, restore test results to the original scale.

Soft-sensor targets normally represent the current quality value and have shape `[B, 1, C_out]`. Forecasting targets cover a future window and normally have shape `[B, pred_len, C_out]`.

### 6.3 Add a dataset

Reuse existing dataset classes whenever possible. Modify core loading code only for genuinely special splitting, missing-value, multimode, or preprocessing behavior.

1. Place the CSV at `data/NewDataset/data.csv`.
2. Inspect it:

```bash
python .codex/skills/induts-add-dataset/tools/inspect_dataset.py --csv ./data/NewDataset/data.csv --target Y
```

3. Add the canonical name to `SUPPORTED_DATASETS` in `data/data_provider.py`, or add a name alias to `DATASET_ALIASES`.
4. Generate minimal YAMLs:

```bash
python .codex/skills/induts-add-dataset/tools/scaffold_dataset_yaml.py --dataset NewDataset --csv ./data/NewDataset/data.csv --target Y --task soft_sensor --model DLinear
python .codex/skills/induts-add-dataset/tools/scaffold_dataset_yaml.py --dataset NewDataset --csv ./data/NewDataset/data.csv --target Y --task short_term_forecasting --model DLinear
```

5. Inspect the smoke plan, then remove `--dry-run` when ready:

```bash
python .codex/skills/induts-add-dataset/tools/smoke_dataset.py --yaml ./scripts/SS_task/NewDataset_scripts/yaml/DLinear.yaml --dry-run
```

Do not silently change the 70/10/20 split, scaling, seeds, or metric definitions. Document any intentionally different protocol in the loader, YAML, and user documentation.

## 7. Experiment Module

### 7.1 End-to-end flow

```text
YAML / run.py
    ↓
utils.configs: load and validate MODEL_CONFIG; build setting and save_dir
    ↓
models.registry: resolve the model package, MODEL_SPEC, and MODEL_CONFIG
    ↓
exp.exp_factory: construct the Experiment selected by task
    ↓
data.data_provider: construct Dataset and DataLoader from task + dataset_type
    ↓
Experiment.train / vali / test
    ↓
results/<Model>/<setting>/
```

### 7.2 Experiment classes

`exp/exp_factory.py` selects the class from the YAML `task`:

- `soft_sensor` → `Exp_Soft_Sensor`;
- `short_term_forecasting` → `Exp_Short_Term_Forecasting`.

Both classes build the model and loaders, select a loss from `MODEL_SPEC.loss_type`, optimize with early stopping, load the best checkpoint, compute test metrics, and save predictions and targets.

### 7.3 Training, validation, and testing

- Each training batch performs forward propagation, loss computation, backpropagation, and optimization.
- Validation loss is computed after every epoch; early stopping saves the current best model as `checkpoint.pth`.
- The best weights are reloaded after training instead of using the final epoch directly.
- Testing saves `metrics.npy`, `pred.npy`, and `true.npy` and can write TensorBoard data.

Metrics include MAE, MSE, RMSE, MAPE, MSPE, WAPE, correlation, and R² for soft-sensor tasks. When `inverse: true`, state that reported metrics were computed on the original scale.

### 7.4 Extending experiment logic

If a new model changes only its architecture or hyperparameters, integrate it through its model package and `MODEL_SPEC`; do not add an Experiment class. Modify `exp/` only when the existing protocol cannot express the model's inputs or outputs, it requires special optimizer groups or multi-objective training, or testing requires different prediction selection.

After changing experiment behavior, run at least one CPU smoke test and verify that prediction and target shapes match. Any change to splitting, scaling, training budgets, or metrics must also be reflected in YAML and documentation.

## Next Steps

- See [`readme.md`](../readme.md) for the project overview, model catalog, and dataset descriptions.
- Start new model integrations from [`models/_template/`](../models/_template/).
- Browse benchmark configurations under [`scripts/SS_task/`](../scripts/SS_task/) and [`scripts/LSF_task/`](../scripts/LSF_task/).
- Agent-assisted workflows are available under [`.codex/skills/`](../.codex/skills/).

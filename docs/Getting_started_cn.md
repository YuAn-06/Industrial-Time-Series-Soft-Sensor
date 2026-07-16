# 🎉 InduTS-SS 快速上手

本文面向第一次使用或希望扩展 InduTS-SS 的用户，介绍从环境安装、数据准备、实验运行，到新增模型和数据集的完整流程。所有命令默认在仓库根目录执行。

InduTS-SS（Industrial Time-Series Soft Sensor）是一个面向工业时间序列动态软测量的统一基准框架。它将数据读取与预处理、模型定义、训练、验证、测试和结果保存组织为可复现的流水线，并支持两类任务：

- `SS_task`：软测量回归。使用一段历史过程变量估计当前难以直接测量的质量变量。
- `LSF_task`：时间序列预测。使用历史过程变量和质量变量预测未来多个时间步的质量变量。

> 公平比较模型时，请保持数据划分、归一化方式、随机种子、序列长度、预测长度、训练轮数和评价指标一致，不要只修改某一个模型的实验预算。

## 目录

1. [安装依赖环境](#1-安装依赖环境)
2. [准备数据集](#2-准备数据集)
3. [快速运行实验](#3-快速运行实验)
4. [详细教程：模型模块](#4-详细教程模型模块)
5. [详细教程：损失函数](#5-详细教程损失函数)
6. [详细教程：数据模块](#6-详细教程数据模块)
7. [详细教程：实验模块](#7-详细教程实验模块)

## 1. 安装依赖环境

### 1.1 克隆仓库

```bash
git clone <repository-url>
cd Industrial-Time-Series-Soft-Sensor
```

建议使用 Python 3.10 和独立的 conda 环境。仓库当前的 `requirements.txt` 固定使用 `torch==2.7.1+cu128`，适用于支持 CUDA 12.8 的 NVIDIA 驱动。CPU、旧驱动或其他 CUDA 版本的机器不应直接照搬这一 PyTorch wheel。

### 1.2 推荐：自动检测硬件并创建环境

先执行只读检测，脚本会检查操作系统、CPU、内存、conda、GPU、驱动和 CUDA，并在 `setup_reports/` 下生成报告：

```bash
python .codex/skills/induts-create-env/scripts/create_env.py --dry-run
```

确认报告给出的安装方案后，创建默认的 `induts-ss` 环境并安装依赖：

```bash
python .codex/skills/induts-create-env/scripts/create_env.py --create
```

也可以指定环境名称：

```bash
python .codex/skills/induts-create-env/scripts/create_env.py --env-name induts-ss-cu128 --create
```

### 1.3 手动安装

如果本机驱动支持 CUDA 12.8，可以使用仓库的固定依赖：

```bash
conda create -n induts-ss python=3.10 -y
conda run -n induts-ss python -m pip install -r requirements.txt
```

安装后验证 PyTorch 和 CUDA：

```bash
conda run -n induts-ss python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果没有 NVIDIA GPU，请先安装与本机匹配的 CPU 版 PyTorch，再安装其余依赖；不要直接安装 `requirements.txt` 中固定的 `+cu128` wheel。GPU 不可用时，也需要把实验 YAML 设置为：

```yaml
use_cuda: false
device: cpu
use_multi_gpu: false
use_amp: false
```

## 2. 准备数据集

仓库中的数据通常位于 `data/<Dataset>/`。目前框架内置映射的数据集包括：

| 名称 | 默认文件 | 说明 |
| --- | --- | --- |
| `DC` | `data/DC/debutanizer_column.csv` | 脱丁烷塔数据集 |
| `SRU` | `data/SRU/SRU_data.csv` | 硫磺回收装置数据集 |
| `Ironmaking` | `data/Ironmaking/Ironmaking.csv` | 炼铁数据集，需按项目说明从原始来源获取 |
| `MP` | `data/MP/MP_data.csv` | 采矿过程数据集 |
| `PPGAS` | `data/PPGAS/gt_2012.csv` | 电厂燃气轮机数据集；`PPGAS2012` 是兼容别名 |

下载缺失数据后，请保持 YAML 中的 `data_name`、`data_path` 和 `target` 与实际文件一致。CSV 至少应满足：

- 可以被 `pandas.read_csv` 读取；
- `target` 指定的质量变量真实存在；
- 可选的 `date` 列用于构造时间特征；
- 可选的 `mode` 列用于多工况数据；
- 过程变量以 `x_` 开头时，加载器会优先将这些列识别为输入特征。

可以先检查本地数据文件是否完整，例如：

```bash
python .codex/skills/induts-add-dataset/tools/inspect_dataset.py --csv ./data/SRU/SRU_data.csv --target SO2
```

## 3. 快速运行实验

### 3.1 使用 YAML 完成训练和测试（推荐）

实验配置位于：

```text
scripts/SS_task/<Dataset>_scripts/yaml/<Model>.yaml
scripts/LSF_task/<Dataset>_scripts/yaml/<Model>.yaml
```

打开 `run_with_yaml.py`，将 `yaml_name` 修改为相对于 `scripts/` 的路径：

```python
yaml_name = "SS_task/SRU_scripts/yaml/TSLambdaGRU.yaml"
```

然后运行：

```bash
python run_with_yaml.py
```

一份典型 YAML 的结构如下：

```yaml
params:
  # 数据与任务
  model: DLinear
  task: soft_sensor
  data_name: SRU
  data_path: ./data/SRU/SRU_data.csv
  target: SO2

  # 输入与输出
  C_in: 20
  C_out: 1
  seq_len: 4
  label_len: 4
  pred_len: 1

  # 训练
  batch_size: 64
  learning_rate: 0.001
  epoch: 300
  patience: 10
  seed: 2021

  # 测试与硬件
  inverse: true
  use_cuda: true
  device: cuda
  gpu: 0
  device_ids: [0]
  use_multi_gpu: false
```

其中，模型专属字段必须在 `models/<Model>/model_config.py` 中声明。YAML 出现未声明字段时，配置加载会直接报错，以避免拼写错误被静默忽略。

### 3.2 使用 Shell 脚本批量运行

仓库保留了基于 `run.py` 的 Shell 实验脚本，例如：

```bash
bash scripts/SS_task/SRU_scripts/sh/DLSTM.sh
```

这类脚本适合 Linux、WSL 或 Git Bash，可用于批量实验和命令行参数覆盖。Windows PowerShell 用户建议优先使用 YAML 与 `python run_with_yaml.py`。`run.py` 是向后兼容入口，新实验更推荐用 YAML 管理配置，便于复现。

### 3.3 只测试已有 checkpoint

只测试时，应使用训练时相同的 YAML 重建模型和数据配置，再加载完整 checkpoint。下面的命令不会训练模型；结果默认写到 checkpoint 同级的 `evaluation/`，不会覆盖原训练结果：

```bash
python -c "from runner import test_from_checkpoint; test_from_checkpoint(r'./scripts/SS_task/SRU_scripts/yaml/TSLambdaGRU.yaml', r'./results/TSLambdaGRU/<setting>/checkpoint.pth')"
```

checkpoint 必须与 YAML 的模型结构和关键超参数一致。默认采用严格权重加载；除非明确知道权重可部分兼容，否则不建议关闭 `strict`。

### 3.4 完整预训练与微调

只有在 `MODEL_SPEC.pretrain_stages` 中声明了预训练阶段的模型，才能使用该流程。例如 FA-SConvAE-LSTM 包含逐层预训练阶段，完整运行命令为：

```bash
python run_pretrain_finetune.py --yaml ./scripts/SS_task/SRU_scripts/yaml/FASConvAELSTM.yaml --mode full
```

运行器会依照模型声明依次执行预训练阶段，将 checkpoint 传递给下一阶段，最后完成微调和测试。若只需要训练、不执行最终测试，可添加：

```bash
python run_pretrain_finetune.py --yaml ./scripts/SS_task/SRU_scripts/yaml/FASConvAELSTM.yaml --mode full --no_test
```

YAML 可以使用以下字段分别覆盖不同阶段的训练预算：

```yaml
pretrain_epoch: 200
finetune_epoch: 300
pretrain_learning_rate: 0.01
finetune_learning_rate: 0.001
```

### 3.5 只微调已有预训练 checkpoint

跳过预训练并从指定权重开始微调：

```bash
python run_pretrain_finetune.py --yaml ./scripts/SS_task/SRU_scripts/yaml/FASConvAELSTM.yaml --mode finetune --checkpoint ./results/FASConvAELSTM/<pretrain-setting>/checkpoint_pretrain_l3.pth
```

也可以通过 `--checkpoint_dir <dir>` 指定各阶段 checkpoint 的归档目录。

### 3.6 查看结果

标准实验输出保存在：

```text
results/<Model>/<setting>/
```

常见文件包括：

- `checkpoint.pth`：验证集表现最佳的模型权重；
- `metrics.npy`：测试指标；
- `pred.npy`：模型预测；
- `true.npy`：真实值；
- 日志、结果图和可选的 TensorBoard 事件文件。

`setting` 由数据集、任务、通用训练字段和模型关键字段自动组成。比较实验时应记录 YAML 路径、目标变量、`seq_len`、`pred_len`、随机种子、是否反归一化以及硬件配置。

## 4. 详细教程：模型模块

### 4.1 模型包的组成

每个模型位于 `models/<Model>/`，推荐从 `models/_template/` 复制。一个完整模型包包含四个文件：

```text
models/<Model>/
├── __init__.py
├── model_arch.py
├── model_config.py
└── model_spec.py
```

其中三个核心文件各自负责不同职责：

| 文件 | 作用 | 不应包含 |
| --- | --- | --- |
| `model_arch.py` | 定义网络结构和 `forward` | 数据路径、训练轮数等实验配置 |
| `model_config.py` | 声明模型专属超参数及默认值 | 稳定的框架能力判断 |
| `model_spec.py` | 声明任务、数据表示、Loss 和预训练阶段 | 学习率、隐藏维度等可调参数 |

`__init__.py` 负责统一导出：

```python
from .model_arch import Model
from .model_config import MODEL_CONFIG
from .model_spec import MODEL_SPEC

__all__ = ["Model", "MODEL_CONFIG", "MODEL_SPEC"]
```

### 4.2 `model_arch.py`：模型结构

模型入口类必须命名为 `Model`，构造函数接收已经解析的 `configs`。标准数据适配器通常向前向传播传入 `x_enc` 以及其他批次字段：

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

默认输出约定为 `[batch_size, pred_len, C_out]`。某些论文模型使用字典输出或专用 Loss，应参考相近模型，并确保实验类的预测选择逻辑能够提取最终预测张量。

### 4.3 `model_config.py`：模型超参数

模型配置继承 `BaseExpConfig`，只新增该模型真正拥有的参数：

```python
from dataclasses import dataclass, field

from models.base import BaseExpConfig


@dataclass
class MyModelConfig(BaseExpConfig):
    hidden_dim: int = field(
        default=64,
        metadata={"help": "Hidden dimension."},
    )


MODEL_CONFIG = MyModelConfig
```

数据路径、任务、输入输出维度、训练轮数、设备等公共字段已由 `BaseExpConfig` 提供，不需要在每个模型中重复声明。配置类会校验 YAML 字段和基本取值。

### 4.4 `model_spec.py`：模型能力名片

`MODEL_SPEC` 描述模型与框架之间稳定的集成契约：

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

主要字段含义：

- `supported_tasks`：模型真正支持的任务，不要为了生成更多 YAML 而虚报能力；
- `dataset_type`：输入表示，当前常用值为 `standard`、`multimode`、`lagged_matrix`；
- `loss_type`：实验模块选择的 Loss 协议；
- `pretrain_stages`：无预训练时为空元组；有预训练时按顺序声明，如 `("pretrain",)` 或多阶段列表；
- 论文信息字段只用于文档，不应影响运行逻辑。

### 4.5 注册并验证新模型

1. 复制模板并完成上述文件。
2. 在 `models/registry.py` 的 `MODEL_REGISTRY` 中添加规范名称和包路径。
3. 为模型真正支持的任务创建 YAML，例如 `scripts/SS_task/DC_scripts/yaml/MyModel.yaml`。
4. 先执行不训练的命令检查：

```bash
python .codex/skills/induts-smoke/scripts/smoke_models.py --yaml scripts/SS_task/DC_scripts/yaml/MyModel.yaml --dry-run
```

5. 再运行小型 CPU 冒烟测试：

```bash
python .codex/skills/induts-smoke/scripts/smoke_models.py --yaml scripts/SS_task/DC_scripts/yaml/MyModel.yaml
```

冒烟测试会复制临时 YAML 到 `smoke_runs/tmp_yaml/`，将训练缩短为少量批次并关闭 CUDA，不会修改正式 YAML。通过标准包括配置加载、模型注册、数据加载、前向形状、小型训练和测试全部成功。它只验证正确性，冒烟指标不能用于模型性能比较。

## 5. 详细教程：损失函数

### 5.1 损失函数的选择机制

所有损失函数集中定义在 `exp/losses.py`。实验类不会根据模型名称编写分支，也不直接从 YAML 读取损失函数名称，而是通过模型的 `MODEL_SPEC.loss_type` 选择损失：

```text
MODEL_SPEC.loss_type
        ↓
exp.losses.LOSS_REGISTRY
        ↓
具体的 BaseLoss 子类
        ↓
Losses.calculate_loss(outputs, trues, flag)
```

例如，一个使用均方误差的普通回归模型应在 `model_spec.py` 中声明：

```python
MODEL_SPEC = ModelSpec(
    name="MyModel",
    module="models.MyModel",
    supported_tasks=("soft_sensor",),
    dataset_type="standard",
    loss_type="mse",
)
```

运行时，`Losses` 会用 `loss_type="mse"` 从 `LOSS_REGISTRY` 中取得 `MSE_Loss`。如果名称未注册，框架会直接列出支持的类型并报错，避免静默回退到错误的损失函数。

### 5.2 当前支持的损失类型

当前 `LOSS_REGISTRY` 包含以下类型：

| `loss_type` | 损失类 | 主要用途 |
| --- | --- | --- |
| `mse` | `MSE_Loss` | 标准回归和预测模型的均方误差 |
| `huber` | `HuberLoss` | 对异常误差更稳健的回归损失，当前 `delta=0.8` |
| `cvaesmc` | `CVAESMC_Loss` | CVAE-SMC 的重构损失与 KL 散度 |
| `dmvaer` | `DMVAER_Loss` | DMVAER 的输入/目标重构和多项 KL 损失 |
| `vrnn` | `VRNN_Loss` | VRNN 的输入重构、目标重构和 KL 损失 |
| `tcvae` | `TCVAE_Loss` | TCVAE 的预测损失和 KL 损失 |
| `gtfts` | `GTFTS_Loss` | GTFTS 的预测损失和 MRMC 正则项 |
| `stdtaem` | `STDTAEm_Loss` | STDTAEm 预训练阶段的重构/三元组损失及微调阶段的回归损失 |
| `fasconvaelstm` | `FASConvAELSTM_Loss` | FA-SConvAE-LSTM 逐层预训练重构损失及微调 MSE |

大多数输出单个预测张量的模型应复用 `mse` 或 `huber`。只有模型输出字典、包含重构分支、潜变量或分阶段训练目标时，才需要新增专用损失。

### 5.3 `BaseLoss` 与日志记录

所有损失类应继承 `BaseLoss`。基类使用 `_loss_lists` 保存一个 epoch 内各损失分量，以便统一输出训练总损失和模型专属细节：

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

每个损失必须注册名为 `loss` 的总损失列表，因为统一日志通过 `mean_total_loss` 读取它。只在 `flag == "train"` 时记录列表，验证阶段只返回标量 Loss；每个 epoch 输出日志后，框架会自动清空这些列表。

### 5.4 新增自定义损失函数

接入新的专用损失需要完成以下步骤：

1. 在 `exp/losses.py` 中新增一个继承 `BaseLoss` 的类。
2. 实现 `forward(preds, trues, flag="train")`，最终返回可反向传播的标量张量。
3. 注册总损失 `loss`，需要展示分项时再注册其他列表并实现 `print_loss_details()`。
4. 将该类加入 `LOSS_REGISTRY`：

```python
LOSS_REGISTRY = {
    # 已有损失……
    "mymodel": MyModelLoss,
}
```

5. 在 `models/MyModel/model_spec.py` 中设置：

```python
loss_type="mymodel"
```

6. 如果损失包含可调权重，在 `models/MyModel/model_config.py` 中声明字段，再通过 YAML 配置，不要将实验超参数写入 `MODEL_SPEC`。
7. 运行 CPU 冒烟测试，确认训练和验证均能计算有限标量，且预测张量与真实值形状一致。

### 5.5 输出与真实值契约

损失函数的输入必须与模型输出和实验类生成的真实值相互匹配：

- 普通模型通常输出预测张量，直接使用 `MSE_Loss` 或 `HuberLoss`；
- 多任务、生成式或预训练模型可以输出字典，但字典键必须与专用 Loss 中访问的键完全一致；
- 真实值可能是目标张量，也可能是包含 `x_true`、`y_true`、`c_true` 等内容的字典；
- Loss 返回值必须是标量 Tensor，不能在返回前调用 `.item()`，否则会切断梯度；
- 计算前应检查预测与真实值的 batch、时间步和输出通道是否一致，不要依赖隐式广播掩盖形状错误。

损失函数决定模型如何优化，测试指标则用于评价最终预测，两者不能混用。修改 Loss 权重或组成会改变训练目标，因此在公平比较中应记录对应 YAML，并保持不同模型的评价指标和测试协议一致。

## 6. 详细教程：数据模块

### 6.1 DataLoader 类型

核心实现位于 `data/data_loader.py`，`data/data_provider.py` 根据“任务类型 + 模型所需输入表示”选择具体类：

| 任务 | `dataset_type` | DataLoader 类 |
| --- | --- | --- |
| `short_term_forecasting` | `standard` | `Dataset_Custom` |
| `soft_sensor` | `standard` | `Dataset_Custom_4_Soft_Sensor` |
| `short_term_forecasting` | `multimode` | `Dataset_MultiMode` |
| `soft_sensor` | `multimode` | `Dataset_MultiMode_4_Soft_Sensor` |
| `soft_sensor` | `lagged_matrix` | `Dataset_LaggedMatrix_4_Soft_Sensor` |

若 YAML 设置 `use_condition_label: true`，框架会优先使用多工况表示；否则读取模型的 `MODEL_SPEC.dataset_type`。

### 6.2 数据预处理流程

标准流程如下：

1. 使用 `pandas.read_csv` 读取 `data_path`；
2. 检查目标列并确定过程变量；
3. 按时间顺序划分 70% 训练集、10% 验证集和 20% 测试集；
4. 只使用训练区间拟合 scaler，再转换验证集和测试集，避免数据泄漏；
5. 依据 `seq_len`、`label_len` 和 `pred_len` 构造滑动窗口；
6. 返回模型需要的批次字典，如 `x_enc`、`x_dec`、时间标记和 `batch_y`；
7. 测试阶段在 `inverse: true` 时将结果恢复到原始量纲。

软测量任务的目标通常是当前时刻的质量变量，`batch_y` 形状通常为 `[B, 1, C_out]`；预测任务则输出未来窗口，形状通常为 `[B, pred_len, C_out]`。

### 6.3 接入新数据集

优先复用现有 DataLoader，只有数据具有特殊划分、缺失值、多工况或预处理规则时才修改核心代码。

1. 将文件放到标准目录：

```text
data/NewDataset/data.csv
```

2. 检查 CSV 和目标变量：

```bash
python .codex/skills/induts-add-dataset/tools/inspect_dataset.py --csv ./data/NewDataset/data.csv --target Y
```

3. 在 `data/data_provider.py` 的 `SUPPORTED_DATASETS` 中加入规范名称。若只是名称别名，可加入 `DATASET_ALIASES`。
4. 为软测量或预测任务生成最小 YAML：

```bash
python .codex/skills/induts-add-dataset/tools/scaffold_dataset_yaml.py --dataset NewDataset --csv ./data/NewDataset/data.csv --target Y --task soft_sensor --model DLinear
python .codex/skills/induts-add-dataset/tools/scaffold_dataset_yaml.py --dataset NewDataset --csv ./data/NewDataset/data.csv --target Y --task short_term_forecasting --model DLinear
```

5. 先检查运行计划：

```bash
python .codex/skills/induts-add-dataset/tools/smoke_dataset.py --yaml ./scripts/SS_task/NewDataset_scripts/yaml/DLinear.yaml --dry-run
```

6. 环境就绪后去掉 `--dry-run`，执行实际冒烟测试。

接入数据集时不要静默改变 70/10/20 划分、scaler、随机种子或指标定义。若确实需要特殊协议，应在 DataLoader、YAML 和文档中同时明确说明。

## 7. 详细教程：实验模块

### 7.1 全流程概览

一次 YAML 实验的调用链为：

```text
YAML / run.py
    ↓
utils.configs：加载并校验 MODEL_CONFIG，生成 setting 和 save_dir
    ↓
models.registry：解析模型包、MODEL_SPEC 和 MODEL_CONFIG
    ↓
exp.exp_factory：按 task 创建对应 Experiment
    ↓
data.data_provider：按 task + dataset_type 创建数据集和 DataLoader
    ↓
Experiment.train / vali / test
    ↓
results/<Model>/<setting>/
```

### 7.2 两类 Experiment

`exp/exp_factory.py` 根据 YAML 的 `task` 选择实验类：

- `soft_sensor` → `Exp_Soft_Sensor`；
- `short_term_forecasting` → `Exp_Short_Term_Forecasting`。

两类实验都负责：

1. 构建模型并移动到 CPU 或 GPU；
2. 创建训练、验证和测试 DataLoader；
3. 根据 `MODEL_SPEC.loss_type` 使用相应 Loss；
4. 执行优化、学习率调整和早停；
5. 加载最佳 checkpoint；
6. 在测试集计算指标并保存预测和真实值。

### 7.3 训练、验证与测试

- 训练阶段按 batch 完成前向传播、Loss 计算、反向传播和参数更新。
- 每个 epoch 后在验证集计算 Loss；早停器将当前最佳模型保存为 `checkpoint.pth`。
- 训练结束后重新加载最佳权重，而不是直接使用最后一个 epoch。
- 测试阶段保存 `metrics.npy`、`pred.npy` 和 `true.npy`，并可写入 TensorBoard。

常用指标包括 MAE、MSE、RMSE、MAPE、MSPE、WAPE、相关系数，以及软测量任务使用的 R²。使用 `inverse: true` 时，报告中应注明指标是在原始量纲下计算的。

### 7.4 扩展实验逻辑时的原则

如果新模型只改变网络结构或超参数，应优先在模型包和 `MODEL_SPEC` 中完成接入，不要新建实验类。只有在以下情况才考虑修改 `exp/`：

- 模型需要框架现有协议无法表达的输入或输出；
- 训练阶段包含特殊的优化器分组或多目标 Loss；
- 测试时需要不同的预测选择或额外评估流程。

修改实验流程后，应至少运行一个 CPU 冒烟测试，并核对预测与真实值形状一致。若改变划分、归一化、训练预算或指标，必须同步更新 YAML 和文档，确保结果仍然可解释、可复现。

## 下一步

- 项目概览、模型列表和数据集说明见 [`readme_cn.md`](../readme_cn.md)。
- 新模型从 [`models/_template/`](../models/_template/) 开始。
- 标准实验配置见 [`scripts/SS_task/`](../scripts/SS_task/) 和 [`scripts/LSF_task/`](../scripts/LSF_task/)。
- Agent 辅助工作流位于 [`.codex/skills/`](../.codex/skills/)。

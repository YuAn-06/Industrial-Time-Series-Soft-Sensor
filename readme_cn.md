<div align="center">
  <img src="docs/Logo.jpg#gh-light-mode-only" height="200">
  <h3><b>面向工业时间序列软测量的统一、公平的工具库与基准</b></h3>
</div>

<p align="center">
  <a href="https://www.python.org/">
    <img alt="Python 版本" src="https://img.shields.io/badge/Python-v3.10+-E97040?logo=python&logoColor=white">
  </a>
  <a href="https://pytorch.org/">
    <img alt="由 PyTorch 驱动" src="https://img.shields.io/badge/PyTorch-v2.7+-E97040?logo=pytorch&logoColor=white">
  </a>
  <a href="https://www.apache.org/licenses/">
    <img alt="Apache 许可证" src="https://img.shields.io/badge/License-Apache2.0-E9BB41?logo=opensourceinitiative&logoColor=white">
  </a>
</p>

---

**工业时间序列软测量（Industrial Time-Series Soft Sensor，InduTS-SS）** 是一个开源工具库，面向从事复杂工业过程时间序列数据分析和软测量建模的研究人员。

InduTS-SS 为工业软测量提供统一框架，涵盖数据集加载、预处理、模型训练、评估和基准测试，旨在让模型比较更加公平、可复现且便捷。

我们诚挚欢迎使用者体验本项目，并通过 [GitHub Issue](https://github.com/YuAn-06/Industrial-Time-Series-Soft-Sensor/issues) 提出问题、报告错误或分享建议。如果您希望将自己的模型加入 InduTS-SS，也欢迎通过 Issue 联系我们，我们很乐意协助您将模型集成到本工具库中。

## 目录

- [最新动态](#whats-new)
- [项目动机](#motivation)
- [快速开始](#getting-started)
- [使用 TensorBoard](#using-tensorboard)
- [什么是软测量？](#what-is-a-soft-sensor)
- [数据驱动的软测量建模](#data-driven-soft-sensor-modeling)
- [可用的软测量模型](#available-models-for-soft-sensors)
- [可用数据集](#available-datasets)
- [Codex 与 Claude Code 技能](#codex-and-claude-code-skills)
- [项目结构](#project-structure)
- [安装](#installation)
- [引用](#citation)
- [致谢](#acknowledgements)
- [许可证](#license)

<a id="whats-new"></a>

## 📢 最新动态

**更新 - 2026/7/23：** InduTS-SS 新增了两个模型：基于 GNN 的 **DAMPNN**，通过动态自适应消息传递实现工业软测量；以及面向工业时间序列建模、基于 CNN 的 **PETC_TNet**。

**更新 - 2026/7/16（v1.1.0）：** InduTS-SS 已从 **v1.0.0** 升级至 **v1.1.0**，并完成模型架构的模块化重构。每个模型现已拆分为独立的自包含包，分别管理模型实现（`model_arch.py`）、模型配置（`model_config.py`）和基准能力声明（`model_spec.py`）。新增的统一模型注册表和可复用实验运行器负责模型发现、配置校验、标准训练与测试、检查点评估，以及预训练/微调流程。此次升级使模型接入和维护更加清晰，同时保持任务定义、数据集划分、评价指标及公平性设置不变。

**更新 - 2026/6/21：** 现已内置支持 **Codex**、**Claude Code** 技能等 **Agent 编程**工作流。这些技能可帮助 Agent 更好地理解基准任务，也能帮助用户更轻松地上手。功能包括环境创建、数据集接入、模型集成、冒烟测试，以及模型性能全流程报告。详情请参阅下文相关章节，并通过 Git 克隆 `codex` 分支。

**更新 - 2026/6/15：** InduTS-SS 新增 **Mining Process（MP）** 数据集。衷心感谢 **Eduardo Magalhaes Oliveira 先生**发布这一宝贵的真实工业数据集并确认其真实性。

**更新 - 2026/6/06：** InduTS-SS 现已支持模型预训练和微调。我们提供 `run_pretrain_finetune.py` 以执行预训练模型的完整训练流程。同时新增了一个基于 GNN 的模型和一个基于 AE 的模型，但其超参数尚未精确确定。

**更新 - 2026/5/25：** InduTS-SS 首个版本已经发布。基于该基准框架的论文目前正在投稿中。欢迎试用本工具库、提交问题并贡献改进。

<a id="motivation"></a>

## ✨ 项目动机

现代机器学习工具库为时间序列建模提供了强大工具，但它们并不总能满足工业过程软测量开发的特定需求。研究人员往往需要花费大量精力从头实现预处理流水线、特征提取方法和评估协议，这会造成代码库碎片化，也让公平的模型比较变得困难。此外，许多已发表的软测量方法并未提供开源实现，给复现与基准测试带来挑战。

为弥补这一缺口，我们推出了开源工具库 **InduTS-SS**，为多变量工业时间序列数据上的软测量构建、训练和评估提供一致、模块化的框架。通过标准化从输入表示到预测范围等常用组件，InduTS-SS 旨在加速研究、提高可比性，并降低该领域新贡献者的入门门槛。

<a id="getting-started"></a>

## 🚀 快速开始

完整教程涵盖环境配置、数据集准备、标准及分阶段运行、模型接入、损失函数和实验流程，详见 **[InduTS-SS 快速上手](docs/Getting_started_cn.md)**。同时提供[英文版教程](docs/Getting_started.md)。

InduTS-SS 提供三种主要的实验运行入口。

### 1. 配置驱动运行（`run_with_yaml.py`）

推荐使用此模式开展完整实验。超参数和环境设置均通过 `.yaml` 文件管理，便于复现和追踪实验。

在 `run_with_yaml.py` 中将 `yaml_name` 设置为要使用的配置文件，例如：

```python
yaml_name = "SS_task/SRU_scripts/yaml/VALSTM.yaml"
```

然后运行：

```bash
python run_with_yaml.py
```

### 2. 预训练与微调运行（`run_pretrain_finetune.py`）

对于需要预训练后再微调等两阶段工作流的模型，请使用 `run_pretrain_finetune.py`。运行器首先以 `model_stage="pretrain"` 训练并保存检查点，随后自动将该检查点传递给微调阶段。

可在 YAML 文件中配置各阶段训练设置：

```yaml
pretrain_epoch: 50
finetune_epoch: 300
pretrain_learning_rate: 0.001
finetune_learning_rate: 0.001
```

然后运行：

```bash
python run_pretrain_finetune.py --yaml ./scripts/SS_task/SRU_scripts/yaml/STDTAEm.yaml
```

完整模式会从模型的 `ModelSpec.pretrain_stages` 读取所需预训练阶段，然后执行微调和测试：

```bash
python run_pretrain_finetune.py --yaml ./scripts/SS_task/SRU_scripts/yaml/STDTAEm.yaml --mode full
```

若要跳过预训练，直接从已有预训练检查点进行微调，请使用 `--mode finetune` 和 `--checkpoint`：

```bash
python run_pretrain_finetune.py --yaml ./scripts/SS_task/SRU_scripts/yaml/STDTAEm.yaml --mode finetune --checkpoint ./results/STDTAEm/your_pretrain_setting/checkpoint.pth
```

可使用 `--no_test` 在训练完成后停止，或通过 `--checkpoint_dir` 指定预训练阶段 checkpoint 的共享归档目录。

### 3. 命令行运行（`run.py`）

此模式适用于自动化任务和批处理。项目为常见配置提供了预配置的 Shell 脚本。

示例：

```bash
bash scripts/LSF_task/DC_scripts/sl16_pl6/iTransformer.sh
```

### ⚠️ 重要说明

首次运行前，请检查源代码、Shell 脚本或 YAML 配置文件中的硬件设置。

- **CUDA：** 根据实际环境设置 `use_gpu` 或对应选项。
- **设备选择：** 通过 `gpu_idx` 或相应脚本参数设置 CUDA 设备索引，默认值通常为 `0`。
- **Windows：** Shell 脚本主要面向类 Linux 环境。在 Windows 中，可通过 `python run_with_yaml.py` 直接运行 YAML 配置，也可使用 Git Bash、WSL 或其他兼容 Shell。

<a id="using-tensorboard"></a>

## 使用 TensorBoard

可在实验配置文件中启用 TensorBoard 日志：

```yaml
use_tensorboard: True
```

实验运行后，TensorBoard 事件文件保存在对应的结果目录下：

```bash
results/{model}/{experiment_setting}/exp_YYYYMMDD_HHMMSS/
```

例如，要查看所有 ARDNN 运行记录：

```bash
tensorboard --logdir ./results/ARDNN
```

然后打开：

```text
http://localhost:6006
```

如果终端中无法使用 `tensorboard` 命令，请先确认已激活正确的 conda 环境：

```bash
conda activate induts-ss
```

在部分 Windows/conda 安装环境中，请通过 Python 启动 TensorBoard：

```bash
python -m tensorboard.main --logdir ./results/ARDNN
```

或者直接调用环境中的可执行文件：

```bash
path/to/conda/envs/induts-ss/Scripts/tensorboard.exe --logdir ./results/ARDNN
```

如果端口 `6006` 已被占用，请指定其他端口：

```bash
python -m tensorboard.main --logdir ./results/ARDNN --port 6007
```

当前训练脚本会将验证损失（如 `Loss/val`）记录到 `Scalars` 面板，并在评估后记录选定的超参数和测试指标。

<a id="what-is-a-soft-sensor"></a>

## 💻 什么是软测量？

### 简单定义

**软测量（Soft Sensor）**利用温度、压力、流量等易于测量的变量以及历史数据，实时估计产品质量、化学浓度、黏度等难以测量的目标变量。软测量可以通过数学模型、机器学习模型或深度学习模型构建。

<a id="data-driven-soft-sensor-modeling"></a>

### 📌 数据驱动的软测量建模

### 定义

随着现代流程工业规模和复杂度不断提高，反应过程或内部过程动态的精确机理模型往往难以获得。因此，数据驱动的软测量模型得到了广泛研究与应用。

与数学模型和机理模型不同，数据驱动模型使用从工业过程中采样的过程数据和质量数据，为工业对象构建黑箱模型。

首先定义主要变量：

- **过程变量：** 表示为

$$
\mathbf{u}_t \in \mathbb{R}^{N_x},
$$

其中 $N_x$ 是过程变量的数量。这些变量表示温度、压力和流量等工业传感器的高频测量值。

- **质量变量：** 表示为

$$
\mathbf{y}_t \in \mathbb{R}^{N_y},
$$

其中 $N_y$ 是质量指标的数量。这些变量表示浓度、纯度等关键产品属性，通常难以测量、测量成本高或存在时延。

- **工况标签：** 表示为

$$
m_t \in \{1, \ldots, K\},
$$

其中 $K$ 是运行工况的数量。这些标签描述稳态、过渡状态和故障状态等不同工业运行状态。

根据现有软测量建模研究，我们定义了三类任务：

1. **软测量回归：** 给定长度为 $T$ 的过程变量历史序列

$$
\mathbf{X}_R =
\{\mathbf{u}_t\}_{t=1}^{T}
\in \mathbb{R}^{T \times N_x},
$$

目标是学习一种映射，推断当前时间步 $T$ 的同步质量变量

$$
\hat{\mathbf{y}}_T \in \mathbb{R}^{N_y}.
$$

2. **软测量预测：** 给定包含过程变量和质量变量、长度为 $T$ 的历史序列

$$
\mathbf{X}_F =
\{(\mathbf{u}_t, \mathbf{y}_t)\}_{t=1}^{T}
\in \mathbb{R}^{T \times (N_x + N_y)},
$$

目标是学习映射 $f$，在前瞻范围 $H$ 内预测未来质量变量

$$
\hat{\mathbf{Y}} =
\{\hat{\mathbf{y}}_t\}_{t=T+1}^{T+H}
\in \mathbb{R}^{H \times N_y}.
$$

3. **软测量序列估计：** 给定由截至时间 $T$ 的过程变量和截至 $T-1$ 的历史质量变量组成的历史序列

$$
\mathbf{X}_S =
\{\mathbf{u}_{1:T}, \mathbf{y}_{1:T-1}\},
$$

目标是学习映射 $f$，估计当前质量变量

$$
\hat{\mathbf{y}}_T \in \mathbb{R}^{N_y}.
$$

第一种形式与软测量的经典定义高度一致：仅使用易获取的过程变量 $\mathbf{U}$ 的当前或历史测量值，估计一个或多个质量变量 $Y$ 的当前值。该设置常见于物理分析仪不可用或速度过慢的实时监测场景。

第二种形式将上述思想扩展到多步超前预测。它假设过程变量和质量变量在历史窗口 $[1,T]$ 内均已被联合观测，并使用该序列预测未来 $H$ 个时间步的质量变量轨迹 $Y=[Y_{T+1},Y_{T+2},\ldots,Y_{T+H}]$。这一设置尤其适用于接近稳态运行的工业过程：虽然实际中质量测量值无法连续获得，但在训练数据中可假设其平滑变化并按固定间隔采样。

### 软测量如何工作？

1. **输入：** 温度、压力、流量等易获取传感器的实时测量值。
2. **模型：** 经过训练、能够学习输入与目标变量关系的模型。
3. **输出：** 持续更新的难测变量估计值。

### 应用

- 过程监测与控制
- 制造过程中的质量预测
- 故障检测与诊断
- 数字孪生与先进过程控制（APC）

<a id="available-models-for-soft-sensors"></a>

## ✨ 可用的软测量模型

感谢原作者所开展的开创性研究。请注意，本工具库中的部分模型为独立复现版本。由于深度学习框架和环境的差异（如从 MATLAB 或 TensorFlow 迁移至 PyTorch），性能可能与原论文报告的结果略有不同。请用户引用相应论文。

我们使用两个缩写表示支持的任务类型：

- **F：** 软测量预测（Soft Sensor Forecasting）
- **R：** 软测量回归（Soft Sensor Regression）

部分模型已经过适配，可同时支持两类任务。详情请参阅 `models/` 中的文件及对应脚本。

| 模型 | 期刊/会议 | F | R | 说明 | 状态 |
| ----- | --------- | - | - | ---- | ---- |
| [SparseTSF](https://ieeexplore.ieee.org/abstract/document/11141354)（Lin 等） | IEEE TPAMI 2026 | 是 |  | 基于 MLP 的时间序列基础模型 | 可用 |
| [PETC-TNet](https://doi.org/10.1109/JSEN.2025.3615736)（He 等） | IEEE Sensors Journal 2025 | 是 |  | 基于 Patch 分解增强 TCN-Transformer 的软测量预测模型 | 可用 |
| [DAMPNN](https://doi.org/10.1109/TII.2024.3475419)（Yan 等） | IEEE TII 2025 |  | 是 | 面向工业软测量的动态自适应消息传递图神经网络 | 可用 |
| [TimeKAN](https://arxiv.org/abs/2502.06910)（Huang 等） | ICLR 2025 | 是 | 是 | 基于 KAN 的时间序列基础模型 | 可用 |
| [TimeFilter](https://arxiv.org/abs/2501.13041)（Hu 等） | ICML 2025 | 是 | 是 | 基于 GNN 的时间序列基础模型 | 可用 |
| [SOFTS](https://arxiv.org/pdf/2404.14197)（Lu 等） | NeurIPS 2024 | 是 | 是 | 基于 Transformer 的基础模型 | 可用 |
| [iTransformer](https://arxiv.org/abs/2310.06625)（Liu 等） | ICLR 2024 | 是 | 是 | 基于 Transformer 的时间序列基础模型 | 可用 |
| [MSGNet](https://dl.acm.org/doi/10.1609/aaai.v38i10.28991)（Cai 等） | AAAI 2024 | 是 | 是 | 基于 GNN 的时间序列基础模型 | 可用 |
| [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2)（Wang 等） | ICLR 2024 | 是 | 是 | 基于 MLP 的时间序列基础模型 | 可用 |
| [FredFormer](https://arxiv.org/abs/2406.09009)（Piao 等） | KDD 2024 | 是 | 是 | 基于 Transformer 的时间序列基础模型 | 可用 |
| [Crossformer](https://openreview.net/pdf?id=vSVLM2j9eie)（Zhang 等） | ICLR 2023 | 是 | 是 | 基于 Transformer 的时间序列基础模型 | 可用 |
| [TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq)（Wu 等） | ICLR 2023 | 是 | 是 | 基于 TCN 的时间序列基础模型 | 可用 |
| [PatchTST](https://arxiv.org/abs/2211.14730)（Nie 等） | ICLR 2023 | 是 | 是 | 基于 Transformer 的时间序列基础模型 | 可用 |
| [DLinear](https://arxiv.org/abs/2205.13504)（Zeng 等） | AAAI 2023 | 是 | 是 | 基于 MLP 的时间序列基础模型 | 可用 |
| [Koopa](https://arxiv.org/pdf/2305.18803)（Liu 等） | NeurIPS 2023 | 是 |  | 基于 Koopman 的时间序列模型 | 可用 |
| [Nonstationary Transformer](https://arxiv.org/abs/2205.14415)（Liu 等） | NeurIPS 2022 | 是 | 是 | 基于 Transformer 的时间序列基础模型 | 可用 |
| [FEDformer](https://proceedings.mlr.press/v162/zhou22g.html)（Zhou 等） | ICML 2022 | 是 | 是 | 基于 Transformer 的时间序列基础模型 | 可用 |
| [Autoformer](https://arxiv.org/abs/2106.13008)（Wu 等） | NeurIPS 2021 | 是 | 是 | 基于 Transformer 的时间序列基础模型 | 可用 |
| [Nystromformer](https://arxiv.org/abs/2102.03902)（Xiong 等） | AAAI 2021 | 是 |  | 基于 Transformer 的时间序列基础模型 | 可用 |
| [TCVAE](https://www.ijcai.org/Proceedings/2019/727)（Wang 等） | IJCAI 2019 | 是 |  | 基于 VAE 的时间序列模型 | 可用 |
| [TCN](https://arxiv.org/abs/1803.01271)（Bai 等） | arXiv 2018 | 是 | 是 | 基于 CNN 的时间序列基础模型 | 可用 |
| [Transformer](https://arxiv.org/abs/1706.03762)（Vaswani 等） | NeurIPS 2017 | 是 | 是 | 基于 Transformer 的时间序列基础模型 | 可用 |
| [VRNN](https://arxiv.org/abs/1506.02216)（Chung 等） | NeurIPS 2015 | 是 | 是 | 基于 RNN 的时间序列软测量模型 | 可用 |
| [LSTM](https://ieeexplore.ieee.org/abstract/document/6795963)（Hochreiter 和 Schmidhuber） | Neural Computation 1997 | 是 | 是 | 基于 RNN 的时间序列软测量模型 | 可用 |
| [LDCNN](https://ieeexplore.ieee.org/document/11408874)（Liu 等） | IEEE TC 2026 |  | 是 | 基于 CNN 的时间序列软测量模型 | 可用 |
| [STDTAEm]() | IEEE TII 2026 | | 是 | 基于 MLP 的时间序列软测量模型 | 参数待定 |
| [FA-SconvAE-LSTM](https://www.sciencedirect.com/science/article/abs/pii/S0952197625005354)（Wu 等） | EAAI 2025 |  | 是 | 基于 CNN 的预训练时间序列软测量模型 | 可用 |
| [ARDNN](https://ieeexplore.ieee.org/document/11122404)（Chen 等） | IEEE Sensors Journal 2025 | 是 |  | 基于 MLP 的时间序列软测量模型 | 可用 |
| [EnvFormer](https://ieeexplore.ieee.org/document/10699388)（Xie 等） | IEEE TIM 2024 | 是 |  | 基于 Transformer 的时间序列软测量模型 | 可用 |
| [MSACNN](https://ieeexplore.ieee.org/document/10465636)（Yuan 等） | IEEE TC 2024 |  | 是 | 基于 CNN 的时间序列软测量模型 | 可用 |
| [GTFTS](https://ieeexplore.ieee.org/document/10664532)（Yan 等） | IEEE TC 2024 | 是 |  | 基于 GNN 的时间序列软测量模型 | 可用 |
| [HSAM-dGRUs](https://ieeexplore.ieee.org/abstract/document/10237000)（He 等） | IEEE TASE 2024 |  | 是 | 基于 RNN 的时间序列软测量模型 | 可用 |
| [GraphSAGE-IMATCN](https://www.sciencedirect.com/science/article/abs/pii/S0957582024009959?via%3Dihub=)（Tuo 等） | PSER 2024 | | 是 | 基于 GNN 的时间序列软测量模型 | 参数待定 |
| [CVAE-SMC](https://ieeexplore.ieee.org/document/10264786)（Sun 等） | IEEE TII 2023 | 是 |  | 基于 VAE 的时间序列软测量模型 | 可用 |
| [DMRIFormer](https://doi.org/10.1109/TII.2022.3227731)（Liu 等） | IEEE TII 2022 | 是 |  | 基于 Transformer 的时间序列软测量模型 | 可用 |
| [DMVAER](https://ieeexplore.ieee.org/document/9797056)（Yao 等） | IEEE TII 2022 |  | 是 | 基于 DVAE 的时间序列软测量模型 | 可用 |
| [GCT](https://ieeexplore.ieee.org/abstract/document/9447941)（Geng 等） | IEEE TII 2021 | 是 | 是 | 基于 Transformer 和 Highway Network 的时间序列软测量模型 | 可用 |
| [DLSTM](https://ieeexplore.ieee.org/document/9531471)（Zhou 等） | IEEE TII 2021 | 是 | 是 | 基于 RNN 的时间序列软测量模型 | 可用 |
| [STALSTM](https://ieeexplore.ieee.org/abstract/document/9062588)（Yuan 等） | IEEE TII 2021 |  | 是 | 基于 RNN 的时间序列软测量模型 | 可用 |
| [DAGRU](https://ieeexplore.ieee.org/document/9174767)（Feng 等） | IEEE TNNLS 2020 |  | 是 | 基于 RNN 的时间序列软测量模型 | 可用 |
| [VALSTM](https://onlinelibrary.wiley.com/doi/10.1002/cjce.23665)（Yuan 等） | CJCE 2019 |  | 是 | 基于 RNN 的时间序列软测量模型 | 可用 |

### 按架构分类的模型

下表按模型架构分组；在每种架构内部，模型按发表年份从新到旧排列。

| 架构 | 模型 | 年份 | 期刊/会议 | F | R | 状态 |
| --- | --- | ---: | --- | :-: | :-: | --- |
| **自编码器 / 生成式模型** | STDTAEm | 2026 | IEEE TII |  | 是 | 可用 |
| **自编码器 / 生成式模型** | [FA-SconvAE-LSTM](https://www.sciencedirect.com/science/article/abs/pii/S0952197625005354)（Wu 等） | 2025 | EAAI |  | 是 | 可用 |
| **自编码器 / 生成式模型** | [CVAE-SMC](https://ieeexplore.ieee.org/document/10264786)（Sun 等） | 2023 | IEEE TII | 是 |  | 可用 |
| **自编码器 / 生成式模型** | [DMVAER](https://ieeexplore.ieee.org/document/9797056)（Yao 等） | 2022 | IEEE TII |  | 是 | 可用 |
| **自编码器 / 生成式模型** | [TCVAE](https://www.ijcai.org/Proceedings/2019/727)（Wang 等） | 2019 | IJCAI | 是 |  | 可用 |
| **自编码器 / 生成式模型** | [VRNN](https://arxiv.org/abs/1506.02216)（Chung 等） | 2015 | NeurIPS | 是 | 是 | 可用 |
| **CNN / TCN** | [LDCNN](https://ieeexplore.ieee.org/document/11408874)（Liu 等） | 2026 | IEEE TC |  | 是 | 可用 |
| **CNN / TCN** | [MSACNN](https://ieeexplore.ieee.org/document/10465636)（Yuan 等） | 2024 | IEEE TC |  | 是 | 可用 |
| **CNN / TCN** | [TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq)（Wu 等） | 2023 | ICLR | 是 | 是 | 可用 |
| **CNN / TCN** | [TCN](https://arxiv.org/abs/1803.01271)（Bai 等） | 2018 | arXiv | 是 | 是 | 可用 |
| **GNN** | [TimeFilter](https://arxiv.org/abs/2501.13041)（Hu 等） | 2025 | ICML | 是 | 是 | 可用 |
| **GNN** | [DAMPNN](https://doi.org/10.1109/TII.2024.3475419)（Yan 等） | 2025 | IEEE TII |  | 是 | 可用 |
| **GNN** | [MSGNet](https://dl.acm.org/doi/10.1609/aaai.v38i10.28991)（Cai 等） | 2024 | AAAI | 是 | 是 | 可用 |
| **GNN** | [GTFTS](https://ieeexplore.ieee.org/document/10664532)（Yan 等） | 2024 | IEEE TC | 是 |  | 可用 |
| **GNN** | [GraphSAGE-IMATCN](https://www.sciencedirect.com/science/article/abs/pii/S0957582024009959?via%3Dihub=)（Tuo 等） | 2024 | PSER |  | 是 | 暂不可用 |
| **MLP / KAN** | [SparseTSF](https://ieeexplore.ieee.org/abstract/document/11141354)（Lin 等） | 2026 | IEEE TPAMI | 是 |  | 可用 |
| **MLP / KAN** | [TimeKAN](https://arxiv.org/abs/2502.06910)（Huang 等） | 2025 | ICLR | 是 | 是 | 可用 |
| **MLP / KAN** | [ARDNN](https://ieeexplore.ieee.org/document/11122404)（Chen 等） | 2025 | IEEE Sensors Journal | 是 |  | 可用 |
| **MLP / KAN** | [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2)（Wang 等） | 2024 | ICLR | 是 | 是 | 可用 |
| **MLP / KAN** | [DLinear](https://arxiv.org/abs/2205.13504)（Zeng 等） | 2023 | AAAI | 是 | 是 | 可用 |
| **RNN** | [HSAM-dGRUs](https://ieeexplore.ieee.org/abstract/document/10237000)（He 等） | 2024 | IEEE TASE |  | 是 | 可用 |
| **RNN** | [DLSTM](https://ieeexplore.ieee.org/document/9531471)（Zhou 等） | 2021 | IEEE TII | 是 | 是 | 可用 |
| **RNN** | [STALSTM](https://ieeexplore.ieee.org/abstract/document/9062588)（Yuan 等） | 2021 | IEEE TII |  | 是 | 可用 |
| **RNN** | [DAGRU](https://ieeexplore.ieee.org/document/9174767)（Feng 等） | 2020 | IEEE TNNLS |  | 是 | 可用 |
| **RNN** | [TS-lambda-GRUs](https://doi.org/10.1109/TIE.2019.2927197) (Xie 等) | 2020 | IEEE TIE | | 是 | Available |
| **RNN** | [VALSTM](https://onlinelibrary.wiley.com/doi/10.1002/cjce.23665)（Yuan 等） | 2019 | CJCE |  | 是 | 可用 |
| **RNN** | [LSTM](https://ieeexplore.ieee.org/abstract/document/6795963)（Hochreiter 和 Schmidhuber） | 1997 | Neural Computation | 是 | 是 | 可用 |
| **Transformer** | [PETC-TNet](https://doi.org/10.1109/JSEN.2025.3615736)（He 等） | 2025 | IEEE Sensors Journal | 是 |  | 可用 |
| **Transformer** | [SOFTS](https://arxiv.org/pdf/2404.14197)（Lu 等） | 2024 | NeurIPS | 是 | 是 | 可用 |
| **Transformer** | [iTransformer](https://arxiv.org/abs/2310.06625)（Liu 等） | 2024 | ICLR | 是 | 是 | 可用 |
| **Transformer** | [FredFormer](https://arxiv.org/abs/2406.09009)（Piao 等） | 2024 | KDD | 是 | 是 | 可用 |
| **Transformer** | [EnvFormer](https://ieeexplore.ieee.org/document/10699388)（Xie 等） | 2024 | IEEE TIM | 是 |  | 可用 |
| **Transformer** | [Crossformer](https://openreview.net/pdf?id=vSVLM2j9eie)（Zhang 等） | 2023 | ICLR | 是 | 是 | 可用 |
| **Transformer** | [PatchTST](https://arxiv.org/abs/2211.14730)（Nie 等） | 2023 | ICLR | 是 | 是 | 可用 |
| **Transformer** | [Nonstationary Transformer](https://arxiv.org/abs/2205.14415)（Liu 等） | 2022 | NeurIPS | 是 | 是 | 可用 |
| **Transformer** | [FEDformer](https://proceedings.mlr.press/v162/zhou22g.html)（Zhou 等） | 2022 | ICML | 是 | 是 | 可用 |
| **Transformer** | [DMRIFormer](https://doi.org/10.1109/TII.2022.3227731)（Liu 等） | 2022 | IEEE TII | 是 |  | 可用 |
| **Transformer** | [Autoformer](https://arxiv.org/abs/2106.13008)（Wu 等） | 2021 | NeurIPS | 是 | 是 | 可用 |
| **Transformer** | [Nystromformer](https://arxiv.org/abs/2102.03902)（Xiong 等） | 2021 | AAAI | 是 |  | 可用 |
| **Transformer** | [GCT](https://ieeexplore.ieee.org/abstract/document/9447941)（Geng 等） | 2021 | IEEE TII | 是 | 是 | 可用 |
| **Transformer** | [Transformer](https://arxiv.org/abs/1706.03762)（Vaswani 等） | 2017 | NeurIPS | 是 | 是 | 可用 |
| **其他** | [Koopa](https://arxiv.org/pdf/2305.18803)（Liu 等） | 2023 | NeurIPS | 是 |  | 可用 |

<a id="available-datasets"></a>

## 🏭 可用数据集

我们提供两个经典基准和三个公开或本地工业数据集：**脱丁烷塔（DC）**、**硫磺回收装置（SRU）**、**炼铁（IM）**、**采矿过程（MP）**、**电厂燃气轮机（PPGAS）**。感谢这些开放数据集的提供者。使用数据集时，请引用相关论文。

### 脱丁烷塔（DC）

**脱丁烷塔**是石油炼制过程中脱硫和石脑油分离装置的关键单元，主要作用是从石脑油物流顶部脱除丙烷和丁烷。由于必须尽可能降低脱丁烷塔塔底的丁烷含量，而直接测量该浓度较为困难，因此构建准确的塔底丁烷浓度软测量模型有助于改善过程控制性能。

| 变量 | 说明 | 类型 |
| ---- | ---- | ---- |
| U1 | 塔顶温度 | 输入 |
| U2 | 塔顶压力 | 输入 |
| U3 | 回流流量 | 输入 |
| U4 | 流向下一工序的流量 | 输入 |
| U5 | 第 6 块塔板温度 | 输入 |
| U6 | 塔底温度 A | 输入 |
| U7 | 塔底温度 B | 输入 |
| Y | C4 中的丁烷含量 | 输出 |

### 硫磺回收装置（SRU）

**硫磺回收装置**是将酸性气体物流转化为单质硫的关键炼化过程。本项目侧重于优化 SRU 的运行性能，尤其关注传感器维护期间的运行情况。

运行效率主要通过调节空气与进料的比例来控制。但由于气体具有腐蚀性，用于测量 $H_2S$ 和 $SO_2$ 的硬件传感器经常损坏。在例行维护或传感器故障期间，实时浓度数据缺失会显著降低 SRU 的运行性能。

| 变量 | 说明 | 类型 |
| ---- | ---- | ---- |
| MEA GAS | MEA 区域的气体流量 | 输入 |
| AIR MEA1 | MEA 区域 1 的空气流量 | 输入 |
| AIR MEA 2 | MEA 区域 2 的空气流量 | 输入 |
| AIR SWS | SWS 区域的空气流量 | 输入 |
| SWS GAS | SWS 区域的气体流量 | 输入 |
| H2S | H2S 浓度 | 输出 |
| SO2 | SO2 浓度 | 输出 |

### 炼铁（IM）

该高铝炼铁数据集源自论文 *A Survey of Data-Driven Soft Sensing in Ironmaking System: Research Status and Opportunities*。数据集包含炼铁生产过程中采集的 **21 个易测过程变量**，并选择**硅含量（Si）**作为软测量建模的质量变量。

在实际炼铁系统中，硅含量等关键质量变量通常需要耗时的离线实验室分析，因此难以进行实时在线测量，并可能产生显著的测量时延。软测量建模可根据易获取的过程测量值估计难测质量变量，是一种有效的数据驱动解决方案。数据集链接：https://github.com/ylkyc/ironmaking-zju。

如果在研究中使用该数据集或相关代码，请引用以下论文：

**[1]** Yan F., Yang C., Zhang X., et al. BTPNet: A Probabilistic Spatial-Temporal Aware Network for Burn-Through Point Multistep Prediction in Sintering Process. *IEEE Transactions on Neural Networks and Learning Systems*, 2024.

**[2]** Yan F., Yang C., Zhang X. DSTED: A denoising spatial-temporal encoder-decoder framework for multistep prediction of burn-through point in sintering process. *IEEE Transactions on Industrial Electronics*, 2022, 69(10): 10735-10744.

### 采矿过程（MP）

**采矿过程**数据集是发布于 Kaggle、名为 *Quality Prediction in a Mining Process* 的真实铁矿石泡沫浮选数据集。泡沫浮选是铁矿石选矿的关键环节，该数据集用于根据常规测量的过程变量预测最终矿石精矿质量。

本地文件 `MP_data.csv` 包含 2017 年 3 月至 9 月的 **3,614 个小时级样本**和 **24 列**数据，包括进料质量变量、药剂流量、矿浆测量值，以及浮选柱空气流量和液位测量值。最终的实验室质量测量指标为 **% Silica Concentrate**。

主要软测量目标是 **% Silica Concentrate**，它表示铁矿石精矿中的杂质水平。由于该质量变量由实验室测量，且至少延迟一小时，因此准确的软测量模型可更早地为过程工程师提供质量估计，并支持及时的纠偏控制。

本仓库提供的版本已由团队进行预处理，包括删除重复时间戳和修正异常值。需要原始文件的用户可从 Kaggle 获取。

数据集来源：https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process。

衷心感谢 **[Eduardo Magalhaes Oliveira 先生](https://www.linkedin.com/in/eduardomoliveira/)**发布这一宝贵的工业数据集，并确认其背景信息。据他提供的信息，该数据集没有对应的特定论文，但它是来自某大型采矿企业的真实数据集，在匿名条件下捐赠，真实性得到保证。

### 电厂燃气轮机（PPGAS）

联合循环电厂的燃烧过程是有害排放物的重要来源，尤其是氮氧化物（NOx）和一氧化碳（CO）。根据欧盟排放法规，NOx 和 CO 排放的干基体积浓度不得超过百万分之 25。因此，准确检测和控制 NOx 排放对电厂十分重要。

软测量技术可以利用间接测量值和数学模型估计、预测 NOx 排放等关键过程变量，从而支持实时估计、及时决策和减排。

该燃气轮机数据集采集自土耳其西北部一家电厂 2011 年的燃气轮机传感器小时平均测量值。样本在三次由部分负荷（75%）快速变化至满负荷（100%）的过程中采集，包括 10 个易测过程变量和 NOx、CO 两个质量变量。

如果在研究中使用该数据集或相关代码，请引用以下论文：

**[1]** Kaya H., Tufekci P., and Uzun E. Predicting CO and NOx emissions from gas turbines: Novel data and a benchmark PEMS. *Turkish Journal of Electrical Engineering and Computer Sciences*, 2019, 27(6): 4783-4796.





<a id="codex-and-claude-code-skills"></a>

## Agent 技能

InduTS-SS 包含面向 **Codex** 和 **Claude Code** 风格工作流的仓库本地说明与技能文件。这些文件可帮助 AI 编程 Agent 理解基准结构、遵守公平比较规则，并复用经过验证的辅助脚本，而非重复编写一次性命令。请通过 Git 克隆 `codex` 分支。

入口文件：

- `AGENTS.md`：面向 Codex 的任务路由和仓库规则。
- `CLAUDE.md`：面向 Claude Code 的任务路由和仓库规则。
- `.codex/skills/`：任务专用技能和可复用脚本。

支持的技能工作流包括：

| 工作流 | 技能 |
| --- | --- |
| 环境配置、CUDA/PyTorch 选择、conda 故障排查 | `.codex/skills/induts-create-env/` |
| 数据集特征报告：平稳性、异常值、相关性、数据划分漂移、绘图 | `.codex/skills/induts-characteristics/` |
| 新数据集接入：CSV 检查、目标验证、YAML 脚手架、冒烟检查 | `.codex/skills/induts-add-dataset/` |
| 新增或修改模型的冒烟测试 | `.codex/skills/induts-smoke/` |
| 最佳结果导出及跨数据集/模型汇总 | `.codex/skills/induts-results-best-export/` |

Codex 或 Claude Code 的提示词示例：

```text
Create a conda environment for this project and choose the correct PyTorch wheel for my GPU.
```

```text
帮我创建适配这个项目和本机 GPU 的 conda 环境。（可自行定义镜像源）
```

```text
Analyze all local datasets and generate stationarity, outlier, correlation, and visualization reports.
```

```text
帮我分析所有本地数据集，生成平稳性、异常值、相关性和可视化报告。
```

```text
Add {file_path}/你的数据集.csv as a new dataset. The target variable is xmeas_38. Generate soft-sensor YAML and run a smoke check.
```

```text
帮我把 {file_path}/你的数据集.csv 加到仓库中，target 变量是 xmeas_38，并生成软测量 YAML 后跑 smoke 检查。
```

```text
Add my new model in {file_path}/VALSTM.py to the model library, register it, create a DC YAML, and run a CPU smoke test.
```

```text
这是我的新模型 {file_path}/VALSTM.py，帮我加入模型库，完成注册，创建 DC YAML，并跑一个 CPU smoke test。
```

```text
Summarize the best soft-sensor results for PatchTST, ARDNN, HSAM_dGRUs, and iTransformer by MSE, and export a report.
```

```text
帮我整理 PatchTST、ARDNN、HSAM_dGRUs、iTransformer 在软测量任务上的最好结果，按 MSE 排序并导出报告。
```

<a id="project-structure"></a>

## 📂 项目结构

```bash
Industrial-Time-Series-Soft-Sensor/
|-- .codex/
|   |-- skills/                    # Codex/Claude Code 基准工作流
|-- data/                          # 数据集与数据处理
|   |-- data_loader.py             # 数据批次与预处理
|   |-- data_provider.py           # 实验数据加载器选择
|   |-- data_visualization.py      # 数据集可视化工具
|   |-- DC/
|   |   |-- debutanizer_column.csv
|   |   |-- mode_labels.txt
|   |-- SRU/
|   |   |-- SRU_data.csv
|   |-- Ironmaking/
|   |   |-- Ironmaking.csv
|   |-- MP/
|   |   |-- MP_data.csv
|   |-- PPGAS/
|   |   |-- gt_2012.csv
|   |-- TE/
|       |-- TEP.csv
|-- exp/                           # 实验工作流
|   |-- exp_basic.py               # 基础实验接口
|   |-- exp_factory.py             # 实验工厂
|   |-- exp_short_term_forecasting.py
|   |-- exp_soft_sensor.py
|   |-- losses.py
|-- layers/                        # 神经网络层
|-- models/                        # 自包含模型包
|   |-- base/                      # 公共 Config 与 ModelSpec 定义
|   |-- registry.py                # 规范模型注册表与别名
|   |-- <Model>/
|       |-- model_arch.py          # 模型架构
|       |-- model_config.py        # 模型配置 dataclass
|       |-- model_spec.py          # Benchmark 集成能力声明
|       |-- __init__.py            # 模型包公开接口
|-- runner/                        # 实验流程编排
|   |-- builder.py                 # 配置、实验、日志与 checkpoint 构建
|   |-- train_test.py              # 普通训练与测试
|   |-- test_checkpoint.py         # 已有 checkpoint 的独立测试
|   |-- pretrain_finetune.py       # 完整预训练微调与只微调流程
|-- scripts/
|   |-- SS_task/                   # 软测量回归脚本
|   |-- LSF_task/                  # 软测量预测脚本
|-- utils/                         # 工具函数
|   |-- configs.py
|   |-- logger.py
|   |-- metrics.py
|   |-- scaler.py
|   |-- tools.py
|-- results/                       # 已保存的结果
|-- AGENTS.md                      # Codex 入口说明
|-- CLAUDE.md                      # Claude Code 入口说明
|-- run.py                         # 命令行入口
|-- run_with_yaml.py               # 基于 YAML 的入口
|-- run_pretrain_finetune.py       # 预训练与微调入口
|-- requirements.txt
|-- LICENSE.txt
|-- readme.md
```

### 目录概览

#### `data/`

该目录负责数据集管理和预处理，包含数据加载器、数据提供函数、可视化工具以及原始 CSV 数据文件。

#### `exp/`

该目录包含实验逻辑和工作流。`exp_basic.py` 定义标准实验接口，`exp_factory.py` 根据配置创建实验实例，各任务专用文件负责训练、验证、测试和评估。

#### `layers/`

该目录包含神经网络构建模块，包括注意力机制，以及 `SelfAttention_Family.py`、`Transformer_EncDec.py`、`DMRIFormer_EncDec.py` 和 `NystromAttention.py` 等编码器—解码器架构。

#### `models/`

每个模型都是一个自包含包，包含模型架构、模型配置和 benchmark 能力声明。`models/registry.py` 管理规范模型名称，并加载模型包导出的 `Model`、`MODEL_CONFIG` 和 `MODEL_SPEC`。

`MODEL_SPEC` 声明模型支持的任务、数据表示、Loss 类型和可选预训练阶段；模型超参数保留在模型自己的 `MODEL_CONFIG` dataclass 和 YAML 中。

#### `scripts/`

该目录存放实验脚本，按任务类型（`SS_task/` 和 `LSF_task/`）及数据集组织，以便复现实验。

#### `utils/`

该目录提供配置、日志、指标、归一化、掩码、早停、数据增强及 TensorBoard 集成等辅助函数。

#### `results/`

该目录存放实验输出，包括已训练模型、日志、指标和预测结果。

<a id="installation"></a>

## 🚩 安装

克隆仓库：

```bash
git clone https://github.com/YuAn-06/Industrial-Time-Series-Soft-Sensor.git
cd Industrial-Time-Series-Soft-Sensor
```

建议新建 conda 环境：

```bash
conda create -n induts-ss python=3.10
conda activate induts-ss
```

安装支持 CUDA 12.8 的 PyTorch：

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

如果 CUDA 版本不同，请从 PyTorch 官方网站安装对应版本。

安装其余依赖：

```bash
pip install -r requirements.txt
```

<a id="citation"></a>

## 引用

如果本仓库对您的研究有所帮助，请考虑引用相关数据集论文、模型论文以及此基准框架。InduTS-SS 的 BibTeX 条目将在相关论文发布后更新。

<a id="acknowledgements"></a>

## ❤️ 致谢

衷心感谢开源社区的贡献。本代码库受到深度学习和时间序列分析领域多个仓库的启发并参考了其中的实现，包括 [Time-Series-Library](https://github.com/thuml/Time-Series-Library) 和 [PyITS](https://github.com/Master-PLC/PyITS)。感谢这些项目的维护者提供清晰透明的文档和模块化设计，它们为本项目组织 `data`、`models` 和 `exp` 模块提供了宝贵参考。同时，也感谢所有让本项目得以实现的科学计算工具库开发者。

<a id="license"></a>

## 许可证

本项目基于 Apache License 2.0 发布。详情请参阅 [LICENSE.txt](LICENSE.txt)。

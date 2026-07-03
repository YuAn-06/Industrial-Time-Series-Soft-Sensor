<div align="center">
  <img src="docs/Logo.jpg#gh-light-mode-only" height="200">
  <h3><b>A Unified and Fair Library and Benchmark for Industrial Time-Series Soft Sensors</b></h3>
</div>

<p align="center">
  <a href="https://www.python.org/">
    <img alt="Python version" src="https://img.shields.io/badge/Python-v3.10+-E97040?logo=python&logoColor=white">
  </a>
  <a href="https://pytorch.org/">
    <img alt="powered by PyTorch" src="https://img.shields.io/badge/PyTorch-v2.7+-E97040?logo=pytorch&logoColor=white">
  </a>
  <a href="https://www.apache.org/licenses/">
    <img alt="Apache License" src="https://img.shields.io/badge/License-Apache2.0-E9BB41?logo=opensourceinitiative&logoColor=white">
  </a>
</p>

<p align="center">
  <a href="readme_cn.md">简体中文介绍</a>
</p>

---

**Industrial Time-Series Soft Sensor (InduTS-SS)** is an open-source library designed for researchers working on time-series data analysis and soft sensor modeling for complex industrial processes.

InduTS-SS provides a unified framework for industrial soft sensing, including dataset loading, preprocessing, model training, evaluation, and benchmarking. It is intended to make model comparison more fair, reproducible, and convenient.

## Table of Contents

- [What's New](#whats-new)
- [Motivation](#motivation)
- [Getting Started](#getting-started)
- [Using TensorBoard](#using-tensorboard)
- [What Is a Soft Sensor?](#what-is-a-soft-sensor)
- [Data-Driven Soft Sensor Modeling](#data-driven-soft-sensor-modeling)
- [Available Models for Soft Sensors](#available-models-for-soft-sensors)
- [Available Datasets](#available-datasets)
- [Codex and Claude Code Skills](#codex-and-claude-code-skills)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

<a id="whats-new"></a>

## 📢 What's New

**Update - 2026/6/21:** We now have built-in support for **agent coding** workflows such as **Codex** and **Claude Code** skills. These skills help agents better understand benchmark tasks and help users get started more easily. Features include environment creation, dataset onboarding, model integration, smoke tests, and full-loop model performance reporting. Please see the sections below for details and git clone `codex` branch.

**Update - 2026/6/15:** We have added the **Mining Process (MP)** dataset to InduTS-SS. We sincerely thank **Mr. Eduardo Magalhaes Oliveira** for releasing this valuable real-world industrial dataset and confirming its authenticity.

**Update - 2026/6/06:** Now, InduTS-SS can support pre-training and fine-tuning models. We provide `run_pretrain_finetune.py` to provide the whole training procedure for pretraining models. Meanwhile, we provide two new models, including a GNN-based model and an AE-based model, while their hyperparameters are not determined accurately.

**Update - 2026/5/25:** We have released the first version of InduTS-SS. Papers based on this benchmark framework are currently under submission. We welcome users to try the library, report issues, and contribute improvements.

<a id="motivation"></a>

## ✨ Motivation

Modern machine learning libraries provide powerful tools for time-series modeling, but they are not always tailored to the specific needs of soft sensor development in industrial processes. Researchers often spend significant effort reimplementing preprocessing pipelines, feature extraction methods, and evaluation protocols from scratch, which leads to fragmented codebases and makes fair model comparison difficult. Moreover, many published soft sensor approaches do not provide open implementations, making reproducibility and benchmarking challenging.

To bridge this gap, we introduce **InduTS-SS**, an open-source library that provides a consistent and modular framework for building, training, and evaluating soft sensors on multivariate industrial time-series data. By standardizing common components, from input representation to forecasting horizons, InduTS-SS aims to accelerate research, improve comparability, and lower the barrier to entry for new contributors in this field.

<a id="getting-started"></a>

## 🚀 Getting Started

InduTS-SS provides three primary entry points for running experiments.

### 1. Configuration-Driven Execution (`run_with_yaml.py`)

This mode is recommended for comprehensive experiments. Hyperparameters and environment settings are managed through `.yaml` files, which makes experiments easier to reproduce and track.

Set `yaml_name` in `run_with_yaml.py` to the configuration file you want to use, for example:

```python
yaml_name = "SS_task/SRU_scripts/yaml/VALSTM.yaml"
```

Then run:

```bash
python run_with_yaml.py
```

### 2. Pretraining and Finetuning Execution (`run_pretrain_finetune.py`)

For models that require a two-stage workflow, such as pretraining followed by finetuning, use `run_pretrain_finetune.py`. The runner first trains with `model_stage="pretrain"`, saves the checkpoint, and then automatically passes that checkpoint to the finetuning stage.

The stage-specific training settings can be configured in the YAML file:

```yaml
pretrain_epoch: 50
finetune_epoch: 300
pretrain_learning_rate: 0.001
finetune_learning_rate: 0.001
```

Then run:

```bash
python run_pretrain_finetune.py --yaml ./scripts/SS_task/SRU_scripts/yaml/STDTAEm.yaml
```

You can also choose stages from the command line:

```bash
python run_pretrain_finetune.py --yaml ./scripts/SS_task/SRU_scripts/yaml/STDTAEm.yaml --stages pretrain finetune --test_stages finetune
```

To skip pretraining and directly finetune from an existing checkpoint, pass `--initial_ckpt`:

```bash
python run_pretrain_finetune.py --yaml ./scripts/SS_task/SRU_scripts/yaml/STDTAEm.yaml --initial_ckpt ./results/STDTAEm/your_pretrain_setting/checkpoint.pth
```

When `initial_ckpt` is not empty, the runner will use only the finetuning stage.

### 3. Command-Line Execution (`run.py`)

This mode is useful for automated tasks and batch processing. Pre-configured shell scripts are provided for common settings.

Example:

```bash
bash scripts/LSF_task/DC_scripts/sl16_pl6/iTransformer.sh
```

### ⚠️ Important Notes

Before launching your first run, please check the hardware settings in the source code, shell scripts, or YAML configuration files.

- **CUDA:** Set `use_gpu` or the equivalent option according to your environment.
- **Device selection:** Set the CUDA device index through `gpu_idx` or the corresponding script argument. The default value is usually `0`.
- **Windows:** Shell scripts are mainly designed for Linux-style environments. On Windows, you can run YAML configurations directly with `python run_with_yaml.py`, or use Git Bash, WSL, or another compatible shell.

<a id="using-tensorboard"></a>

## Using TensorBoard

TensorBoard logging can be enabled from the experiment configuration file:

```yaml
use_tensorboard: True
```

After running an experiment, TensorBoard event files are saved under the corresponding result directory:

```bash
results/{model}/{experiment_setting}/exp_YYYYMMDD_HHMMSS/
```

For example, to view all ARDNN runs:

```bash
tensorboard --logdir ./results/ARDNN
```

Then open:

```text
http://localhost:6006
```

If the `tensorboard` command is not available in your terminal, first make sure the correct conda environment is activated:

```bash
conda activate induts-ss
```

On some Windows/conda installations, launch TensorBoard through Python instead:

```bash
python -m tensorboard.main --logdir ./results/ARDNN
```

Alternatively, call the executable from the environment directly:

```bash
path/to/conda/envs/induts-ss/Scripts/tensorboard.exe --logdir ./results/ARDNN
```

If port `6006` is already occupied, specify another port:

```bash
python -m tensorboard.main --logdir ./results/ARDNN --port 6007
```

The current training scripts log validation loss to the `Scalars` dashboard, such as `Loss/val`, and selected hyperparameters and test metrics after evaluation.

<a id="what-is-a-soft-sensor"></a>

## 💻 What Is a Soft Sensor?

### Simple Definition

A **soft sensor** uses easily measurable variables, such as temperature, pressure, and flow rate, together with historical data to estimate hard-to-measure target variables, such as product quality, chemical concentration, or viscosity, in real time. Soft sensors can be built using mathematical models, machine learning models, or deep learning models.

<a id="data-driven-soft-sensor-modeling"></a>

### 📌 Data-Driven Soft Sensor Modeling

### Definition

Due to the increasing scale and complexity of modern process industries, it is often difficult to obtain accurate mechanistic models of reaction processes or internal process dynamics. Therefore, data-driven soft sensing models have been widely studied and applied.

Unlike mathematical and mechanistic models, data-driven models use process and quality data sampled from industrial processes to construct black-box models for industrial objects.

We first define the main variables:

- **Process Variables:** Denoted as

$$
\mathbf{u}_t \in \mathbb{R}^{N_x},
$$

where $N_x$ is the number of process variables. These variables represent high-frequency measurements from industrial sensors, such as temperature, pressure, and flow rate.

- **Quality Variables:** Denoted as

$$
\mathbf{y}_t \in \mathbb{R}^{N_y},
$$

where $N_y$ is the number of quality indicators. These variables represent key product properties, such as concentration and purity, that are often difficult, costly, or time-delayed to measure.

- **Mode Labels:** Denoted as

$$
m_t \in \{1, \ldots, K\},
$$

where $K$ is the number of operating modes. These labels characterize different industrial regimes, such as steady states, transitions, and fault conditions.

Based on existing soft sensor modeling studies, we define three tasks:

1. **Soft Sensor Regression:** Given a historical sequence of process variables of length $T$, denoted as

$$
\mathbf{X}_R =
\{\mathbf{u}_t\}_{t=1}^{T}
\in \mathbb{R}^{T \times N_x},
$$

the objective is to learn a mapping that infers the synchronous quality variable

$$
\hat{\mathbf{y}}_T \in \mathbb{R}^{N_y}
$$

at the current time step $T$.

2. **Soft Sensor Forecasting:** Given a historical sequence of length $T$ containing both process and quality variables, denoted as

$$
\mathbf{X}_F =
\{(\mathbf{u}_t, \mathbf{y}_t)\}_{t=1}^{T}
\in \mathbb{R}^{T \times (N_x + N_y)},
$$

the objective is to learn a mapping $f$ that forecasts the future quality variables

$$
\hat{\mathbf{Y}} =
\{\hat{\mathbf{y}}_t\}_{t=T+1}^{T+H}
\in \mathbb{R}^{H \times N_y}
$$

over a look-ahead horizon $H$.

3. **Soft Sensor Sequential Estimation:** Given a historical sequence consisting of process variables up to time $T$ and past quality variables up to $T-1$, denoted as

$$
\mathbf{X}_S =
\{\mathbf{u}_{1:T}, \mathbf{y}_{1:T-1}\},
$$

the objective is to learn a mapping $f$ that estimates the current quality variable

$$
\hat{\mathbf{y}}_T \in \mathbb{R}^{N_y}.
$$

The first formulation aligns closely with the classical definition of soft sensing: estimating the current value of one or more quality variables $Y$ using only contemporaneous or historical measurements of easily accessible process variables $\mathbf{U}$. This setting is common in real-time monitoring scenarios where physical analyzers are unavailable or too slow.

The second formulation extends this idea to multi-step-ahead forecasting. It assumes that both process variables and quality variables have been jointly observed over a historical window $[1, T]$, and uses this sequence to predict a future trajectory of quality variables $Y = [Y_{T+1}, Y_{T+2}, \ldots, Y_{T+H}]$ over a horizon of $H$ steps. This setup is particularly relevant when the underlying industrial process operates near steady state, where quality measurements, although not continuously available in practice, are assumed to evolve smoothly and can be sampled at regular intervals in the training data.

### How Do Soft Sensors Work?

1. **Input:** Real-time measurements from easy-to-access sensors, such as temperature, pressure, and flow.
2. **Model:** A trained model that learns the relationship between inputs and the target variable.
3. **Output:** An estimated value of the hard-to-measure variable, updated continuously.

### Applications

- Process monitoring and control
- Quality prediction in manufacturing
- Fault detection and diagnosis
- Digital twins and advanced process control (APC)

<a id="available-models-for-soft-sensors"></a>

## ✨ Available Models for Soft Sensors

We thank the original authors for their pioneering research. Please note that several models in this library are independent reimplementations. Due to differences in deep learning frameworks and environments, such as migration from MATLAB or TensorFlow to PyTorch, performance may differ slightly from originally reported results. Users are kindly asked to cite the relevant papers.

We use two abbreviations to indicate the supported task types:

- **F:** Soft Sensor Forecasting
- **R:** Soft Sensor Regression

Some models have been adapted to support both tasks. Please refer to the files in `models/` and the corresponding scripts for details.

| Model | Journal/Conference | F | R | Remark | Status |
| ----- | ------------------ | - | - | ------ | ------ |
| [SparseTSF](https://ieeexplore.ieee.org/abstract/document/11141354) (Lin et al.) | IEEE TPAMI 2026 | Yes |  | MLP-based time-series foundation model | Available |
| [TimeKAN](https://arxiv.org/abs/2502.06910) (Huang et al.) | ICLR 2025 | Yes | Yes | KAN-based time-series foundation model | Available |
| [TimeFilter](https://arxiv.org/abs/2501.13041) (Hu et al.) | ICML 2025 | Yes | Yes | GNN-based time-series foundation model | Available |
| [SOFTS](https://arxiv.org/pdf/2404.14197) (Lu et al.) | NeurIPS 2024 | Yes | Yes | Transformer-based foundation model | Available |
| [iTransformer](https://arxiv.org/abs/2310.06625) (Liu et al.) | ICLR 2024 | Yes | Yes | Transformer-based time-series foundation model | Available |
| [MSGNet](https://dl.acm.org/doi/10.1609/aaai.v38i10.28991) (Cai et al.) | AAAI 2024 | Yes | Yes | GNN-based time-series foundation model | Available |
| [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2) (Wang et al.) | ICLR 2024 | Yes | Yes | MLP-based time-series foundation model | Available |
| [FredFormer](https://arxiv.org/abs/2406.09009) (Piao et al.) | KDD 2024 | Yes | Yes | Transformer-based time-series foundation model | Available |
| [Crossformer](https://openreview.net/pdf?id=vSVLM2j9eie) (Zhang et al.) | ICLR 2023 | Yes | Yes | Transformer-based time-series foundation model | Available |
| [TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq) (Wu et al.) | ICLR 2023 | Yes | Yes | TCN-based time-series foundation model | Available |
| [PatchTST](https://arxiv.org/abs/2211.14730) (Nie et al.) | ICLR 2023 | Yes | Yes | Transformer-based time-series foundation model | Available |
| [DLinear](https://arxiv.org/abs/2205.13504) (Zeng et al.) | AAAI 2023 | Yes | Yes | MLP-based time-series foundation model | Available |
| [Koopa](https://arxiv.org/pdf/2305.18803) (Liu et al.) | NeurIPS 2023 | Yes |  | Koopman-based time-series model | Available |
| [Nonstationary Transformer](https://arxiv.org/abs/2205.14415) (Liu et al.) | NeurIPS 2022 | Yes | Yes | Transformer-based time-series foundation model | Available |
| [FEDformer](https://proceedings.mlr.press/v162/zhou22g.html) (Zhou et al.) | ICML 2022 | Yes | Yes | Transformer-based time-series foundation model | Available |
| [Autoformer](https://arxiv.org/abs/2106.13008) (Wu et al.) | NeurIPS 2021 | Yes | Yes | Transformer-based time-series foundation model | Available |
| [Nystromformer](https://arxiv.org/abs/2102.03902) (Xiong et al.) | AAAI 2021 | Yes |  | Transformer-based time-series foundation model | Available |
| [TCVAE](https://www.ijcai.org/Proceedings/2019/727) (Wang et al.) | IJCAI 2019 | Yes |  | VAE-based time-series model | Available |
| [TCN](https://arxiv.org/abs/1803.01271) (Bai et al.) | arXiv 2018 | Yes | Yes | CNN-based time-series foundation model | Available |
| [Transformer](https://arxiv.org/abs/1706.03762) (Vaswani et al.) | NeurIPS 2017 | Yes | Yes | Transformer-based time-series foundation model | Available |
| [VRNN](https://arxiv.org/abs/1506.02216) (Chung et al.) | NeurIPS 2015 | Yes | Yes | RNN-based time-series soft sensor model | Available |
| [LSTM](https://ieeexplore.ieee.org/abstract/document/6795963) (Hochreiter and Schmidhuber) | Neural Computation 1997 | Yes | Yes | RNN-based time-series soft sensor model | Available |
| [LDCNN](https://ieeexplore.ieee.org/document/11408874) (Liu et al.) | IEEE TC 2026 |  | Yes | CNN-based time-series soft sensor model | Available |
| [STDTAEm]() | IEEE TII 2026 | | Yes | MLP-based time-series soft sensor model | Unavailable |
| [ARDNN](https://ieeexplore.ieee.org/document/11122404) (Chen et al.) | IEEE Sensors Journal 2025 | Yes |  | MLP-based time-series soft sensor model | Available |
| [Envformer](https://ieeexplore.ieee.org/document/10699388) (Xie et al.) | IEEE TIM 2024 | Yes |  | Transformer-based time-series soft sensor model | Available |
| [MSACNN](https://ieeexplore.ieee.org/document/10465636) (Yuan et al.) | IEEE TC 2024 |  | Yes | CNN-based time-series soft sensor model | Available |
| [GTFTS](https://ieeexplore.ieee.org/document/10664532) (Yan et al.) | IEEE TC 2024 | Yes |  | GNN-based time-series soft sensor model | Available |
| [HSAM-dGRUs](https://ieeexplore.ieee.org/abstract/document/10237000) (He et al.) | IEEE TASE 2024 |  | Yes | RNN-based time-series soft sensor model | Available |
| [GraphSAGE-IMATCN](https://www.sciencedirect.com/science/article/abs/pii/S0957582024009959?via%3Dihub=) (Tuo et al.) | PSER 2024 | | Yes | GNN-based time-series soft sensor model | Unavailable |
| [CVAE-SMC](https://ieeexplore.ieee.org/document/10264786) (Sun et al.) | IEEE TII 2023 | Yes |  | VAE-based time-series soft sensor mode                       | Available |
| [DMRIFormer](https://doi.org/10.1109/TII.2022.3227731) (Liu et al.) | IEEE TII 2022 | Yes |  | Transformer-based time-series soft sensor model | Available |
| [DMVAER](https://ieeexplore.ieee.org/document/9797056) (Yao et al.) | IEEE TII 2022 |  | Yes | DVAE-based time-series soft sensor model | Available |
| [GCT](https://ieeexplore.ieee.org/abstract/document/9447941) (Geng et al.) | IEEE TII 2021 | Yes | Yes | Transformer and highway-network-based time-series soft sensor model | Available |
| [DLSTM](https://ieeexplore.ieee.org/document/9531471) (Zhou et al.) | IEEE TII 2021 | Yes | Yes | RNN-based time-series soft sensor model | Available |
| [STALSTM](https://ieeexplore.ieee.org/abstract/document/9062588) (Yuan et al.) | IEEE TII 2021 |  | Yes | RNN-based time-series soft sensor model | Available |
| [DAGRU](https://ieeexplore.ieee.org/document/9174767) (Feng et al.) | IEEE TNNLS 2020 |  | Yes | RNN-based time-series soft sensor model | Available |
| [VALSTM](https://onlinelibrary.wiley.com/doi/10.1002/cjce.23665) (Yuan et al.) | CJCE 2019 |  | Yes | RNN-based time-series soft sensor model | Available |



<a id="available-datasets"></a>

## 🏭 Available Datasets

We provide two classic benchmarks and three public or local industrial datasets: **Debutanizer Column (DC)**, **Sulfur Recovery Unit (SRU)**, **Ironmaking (IM)**, **Mining Process (MP)**, and **Power Plant Gas Turbine (PPGAS)**. We are grateful to the providers of these open datasets. If you use these datasets, please cite the relevant papers.

### Debutanizer Column (DC)

The **Debutanizer Column** is a critical unit in the desulfurization and naphtha splitter plant within petroleum refining processes. Its primary function is to remove propane and butane as overhead products from the naphtha stream. Since the butane content in the debutanizer bottom must be minimized, and direct measurement of this concentration is challenging, an accurate soft sensor for bottom butane concentration is valuable for improving process control performance.

| Variable | Description | Type |
| -------- | ----------- | ---- |
| U1 | Top temperature | Input |
| U2 | Top pressure | Input |
| U3 | Reflux flow | Input |
| U4 | Flow to next process | Input |
| U5 | 6th tray temperature | Input |
| U6 | Bottom temperature A | Input |
| U7 | Bottom temperature B | Input |
| Y | Butane content in C4 | Output |

### Sulfur Recovery Unit (SRU)

The **Sulfur Recovery Unit** is a critical refinery process designed to convert acid gas streams into elemental sulfur. This project focuses on optimizing SRU operational performance, particularly during periods of sensor maintenance.

Operational efficiency is primarily controlled by regulating the air-to-feed ratio. However, hardware sensors for $H_2S$ and $SO_2$ are frequently damaged due to the corrosive nature of the gases. During routine maintenance or sensor failure, the lack of real-time concentration data significantly degrades SRU performance.

| Variable | Description | Type |
| -------- | ----------- | ---- |
| MEA GAS | Gas flow in the MEA zone | Input |
| AIR MEA1 | Air flow in MEA zone 1 | Input |
| AIR MEA 2 | Air flow in MEA zone 2 | Input |
| AIR SWS | Air flow in the SWS zone | Input |
| SWS GAS | Gas flow in the SWS zone | Input |
| H2S | H2S concentration | Output |
| SO2 | SO2 concentration | Output |

### Ironmaking (IM)

This high-alumina ironmaking dataset is derived from the article *A Survey of Data-Driven Soft Sensing in Ironmaking System: Research Status and Opportunities*. The dataset contains **21 easily measured process variables** collected from an ironmaking production process, while **silicon content (Si)** is selected as the quality variable for soft sensor modeling.

In practical ironmaking systems, key quality variables such as silicon content are difficult to measure online in real time because they usually require offline laboratory analysis, which is time-consuming and may introduce significant measurement delays. Soft sensor modeling provides an effective data-driven solution by estimating hard-to-measure quality variables from readily available process measurements. The link of dataset is in https://github.com/ylkyc/ironmaking-zju.

If you use this dataset or the related code in your research, please cite the following papers:

**[1]** Yan F., Yang C., Zhang X., et al. BTPNet: A Probabilistic Spatial-Temporal Aware Network for Burn-Through Point Multistep Prediction in Sintering Process. *IEEE Transactions on Neural Networks and Learning Systems*, 2024.

**[2]** Yan F., Yang C., Zhang X. DSTED: A denoising spatial-temporal encoder-decoder framework for multistep prediction of burn-through point in sintering process. *IEEE Transactions on Industrial Electronics*, 2022, 69(10): 10735-10744.

### Mining Process (MP)

The **Mining Process** dataset is a real-world iron ore froth flotation dataset released on Kaggle as *Quality Prediction in a Mining Process*. Froth flotation is a key mineral processing stage used to concentrate iron ore, and the dataset is intended for predicting final ore concentrate quality from routinely measured process variables.

The local file `MP_data.csv` contains **3,614 hourly samples** and **24 columns** from March 2017 to September 2017. It includes feed quality variables, reagent flows, ore pulp measurements, and flotation column air-flow and level measurements. The final laboratory quality measurements are **% Silica Concentrate**.

The main soft sensor target is **% Silica Concentrate**, which represents the impurity level in the iron ore concentrate. Since this quality variable is measured in the laboratory and is delayed by at least one hour, accurate soft sensor modeling can provide earlier quality estimates for process engineers and support timely corrective control.

The version provided in this repository has been preprocessed by our team, including removing duplicated timestamps and correcting abnormal values. Users who need the original raw file can access it from Kaggle.

Dataset source: https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process.

We sincerely thank **[Mr. Eduardo Magalhaes Oliveira](https://www.linkedin.com/in/eduardomoliveira/)** for releasing this valuable industrial dataset and for confirming its background. According to his kind information, there is no specific paper associated with this dataset; however, it is a real dataset from a large-scale mining industry, donated under the condition of anonymity, with its authenticity guaranteed.

### Power Plant Gas Turbine (PPGAS)

The combustion process in combined cycle power plants is a major contributor to harmful emissions, especially nitrogen oxides (NOx) and carbon monoxide (CO). Under European Union emission regulations, NOx and CO emissions are limited to 25 parts per million by dry volume. Accurate detection and control of NOx emissions are therefore important for power plants.

Soft sensor technology can estimate and predict key process variables, such as NOx emissions, using indirect measurements and mathematical models. This supports real-time estimation, timely decision-making, and emission mitigation.

The gas turbine dataset is collected from hourly averaged sensor measurements of a power plant gas turbine in northwestern Turkey in 2011. The samples were collected under three rapid operating changes from part load (75%) to full load (100%). It includes 10 easily measurable process variables and 2 quality variables: NOx and CO.

If you use this dataset or the related code in your research, please cite the following paper:

**[1]** Kaya H., Tufekci P., and Uzun E. Predicting CO and NOx emissions from gas turbines: Novel data and a benchmark PEMS. *Turkish Journal of Electrical Engineering and Computer Sciences*, 2019, 27(6): 4783-4796.



### Dataset Citation Notice

If you use these datasets in your paper, please also cite:

**L. Fortuna, S. Graziani, A. Rizzo, and M. G. Xibilia. Soft Sensors for Monitoring and Control of Industrial Processes. Advances in Industrial Control, 2007.**

<a id="codex-and-claude-code-skills"></a>

## Agent Skills

InduTS-SS includes repository-local instructions and skills for **Codex** and **Claude Code** style workflows. These files help AI coding agents understand the benchmark structure, preserve fair-comparison rules, and reuse validated helper scripts instead of rewriting one-off commands. Please git clone `codex` branch.

Entry files:

- `AGENTS.md`: Codex-oriented routing and repository rules.
- `CLAUDE.md`: Claude Code-oriented routing and repository rules.
- `.codex/skills/`: task-specific skills and reusable scripts.

Supported skill workflows include:

| Workflow | Skill |
| --- | --- |
| Environment setup, CUDA/PyTorch selection, conda troubleshooting | `.codex/skills/induts-create-env/` |
| Dataset characteristics reports: stationarity, outliers, correlations, split drift, plots | `.codex/skills/induts-characteristics/` |
| New dataset onboarding: CSV inspection, target validation, YAML scaffolding, smoke checks | `.codex/skills/induts-add-dataset/` |
| New or edited model smoke tests | `.codex/skills/induts-smoke/` |
| Best-result export and cross-dataset/model summaries | `.codex/skills/induts-results-best-export/` |

Example prompts for Codex or Claude Code:

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

## 📂 Project Structure

```bash
Industrial-Time-Series-Soft-Sensor/
|-- .codex/
|   |-- skills/                   # Codex/Claude Code benchmark workflows
|-- data/                         # Datasets and data processing
|   |-- data_loader.py             # Data batches and preprocessing
|   |-- data_provider.py           # Dataloader selection for experiments
|   |-- data_visualization.py      # Dataset visualization utilities
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
|-- exp/                          # Experiment workflows
|   |-- exp_basic.py              # Basic experiment interface
|   |-- exp_factory.py            # Experiment factory
|   |-- exp_short_term_forecasting.py
|   |-- exp_soft_sensor.py
|   |-- losses.py
|-- layers/                       # Neural network layers
|-- models/                       # Model implementations
|-- scripts/
|   |-- SS_task/                  # Soft sensor regression scripts
|   |-- LSF_task/                 # Soft sensor forecasting scripts
|-- utils/                        # Utility functions
|   |-- configs.py
|   |-- ExpConfigs.py
|   |-- logger.py
|   |-- metrics.py
|   |-- scaler.py
|   |-- tools.py
|-- results/                      # Saved results
|-- AGENTS.md                     # Codex entry instructions
|-- CLAUDE.md                     # Claude Code entry instructions
|-- run.py                        # Command-line entry point
|-- run_with_yaml.py              # YAML-based entry point
|-- requirements.txt
|-- LICENSE.txt
|-- readme.md
```

### Directory Overview

#### `data/`

This directory is responsible for dataset management and preprocessing. It contains data loaders, provider functions, visualization utilities, and raw CSV data files.

#### `exp/`

This directory contains experimental logic and workflows. `exp_basic.py` defines the standard experiment interface, `exp_factory.py` creates experiment instances based on configuration, and task-specific files implement training, validation, testing, and evaluation.

#### `layers/`

This directory contains neural network building blocks, including attention mechanisms and encoder-decoder architectures such as `SelfAttention_Family.py`, `Transformer_EncDec.py`, `DMRIFormer_EncDec.py`, and `NystromAttention.py`.

#### `models/`

This directory contains model implementations, such as `ARDNN.py`, `Autoformer.py`, `PatchTST.py`, `DMVAER.py`, `VALSTM.py`, and `iTransformer.py`.

#### `scripts/`

This directory stores experiment scripts. It is organized by task type (`SS_task/` and `LSF_task/`) and dataset to facilitate reproducible experiments.

#### `utils/`

This directory provides helper functions for configuration, logging, metrics, normalization, masking, early stopping, data augmentation, and TensorBoard integration.

#### `results/`

This directory stores experiment outputs, including trained models, logs, metrics, and prediction results.



<a id="installation"></a>

## 🚩 Installation

Clone the repository:

```bash
git clone https://github.com/YuAn-06/Industrial-Time-Series-Soft-Sensor.git
cd Industrial-Time-Series-Soft-Sensor
```

We recommend creating a new conda environment:

```bash
conda create -n induts-ss python=3.10
conda activate induts-ss
```

Install PyTorch with CUDA 12.8 support:

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

If your CUDA version is different, please install the corresponding PyTorch version from the official PyTorch website.

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Citation

If you find this repository useful for your research, please consider citing the relevant dataset papers, model papers, and this benchmark framework. The BibTeX entry for InduTS-SS will be updated after the related paper is available.

<a id="acknowledgements"></a>

## ❤️ Acknowledgements

We gratefully acknowledge the contributions of the open-source community. This codebase has been influenced by and references several repositories, including [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and [PyITS](https://github.com/Master-PLC/PyITS), in the field of deep learning and time-series analysis. We thank the maintainers of these projects for their transparent documentation and modular design, which served as valuable references for structuring our `data`, `models`, and `exp` modules. We are also indebted to the developers of the scientific computing libraries that made this implementation possible.

## License

This project is released under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt) for details.



## Star History

<a href="https://www.star-history.com/?repos=YuAn-06%2FIndustrial-Time-Series-Soft-Sensor&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=YuAn-06/Industrial-Time-Series-Soft-Sensor&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=YuAn-06/Industrial-Time-Series-Soft-Sensor&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=YuAn-06/Industrial-Time-Series-Soft-Sensor&type=date&legend=top-left" />
 </picture>
</a>

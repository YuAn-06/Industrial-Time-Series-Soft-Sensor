# Industrial Time-Series Soft Sensing Library (InduTS-SL)



<div align="center">
  <img src="docs/Logo.jpg#gh-light-mode-only" height=200>
      <h3><b> Industrial Time-Series Soft Sensing Library. </b></h3>
    <p align="center"><i>A Unified Library of Industrial Soft Sensor Models based on Time Series Analysis</i></p>
    </div>






<p align="center">
    <a href="https://www.python.org/">
       <img alt="Python version" src="https://img.shields.io/badge/Python-v3.10+-E97040?logo=python&logoColor=white">
    </a>
    <a href="https://pytorch.org/">
        <img alt="powered by Pytorch" src="https://img.shields.io/badge/PyTorch-v2.7+-E97040?logo=pytorch&logoColor=white">
    </a>
    <a href="https://www.apache.org/licenses/">
        <img alt="Apache License" src="https://img.shields.io/badge/License-Apache2.0-E9BB41%3Flogo%3Dopensourceinitiative%26logoColor%3Dwhite">
    </a>


---



**Industrial Time-Series Soft Sensing Library  (InduTS-SL)** is an **open-source** library specifically designed for researchers working on time series data analysis and soft sensor modeling of complex industrial processes.





## ☀️ The motivation of InduTS-SL

While modern machine learning libraries offer powerful tools for time series modeling, they are not tailored to the specific needs of soft sensor development in industrial processes. Researchers often spend significant effort reimplementing preprocessing pipelines, feature extraction methods, and evaluation protocols from scratch—leading to fragmented codebases and difficulties in fair model comparison. Moreover, many published soft sensor approaches lack open implementations, making reproducibility and benchmarking challenging. 

To bridge this gap, we introduce Time Series Library for Industrial Soft Sensing Modeling: an open-source library that provides a consistent, modular framework for building, training, and evaluating soft sensors on multivariate time series data. By standardizing common components—from input representation to forecasting horizons—InduTS-SL aims to accelerate research, improve comparability, and lower the barrier to entry for new contributors in the field.



## 🧠 What is Soft Sensor?

### **📌**Simple Definition

A **soft sensor** uses easily measurable variables (like temperature, pressure, flow rate) and historical data to **estimate a hard-to-measure target variable** (like product quality, chemical concentration, or viscosity) in real time—using mathematical, machine learning models, deep learning models.



## 💡Data-Driven Soft Sensor Modeling

### **📌** Definition

Due to the increasingly large scale of modern process industries, it has become difficult to obtain mechanistic models of reaction processes or internal process dynamics.  Therefore, data-driven soft sensing models are being widely used by researchers.



Unlike mathematical and mechanistic models, data-driven models utilize process and quality data sampled from industrial processes to construct a black-box model for modeling industrial objects.



Specifically, based on currently published articles on soft sensor modeling, we define the task as follows:

1. **Classical Definition:** Given a length-$T$ sequence $X =[X_1,X_2,...,X_T] \in \mathbb{R}^{T\times C_x}$ including $C_x$ easily measurable process variables, the goal is to build a model that predicts the quality variables  $Y_{T} \in \mathbb{R}^{C_y}$ at time T.
2. **Another Definition:** Given a length-$T$ sequence ${X} =[(X_1,Y_1), (X_2,Y2)...,(X_T,Y_T) ] \in \mathbb{R}^{T\times (C_x+C_y)}$ including $C_x$ easy-measure process variables and $C_y$ quality variables, the goal is to build a model that forecasts the future quality trajectory $Y=[Y_{T+1},Y_{T+2},...,Y_{T+H}]$ over a prediction horizon of $H$ steps.



The first formulation aligns closely with the classical definition of soft sensing: it aims to estimate the current value of one or more quality variables $Y_T$ using only the contemporaneous (or historical) measurements of easily accessible process variables $X$. This setting is typical in real-time monitoring scenarios where physical analyzers are unavailable or too slow, and an instantaneous prediction is required for process control or decision-making.

The second formulation extends this idea to multi-step-ahead forecasting. It assumes that both process variables and quality variables have been jointly observed over a historical window $[1, T]$, and leverages this combined sequence to predict a future trajectory of quality variables $Y = [Y_{T+1}, Y_{T+2}, \ldots, Y_{T+H}]$ over a horizon of H steps. This setup is particularly relevant when the underlying industrial process operates near steady state, where quality measurements—though not continuously available in practice—are assumed to evolve smoothly and can be sampled at regular intervals in the training data. Such a formulation supports applications like predictive quality control, early fault warning, and planning under uncertainty.



### **🧠 How Do They Work?**

1. **Input**:  Real-Time measurements from easy-to-access sensors (e.g., temperature, pressure, flow).
2. **Model**: A trained model that learns the relationship between inputs and the target variable.
3. **Output**: Estimated value of the hard-to-measure variable—updated continuously.



### **🏭 Applications**

- Process monitoring & control
- Quality prediction in manufacturing
- Fault detection and diagnosis
- Digital twins and advanced process control (APC)











## ✨ Available Models for Soft Sensors

We have assigned two abbreviated labels to the two definitions mentioned above, to better indicate which definition each available model is specifically designed to serve, Definition (1) : classical soft sensor (C), Definition: (2) long-short-term forecasting (LSF).



| Models                                                       | Journal/Conference | Type (LSF) | Type (C) | Remark                                            | Status                                                       |
| ------------------------------------------------------------ | ------------------ | ---------- | -------- | ------------------------------------------------- | ------------------------------------------------------------ |
| [iTransformer](https://arxiv.org/abs/2310.06625) (Liu et al) | ICLR 2024          | ✅          |          | A Transformer-based time-series foundation model  |                                                              |
| [PatchTST](https://arxiv.org/abs/2211.14730) (Nie et al)     | ICLR 2023          | ✅          |          | A Transformer-based time-series foundation model  |                                                              |
| [FredFormer](https://arxiv.org/abs/2406.09009) (Piao et al)  | KDD 2024           | ✅          |          | A Transformer-based time-series foundation model  |                                                              |
| [Nonstationary Transformer](https://arxiv.org/abs/2205.14415) (Liu, etal) | NeurlPS2022        | ✅          |          | A Transformer-based time-series foundation model  |                                                              |
| [Autoformer](https://arxiv.org/abs/2106.13008) (Wu et al)    | ICLR 2021          | ✅          |          | A Transformer-based time-series foundation model  |                                                              |
| [DLinear](https://arxiv.org/abs/2205.13504) (Zeng et al)     | AAAI 2023          | ✅          |          | A MLP-based time-series foundation model          |                                                              |
| [Nystroformer](https://arxiv.org/abs/2102.03902) (Xiong et al) | AAAI 2021          | ✅          |          | A Transformer-based time-series foundation model  |                                                              |
| [TCVAE](https://www.ijcai.org/Proceedings/2019/727) (Wang et al) | IJCAI 2019         | ✅          |          | A Transformer-based time-series foundation model  |                                                              |
| [Transformer](https://arxiv.org/abs/1706.03762) (Vaswani et al) | NIPS 2017          | ✅          |          | A Transformer-based time-series foundation model  |                                                              |
| [LDCNN](https://ieeexplore.ieee.org/document/11408874) (Liu et al) | IEEE TC 2026       | ✅          |          | A CNN-based time-series soft sensor model         | The model debugging is not yet complete.                     |
| [VRNN](https://arxiv.org/abs/1506.02216) (Chung et al)       | NIPS 2015          | ✅          | ✅        | A RNN-based time-series soft sensor model         |                                                              |
| [ARDNN](https://ieeexplore.ieee.org/document/11122404) (Chen et al) | IEEE Sensor J 2025 | ✅          |          | A MLP-based time-series soft sensor model         |                                                              |
| [Envformer](https://ieeexplore.ieee.org/document/10699388) (Xie et al) | IEEE TIM 2024      | ✅          |          | A Transformer-based time-series soft sensor model |                                                              |
| [MSACNN ](https://ieeexplore.ieee.org/document/10465636) (Yuan et al) | IEEE TC 2024       |            | ✅        | A CNN-based time-series soft sensor model         | The model debugging is not yet complete.                     |
| [HSAM-dGRUs](https://ieeexplore.ieee.org/abstract/document/10237000) (He et al) | IEEE TASE 2024     |            | ✅        | A RNN-based time-series soft sensor model         |                                                              |
| [CVAE-SMC](https://ieeexplore.ieee.org/document/10264786) (Sun et al) | IEEE TII 2023      | ✅          |          | A VAE-based time series soft sensor model         | SMC sampling for multi-step prediction is not yet available. |
| [DMRIFormer](10.1109/TII.2022.3227731) (Liu et al)           | IEEE TII 2022      | ✅          | ✅        | A Transformer-based time-series soft sensor model |                                                              |
| [DMVAER](https://ieeexplore.ieee.org/document/9797056) (Yao et al) | IEEE TII 2022      |            | ✅        | A DVAE-based time-series soft sensor model        |                                                              |

If you use these models, please cite the relevant articles.



## 📊 Available Datasets

We have provided two classic benchmarks, including **debutanizer column  (DC)** and **sulfur recovery units (SRU)**.  We are very grateful to the providers of these datasets, which are all open-source and included with this book, along with relevant descriptions. If you use these datasets, please cite the relevant articles.



###  🏭 Debutanizer Column  (DC)

The **Debutanizer Column ** is a critical unit in the desulfurization and naphtha splitter plant within petroleum refining processes. Its primary function is to remove propane and butane as overhead products from the naphtha stream. Since the butane content in the debutanizer bottom must be minimized—and direct measurement of this concentration is challenging—an accurate soft sensor for bottom butane concentration is highly valuable for enhancing process control performance.



| Varaibles | Description                 | Type   |
| --------- | --------------------------- | ------ |
| U1        | Top temperature             | Input  |
| U2        | Top pressure                | Input  |
| U3        | Reflux flow                 | Input  |
| U4        | Flow to next process        | Input  |
| U5        | 6th tray temperature        | Input  |
| U6        | Bottom temperature A        | Input  |
| U7        | Bottom temperature B        | Input  |
| Y         | the content of butane on C4 | Output |



###  🏭 Sulfur Recovery Unit (SRU)

The **Sulfur Recovery Unit** is a critical refinery process designed to convert acid gas streams into elemental sulfur. This project focuses on optimizing SRU operational performance, particularly during periods of sensor maintenance.

Operational efficiency is primarily controlled by regulating the air-to-feed ratio. However, hardware sensors for $H_2S$ and $SO_2$ frequently suffer damage due to the corrosive nature of the gases. During routine maintenance or sensor failure, the lack of real-time concentration data significantly degrades the SRU's performance.



| Variables | Description                | Type   |
| --------- | -------------------------- | ------ |
| MEA GAS   | the gas flow in MEA zone   | Input  |
| AIR MEA1  | the air flow in MEA zone 1 | Input  |
| AIR MEA 2 | the air flow in MEA zone   | Input  |
| AIR SWS   | the air flow in SWS zone   | Input  |
| SWS GAS   | the gas flow in SWS zone   | Input  |
| H2S       | the concentrate of H2S     | Output |
| SO2       | the concentrate of SO2     | Output |



### **Notice:** 

If you use these datasets in your paper, please remember to cite the following paper:

**L. Fortuna, S. Graziani, A. Rizzo, and M. G. Xibilia, “Soft sensors for monitoring and control of industrial processes,” Advances in Industrial Control, 2007**

## 📂 Project structure

```bash
├── data/ # datasets
   	|── data_interpolation.py # 
   	|── data_loader.py # Provides data batches and preprocessing functions
   	|── data_provider.py # Choose proper dataloader for each experiements
   	|── DC/
   		|── debutanizer_column.csv
   	|── SRU/
   		|── SRU_data.csv
├── exp/ # datasets
	|── exp_basic.py # defines the basic interface and common methods for experiments
	|── exp_factory.py # creates different experiment instances based on configuration
   	|── exp_short_term_forecasting.py # Implements training, evaluation and testing for LSF tasks 
   	|── exp_soft_sensor.py # Implements training, evaluation and and testing for SS tasks 
   	|── losses.py # Instantiate the loss function for each model
   	|── init__.py
├── layers/ # Attention layers for models 
	|──	SelfAttention_Family.py
	|── Transformer_EncDec.py
	|── DMRIFormer_EncDec.py
	|── NystroAttention.py, etc
├── models/             # Model implementations
│   ├── ARDNN.py
│   ├── Autoformer.py
	├── PatchTST.py
│   └── DMVAER.py, etc
├── scripts/ 
	├── SS task/
		├── DC scripts
		├──	SRU scripts
	├── LSF task/ # long-short term forecasting run scripts
		├── DC scripts # Debutainzer Columns run scripts
		├──	SRU scripts # Sulfur Recovery Unit run scripts
├── utils/              # utilities
	├── configs.py # Parses and processes experiment configuration parameters
	├── dtw.py
	├── dtw_metric.py
	├── ExpConfigs.py # Define variable type
	├── logger.py # Provide Logger definition
	├── masking.py # Masking for different attention
	├── metrics.py # Experiement metrics
	├── print_configs.py # formats and outputs experiment configurations
	├── scaler.py # Provide data normalization classes
	├── timefeatures.py
	├── tools.py # Some functions like Early Stopping, Data Augumentation for SRU and DC dataset and tensorboad et al 
           # run scripts
├── results/            # saved results
└── README.md # Offical documents for introducing this repo
└── requirements.txt # pip dependecy list
└── licenses.txt # pip dependecy list
└── .gitignore.txt # ignore some files when git

```



### Project Structure Overview

This repository is organized into several key directories, each serving a specific purpose in the data processing, model training, and evaluation pipeline.

#### `data/`

This directory is responsible for **dataset management and preprocessing**.

- **Scripts:** It contains `data_loader.py` for providing data batches and preprocessing functions, `data_interpolation.py` for handling missing values, and `data_provider.py` to select the appropriate dataloader for specific experiments.
- **Datasets:** The subdirectories `DC/` and `SRU/` store the raw CSV data files for the Debutanizer Column and Sulfur Recovery Unit processes, respectively.

#### `exp/`

This directory contains the **experimental logic and workflows**.

- **Core Logic:** `exp_basic.py` defines the standard interface and common methods for experiments, while `exp_factory.py` handles the creation of experiment instances based on configuration.
- **Tasks:** Specific implementations for tasks are found in `exp_short_term_forecasting.py` (for Long-Short Term Forecasting) and `exp_soft_sensor.py` (for Soft Sensor tasks).
- **Losses:** The `losses.py` file instantiates the specific loss functions used during model training.

#### `layers/`

This directory houses the **fundamental building blocks for the neural networks**. It includes various attention mechanisms and encoder-decoder architectures, such as `SelfAttention_Family.py`, `Transformer_EncDec.py`, `DMRIFormer_EncDec.py`, and `NystroAttention.py`.

#### `models/`

This directory contains the **implementations of the specific deep learning models** used in the project, such as `ARDNN.py`, `Autoformer.py`, `PatchTST.py`, and `DMVAER.py`.

#### `scripts/`

This directory stores the **execution scripts** used to run experiments. It is organized by task type (`SS task/` and `LSF task/`) and further divided by dataset (`DC scripts` and `SRU scripts`) to facilitate easy reproduction of results.

#### ️ `utils/`

This directory provides **utility functions and helper tools** essential for the project's operation.

- **Configuration:** `configs.py`, `ExpConfigs.py`, and `print_configs.py` handle parameter parsing and formatting.
- **Metrics & Evaluation:** `metrics.py`, `dtw.py`, and `dtw_metric.py` are used for performance evaluation.
- **Tools:** Other utilities include `logger.py` for logging, `masking.py` for attention masks, `scaler.py` for data normalization, and `tools.py` which contains functions for early stopping, data augmentation, and TensorBoard integration.

#### `results/`

This directory is designated for **saving the outputs** of the experiments, including trained models and generated prediction data.

------

> **Note:** The project root also includes a `README.md` for official documentation and a `requirements.txt` file listing the necessary Python dependencies.



## 🚀 Getting Started

We provide two primary entry points to execute the models, catering to different development workflows:

### 1. Configuration-Driven Execution (`run_with_yaml.py`)

**Best for:** Comprehensive experiments.

- **How it works:** All hyperparameters and environment settings are managed via `.yaml` files, facilitating version control and systematic tracking.
- **Usage:** You can run this script directly within your IDE (e.g., **PyCharm**, **VS Code**) or via the terminal.

### 2. Command-Line Execution (`run.py`)

**Best for:** Automated tasks and batch processing in Linux environments (windows also).

- **How it works:** We provide pre-configured `.sh` scripts to streamline the execution process from the command line.

- **Example Command:**

  Bash

  ```
  bash scripts/LSF_task/DC_scripts/iTransformer.sh
  ```

------

### ⚠️ Important Notes

**CUDA Configuration:** Before launching your first run, please ensure your hardware settings are correctly configured in the source code or config files:

- **Enable CUDA:** Verify the `use_gpu` (or equivalent) flag is set correctly.
- **Device Selection:** Specify the appropriate CUDA device index via the `gpu_idx` (default is `0`).





## 🚀Installation

 **To clone the repository locally, run the following command:**



## ❤️Acknowledge

We gratefully acknowledge the contributions of the open-source community. This codebase has been influenced by and references several repositories  ([TSLib](https://github.com/thuml/Time-Series-Library)  and [PYITS](https://github.com/Master-PLC/PyITS)) in the field of deep learning and time series analysis. We thank the maintainers of these projects for their transparent documentation and modular design, which served as a valuable reference for structuring our `data`, `models`, and `exp` modules. We are also indebted to the developers of the core scientific computing libraries that made this implementation possible.




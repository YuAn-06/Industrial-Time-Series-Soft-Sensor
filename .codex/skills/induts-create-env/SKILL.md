---
name: induts-create-env
description: Use when setting up the InduTS-SS benchmark on a new machine, detecting local hardware, choosing the correct PyTorch CPU/CUDA wheel, checking CPU/RAM/GPU/CUDA/driver/conda/pip/uv availability, creating a conda environment, installing packages from requirements.txt, or troubleshooting torch.cuda.is_available() == False, CUDA/driver mismatch, torch import errors, PyTorch wheel errors, or user questions like "which torch version do I need for my GPU?"
---

# InduTS Create Env

## Purpose

Use this skill to prepare a reproducible Python environment for the InduTS-SS benchmark after inspecting the user's hardware. Keep install actions explicit: detect and plan first, then create/install only when the user asks for it or when `--create` is passed.

## Trigger Cases

Use this skill for:

- New-machine benchmark setup where the user needs the correct PyTorch build.
- `torch.cuda.is_available() == False` after installing dependencies.
- CUDA runtime, NVIDIA driver, or PyTorch wheel mismatch.
- `import torch` failures, missing DLL/shared-library errors, or invalid wheel/index errors.
- Questions such as "how do I install dependencies for my GPU?", "which torch version do I need?", or "what should `UV_TORCH_BACKEND` be?"

## Quick Start

Hardware and environment report only:

```bash
python .codex/skills/induts-create-env/scripts/create_env.py --dry-run
```

Create the default conda environment and install `requirements.txt`:

```bash
python .codex/skills/induts-create-env/scripts/create_env.py --create
```

Use a custom environment name:

```bash
python .codex/skills/induts-create-env/scripts/create_env.py --env-name induts-ss-cu128 --create
```

## Workflow

1. Run the script in dry-run mode first.
2. Review the report for OS, Python, conda, pip, CPU, RAM, NVIDIA GPU, driver, and CUDA runtime visibility.
3. Choose the PyTorch backend from the table below.
4. If an NVIDIA GPU is detected and the driver supports CUDA 12.8, prefer the repository's pinned `requirements.txt`, which currently uses PyTorch CUDA 12.8 wheels.
5. If no NVIDIA GPU is detected, warn that `requirements.txt` pins `torch==2.7.1+cu128`; ask before switching to CPU-only PyTorch or editing requirements.
6. If the user confirms installation, run the script with `--create`.
7. After installation, verify with:

```bash
conda run -n induts-ss python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## PyTorch Backend Table

Use this table for PyTorch 2.7.1, matching this repository's dependency style. If upgrading PyTorch, re-check the official PyTorch and NVIDIA tables.

| Detected machine state | Driver threshold | Recommended backend | `UV_TORCH_BACKEND` | PyTorch version guidance |
| --- | --- | --- | --- | --- |
| No NVIDIA GPU, macOS, or CPU-only run | N/A | CPU | `cpu` or `auto` | Use CPU wheels. Nearest upgrade: `torch==2.8.0`; latest known in table: `2.11.0`; do not use repo CUDA requirements unchanged |
| NVIDIA GPU, CUDA 13-capable driver | `>=580` | CUDA 13.0 | `cu130` or `auto` | No repo-pinned 2.7.1 wheel. Nearest official CUDA 13 wheel: `torch==2.9.0`; latest known: `2.11.0` |
| NVIDIA GPU, modern driver | Windows `>=570.65`, Linux `>=570.26` | CUDA 12.8 | `cu128` or `auto` | Repro first: repo default `torch==2.7.1+cu128`; nearest upgrade: `2.8.0`; latest known: `2.11.0` |
| NVIDIA GPU, driver supports CUDA 12.6 but not 12.8 | Windows `>=560.76`, Linux `>=560.28.03` | CUDA 12.6 | `cu126` | Repro-compatible: `torch==2.7.1` with cu126 index; nearest upgrade: `2.8.0`; latest known: `2.11.0` |
| NVIDIA GPU, driver supports CUDA 11.8 but not 12.6 | Windows `>=520.06`, Linux `>=520.61.05` | CUDA 11.8 | `cu118` | Closest official choice: `torch==2.7.1`; newer 2.8+ official wheels no longer list cu118 |
| NVIDIA GPU, older driver | Below CUDA 11.8 threshold | Update driver first or use CPU | `cpu` only as fallback | Prefer driver update before benchmark GPU runs |

`UV_TORCH_BACKEND=auto` is useful when using `uv pip`: uv queries the installed GPU/driver and selects a compatible PyTorch index, falling back to CPU if no supported GPU is found. For this repository's pinned `requirements.txt`, still verify that the selected backend matches the pinned torch local version (`+cu128`, `+cu126`, `+cu118`, or `+cpu`).

## Torch Version Selection

Use `torch==2.7.1` first when reproducibility with this benchmark matters. When the user asks for a close newer version, choose the nearest official wheel above 2.7.1 for the selected backend:

| Backend | Repro-first version | Nearest version >= 2.7.1 | Latest known official table version |
| --- | --- | --- | --- |
| `cpu` | `2.7.1` | `2.8.0` | `2.11.0` |
| `cu118` | `2.7.1` | `2.7.1` | `2.7.1` |
| `cu126` | `2.7.1` | `2.8.0` | `2.11.0` |
| `cu128` | `2.7.1` | `2.8.0` | `2.11.0` |
| `cu130` | not repo-pinned | `2.9.0` | `2.11.0` |

## Script Behavior

`scripts/create_env.py`:

- Detects OS, CPU count, RAM, current Python, `conda`, `pip`, and `nvidia-smi`.
- Parses `requirements.txt` for PyTorch/CUDA pins.
- Detects `uv` and the current `UV_TORCH_BACKEND` environment variable.
- Recommends a PyTorch backend from NVIDIA driver thresholds.
- Writes a Markdown report under `setup_reports/`.
- Prints the exact conda/pip commands it will run.
- Uses `conda create -n <env> python=3.10 -y`.
- Installs packages with `conda run -n <env> python -m pip install -r requirements.txt`.
- Does not modify YAML files or benchmark configs.

## Safety Rules

- Do not run installation commands silently. Use dry-run first unless the user explicitly asks to create the environment.
- Network/package installation may require user approval in restricted environments.
- Do not downgrade or rewrite `requirements.txt` automatically.
- If CUDA install fails, report the failed command and the detected GPU/driver facts before suggesting alternatives.
- On Windows, prefer `conda run -n <env> ...` commands instead of shell activation inside scripts.

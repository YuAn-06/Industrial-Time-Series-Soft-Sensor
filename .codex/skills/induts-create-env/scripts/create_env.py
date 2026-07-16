#!/usr/bin/env python
"""Detect hardware and optionally create the InduTS-SS conda environment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


TORCH_VERSION_CATALOG = {
    "cpu": {
        "repo_pinned": "2.7.1",
        "nearest_ge_2_7_1": "2.8.0",
        "latest_known": "2.11.0",
        "index_url": "https://download.pytorch.org/whl/cpu",
        "note": "CPU wheels are available for all listed versions.",
    },
    "cu118": {
        "repo_pinned": "2.7.1",
        "nearest_ge_2_7_1": "2.7.1",
        "latest_known": "2.7.1",
        "index_url": "https://download.pytorch.org/whl/cu118",
        "note": "PyTorch 2.7.1 is the closest supported choice for CUDA 11.8 in the official version table.",
    },
    "cu126": {
        "repo_pinned": "2.7.1",
        "nearest_ge_2_7_1": "2.8.0",
        "latest_known": "2.11.0",
        "index_url": "https://download.pytorch.org/whl/cu126",
        "note": "CUDA 12.6 wheels are available for 2.7.1 and newer listed versions.",
    },
    "cu128": {
        "repo_pinned": "2.7.1",
        "nearest_ge_2_7_1": "2.8.0",
        "latest_known": "2.11.0",
        "index_url": "https://download.pytorch.org/whl/cu128",
        "note": "This repository pins torch==2.7.1+cu128; use it first for reproducible benchmark runs.",
    },
    "cu130": {
        "repo_pinned": None,
        "nearest_ge_2_7_1": "2.9.0",
        "latest_known": "2.11.0",
        "index_url": "https://download.pytorch.org/whl/cu130",
        "note": "CUDA 13.0 wheels start at PyTorch 2.9.0 in the official version table.",
    },
}

TORCH_COMPANION_VERSIONS = {
    "2.7.1": {"torchvision": "0.22.1", "torchaudio": "2.7.1"},
    "2.8.0": {"torchvision": "0.23.0", "torchaudio": "2.8.0"},
    "2.9.0": {"torchvision": "0.24.0", "torchaudio": "2.9.0"},
    "2.11.0": {"torchvision": "0.26.0", "torchaudio": "2.11.0"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the InduTS-SS benchmark environment.")
    parser.add_argument("--env-name", default="induts-ss", help="Conda environment name.")
    parser.add_argument("--python", default="3.10", help="Python version for conda create.")
    parser.add_argument("--requirements", default="requirements.txt", help="Path to requirements.txt.")
    parser.add_argument("--report-dir", default="setup_reports", help="Directory for setup reports.")
    parser.add_argument("--create", action="store_true", help="Actually create env and install requirements.")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect hardware and print commands.")
    parser.add_argument("--skip-install", action="store_true", help="Create env but skip pip install.")
    parser.add_argument("--force", action="store_true", help="Pass through even if the conda env already exists.")
    return parser.parse_args()


def run_capture(command: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except FileNotFoundError:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "command not found"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "command timed out"}


def detect_memory_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        return None


def detect_nvidia() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    result: dict[str, Any] = {"available": False, "path": nvidia_smi, "gpus": []}
    if not nvidia_smi:
        return result

    overview = run_capture([nvidia_smi])
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", overview.get("stdout", ""))
    smi_cuda_version = cuda_match.group(1) if cuda_match else None

    query = [
        nvidia_smi,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    output = run_capture(query)
    result["raw"] = output
    result["overview"] = overview
    if not output["ok"] or not output["stdout"]:
        return result

    result["available"] = True
    for line in output["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            result["gpus"].append(
                {
                    "name": parts[0],
                    "memory_total": parts[1],
                    "driver_version": parts[2],
                    "cuda_version": smi_cuda_version,
                }
            )
    return result


def conda_info() -> dict[str, Any]:
    conda = shutil.which("conda")
    info: dict[str, Any] = {"available": bool(conda), "path": conda, "envs": []}
    if not conda:
        return info
    info["version"] = run_capture([conda, "--version"]).get("stdout", "")
    env_output = run_capture([conda, "env", "list", "--json"])
    if env_output["ok"] and env_output["stdout"]:
        try:
            info["envs"] = json.loads(env_output["stdout"]).get("envs", [])
        except json.JSONDecodeError:
            info["envs"] = []
    return info


def parse_requirements(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "torch_pin": None,
        "cuda_tag": None,
        "extra_index_urls": [],
    }
    if not path.exists():
        return info

    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--extra-index-url"):
            info["extra_index_urls"].append(stripped.split(maxsplit=1)[-1])
        if stripped.startswith("torch=="):
            info["torch_pin"] = stripped
            match = re.search(r"\+cu(\d+)", stripped)
            if match:
                tag = match.group(1)
                info["cuda_tag"] = f"{tag[:2]}.{tag[2:]}"
    return info


def version_tuple(version: str | None) -> tuple[int, ...]:
    if not version:
        return ()
    return tuple(int(part) for part in re.findall(r"\d+", version)[:4])


def version_gte(version: str | None, minimum: str) -> bool:
    current = version_tuple(version)
    required = version_tuple(minimum)
    if not current:
        return False
    width = max(len(current), len(required))
    current = current + (0,) * (width - len(current))
    required = required + (0,) * (width - len(required))
    return current >= required


def recommend_torch_backend(report: dict[str, Any]) -> dict[str, Any]:
    system = report["platform"]["system"].lower()
    is_windows = system == "windows"
    nvidia = report["nvidia"]
    requirements = report["requirements"]
    current_uv_backend = os.environ.get("UV_TORCH_BACKEND")

    if not nvidia["available"] or not nvidia["gpus"]:
        recommendation = {
            "backend": "cpu",
            "uv_torch_backend": "cpu or auto",
            "reason": "No NVIDIA GPU detected.",
            "matches_requirements": requirements.get("cuda_tag") is None,
            "current_uv_torch_backend": current_uv_backend,
        }
        recommendation["torch_versions"] = recommend_torch_versions(recommendation["backend"])
        return recommendation

    driver_version = nvidia["gpus"][0].get("driver_version")
    thresholds = [
        ("cu130", "13.0", "580.00"),
        ("cu128", "12.8", "570.65" if is_windows else "570.26"),
        ("cu126", "12.6", "560.76" if is_windows else "560.28.03"),
        ("cu118", "11.8", "520.06" if is_windows else "520.61.05"),
    ]
    for backend, cuda_version, minimum_driver in thresholds:
        if version_gte(driver_version, minimum_driver):
            expected_tag = cuda_version
            recommendation = {
                "backend": backend,
                "cuda_version": expected_tag,
                "minimum_driver": minimum_driver,
                "detected_driver": driver_version,
                "uv_torch_backend": f"{backend} or auto",
                "reason": f"NVIDIA driver supports CUDA {cuda_version} PyTorch wheels.",
                "matches_requirements": requirements.get("cuda_tag") == expected_tag,
                "current_uv_torch_backend": current_uv_backend,
            }
            recommendation["torch_versions"] = recommend_torch_versions(backend)
            return recommendation

    recommendation = {
        "backend": "driver-update-or-cpu",
        "uv_torch_backend": "cpu as fallback",
        "detected_driver": driver_version,
        "reason": "NVIDIA driver is below the CUDA 11.8 PyTorch wheel threshold.",
        "matches_requirements": False,
        "current_uv_torch_backend": current_uv_backend,
    }
    recommendation["torch_versions"] = recommend_torch_versions("cpu")
    return recommendation


def recommend_torch_versions(backend: str) -> dict[str, Any]:
    catalog = TORCH_VERSION_CATALOG.get(backend, TORCH_VERSION_CATALOG["cpu"])
    preferred = catalog["repo_pinned"] or catalog["nearest_ge_2_7_1"]
    return {
        **catalog,
        "preferred_for_benchmark": preferred,
        "preferred_install": build_torch_install_command(preferred, backend if backend in TORCH_VERSION_CATALOG else "cpu"),
        "nearest_install": build_torch_install_command(catalog["nearest_ge_2_7_1"], backend if backend in TORCH_VERSION_CATALOG else "cpu"),
        "latest_install": build_torch_install_command(catalog["latest_known"], backend if backend in TORCH_VERSION_CATALOG else "cpu"),
    }


def build_torch_install_command(version: str | None, backend: str) -> str | None:
    if not version:
        return None
    index_url = TORCH_VERSION_CATALOG[backend]["index_url"]
    companions = TORCH_COMPANION_VERSIONS[version]
    return (
        f"python -m pip install torch=={version} "
        f"torchvision=={companions['torchvision']} "
        f"torchaudio=={companions['torchaudio']} "
        f"--index-url {index_url}"
    )


def detect_hardware(requirements: Path) -> dict[str, Any]:
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
        },
        "cpu": {
            "logical_cores": os.cpu_count(),
            "ram_gb": detect_memory_gb(),
        },
        "conda": conda_info(),
        "pip": {
            "path": shutil.which("pip"),
            "version": run_capture([sys.executable, "-m", "pip", "--version"]).get("stdout", ""),
        },
        "uv": {
            "path": shutil.which("uv"),
            "version": run_capture([shutil.which("uv") or "uv", "--version"]).get("stdout", ""),
            "UV_TORCH_BACKEND": os.environ.get("UV_TORCH_BACKEND"),
        },
        "nvidia": detect_nvidia(),
        "requirements": parse_requirements(requirements),
    }
    report["torch_backend_recommendation"] = recommend_torch_backend(report)
    return report


def env_exists(conda: dict[str, Any], env_name: str) -> bool:
    suffixes = {Path(env).name for env in conda.get("envs", [])}
    return env_name in suffixes


def build_commands(args: argparse.Namespace, report: dict[str, Any]) -> list[list[str]]:
    conda_path = report["conda"].get("path") or "conda"
    commands = [[conda_path, "create", "-n", args.env_name, f"python={args.python}", "-y"]]
    if not args.skip_install:
        commands.append(
            [
                conda_path,
                "run",
                "-n",
                args.env_name,
                "python",
                "-m",
                "pip",
                "install",
                "-r",
                args.requirements,
            ]
        )
    commands.append(
        [
            conda_path,
            "run",
            "-n",
            args.env_name,
            "python",
            "-c",
            "import torch; print(torch.__version__, torch.cuda.is_available())",
        ]
    )
    return commands


def command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def write_report(path: Path, report: dict[str, Any], commands: list[list[str]], args: argparse.Namespace) -> None:
    nvidia = report["nvidia"]
    req = report["requirements"]
    backend = report["torch_backend_recommendation"]
    torch_versions = backend.get("torch_versions", {})
    lines = [
        "# InduTS-SS Environment Setup Report",
        "",
        f"- Timestamp: `{report['timestamp']}`",
        f"- OS: `{report['platform']['system']} {report['platform']['release']}`",
        f"- Machine: `{report['platform']['machine']}`",
        f"- CPU logical cores: `{report['cpu']['logical_cores']}`",
        f"- RAM GB: `{report['cpu']['ram_gb']}`",
        f"- Current Python: `{report['python']['version']}`",
        f"- Conda available: `{report['conda']['available']}`",
        f"- Conda version: `{report['conda'].get('version', '')}`",
        f"- uv available: `{bool(report['uv'].get('path'))}`",
        f"- uv version: `{report['uv'].get('version', '')}`",
        f"- Current `UV_TORCH_BACKEND`: `{report['uv'].get('UV_TORCH_BACKEND')}`",
        f"- NVIDIA GPU available: `{nvidia['available']}`",
        f"- Requirements: `{req['path']}`",
        f"- Torch pin: `{req.get('torch_pin')}`",
        f"- CUDA tag from requirements: `{req.get('cuda_tag')}`",
        f"- Recommended backend: `{backend.get('backend')}`",
        f"- Recommended `UV_TORCH_BACKEND`: `{backend.get('uv_torch_backend')}`",
        f"- Backend matches requirements: `{backend.get('matches_requirements')}`",
        f"- Preferred torch for benchmark: `{torch_versions.get('preferred_for_benchmark')}`",
        f"- Nearest torch >= 2.7.1 for backend: `{torch_versions.get('nearest_ge_2_7_1')}`",
        f"- Latest known torch for backend: `{torch_versions.get('latest_known')}`",
        "",
    ]
    if nvidia["gpus"]:
        lines.extend(["## GPUs", ""])
        for gpu in nvidia["gpus"]:
            lines.append(
                f"- `{gpu['name']}` memory={gpu['memory_total']}, driver={gpu['driver_version']}, cuda={gpu['cuda_version']}"
            )
        lines.append("")

    lines.extend(
        [
            "## PyTorch Backend Decision",
            "",
            f"- Reason: {backend.get('reason')}",
            f"- Detected driver: `{backend.get('detected_driver')}`",
            f"- Minimum driver for recommendation: `{backend.get('minimum_driver')}`",
            "",
            "| Machine state | Driver threshold | Backend | `UV_TORCH_BACKEND` | Notes |",
            "| --- | --- | --- | --- | --- |",
            "| CPU-only / no NVIDIA GPU | N/A | CPU | `cpu` or `auto` | Use CPU index; do not keep CUDA torch pin unchanged |",
            "| NVIDIA, CUDA 13.0-capable driver | `>=580` | CUDA 13.0 | `cu130` or `auto` | Requires PyTorch >= 2.9.0; not repo-pinned |",
            "| NVIDIA, CUDA 12.8-capable driver | Windows `>=570.65`, Linux `>=570.26` | CUDA 12.8 | `cu128` or `auto` | Matches repo default `torch==2.7.1+cu128` |",
            "| NVIDIA, CUDA 12.6-capable driver | Windows `>=560.76`, Linux `>=560.28.03` | CUDA 12.6 | `cu126` | Requires CUDA 12.6 requirements variant |",
            "| NVIDIA, CUDA 11.8-capable driver | Windows `>=520.06`, Linux `>=520.61.05` | CUDA 11.8 | `cu118` | Requires CUDA 11.8 requirements variant |",
            "| Older NVIDIA driver | Below CUDA 11.8 threshold | Update driver or CPU | `cpu` fallback | Update driver before GPU benchmark runs |",
            "",
        ]
    )

    lines.extend(
        [
            "## Torch Version Recommendation",
            "",
            "| Backend | Preferred for benchmark | Nearest >= 2.7.1 | Latest known | Index |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for catalog_backend, item in TORCH_VERSION_CATALOG.items():
        preferred = item["repo_pinned"] or item["nearest_ge_2_7_1"]
        marker = " (selected)" if catalog_backend == backend.get("backend") else ""
        lines.append(
            f"| `{catalog_backend}`{marker} | `{preferred}` | `{item['nearest_ge_2_7_1']}` | `{item['latest_known']}` | `{item['index_url']}` |"
        )
    lines.extend(
        [
            "",
            f"Selected backend note: {torch_versions.get('note')}",
            "",
            "Suggested PyTorch-only install commands, useful when repairing a torch wheel mismatch:",
            "",
            f"```bash\n{torch_versions.get('preferred_install')}\n```",
            "",
            f"```bash\n{torch_versions.get('nearest_install')}\n```",
            "",
            f"```bash\n{torch_versions.get('latest_install')}\n```",
            "",
        ]
    )

    lines.extend(["## Recommended Commands", ""])
    for command in commands:
        lines.append(f"```bash\n{command_text(command)}\n```")
        lines.append("")

    lines.extend(["## Notes", ""])
    if not report["conda"]["available"]:
        lines.append("- Conda was not detected. Install Anaconda/Miniconda before running `--create`.")
    if not nvidia["available"] and req.get("cuda_tag"):
        lines.append(
            "- No NVIDIA GPU was detected, but requirements pin CUDA PyTorch. Confirm before installing or prepare a CPU-only requirements variant."
        )
    if backend.get("current_uv_torch_backend") and backend.get("backend") not in str(backend.get("current_uv_torch_backend")):
        lines.append(
            f"- Current `UV_TORCH_BACKEND={backend.get('current_uv_torch_backend')}` may not match the recommended backend `{backend.get('backend')}`."
        )
    if not backend.get("matches_requirements"):
        lines.append(
            "- Recommended backend does not match the pinned requirements. Do not install blindly; choose a matching torch wheel or update the driver."
        )
    if nvidia["available"] and req.get("cuda_tag") and backend.get("matches_requirements"):
        lines.append("- NVIDIA GPU detected. The repository CUDA PyTorch requirements are appropriate to try first.")
    elif nvidia["available"] and req.get("cuda_tag"):
        lines.append(
            "- NVIDIA GPU detected, but the pinned CUDA PyTorch requirement does not match this driver. Use the recommended backend or update the driver."
        )
    if args.dry_run or not args.create:
        lines.append("- Dry-run only: no conda environment or packages were installed.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute_commands(commands: list[list[str]]) -> int:
    for command in commands:
        print(f"Running: {command_text(command)}")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            print(f"Command failed with exit code {completed.returncode}: {command_text(command)}")
            return completed.returncode
    return 0


def main() -> int:
    args = parse_args()
    requirements = Path(args.requirements)
    report = detect_hardware(requirements)
    commands = build_commands(args, report)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"create_env_{args.env_name}.md"
    write_report(report_path, report, commands, args)

    print(f"Saved setup report: {report_path}")
    print("Detected hardware summary:")
    print(f"  OS: {report['platform']['system']} {report['platform']['release']}")
    print(f"  CPU logical cores: {report['cpu']['logical_cores']}")
    print(f"  RAM GB: {report['cpu']['ram_gb']}")
    print(f"  NVIDIA GPU available: {report['nvidia']['available']}")
    print(f"  uv available: {bool(report['uv'].get('path'))}")
    print(f"  UV_TORCH_BACKEND: {report['uv'].get('UV_TORCH_BACKEND')}")
    print(f"  Recommended PyTorch backend: {report['torch_backend_recommendation'].get('backend')}")
    print(
        "  Preferred torch for benchmark: "
        f"{report['torch_backend_recommendation'].get('torch_versions', {}).get('preferred_for_benchmark')}"
    )
    for gpu in report["nvidia"]["gpus"]:
        print(f"  GPU: {gpu['name']} ({gpu['memory_total']}), driver {gpu['driver_version']}, CUDA {gpu['cuda_version']}")
    print("Planned commands:")
    for command in commands:
        print(f"  {command_text(command)}")

    if args.dry_run or not args.create:
        return 0
    if not report["conda"]["available"]:
        print("Conda is not available; cannot create environment.")
        return 2
    if env_exists(report["conda"], args.env_name) and not args.force:
        print(f"Conda environment '{args.env_name}' already exists. Pass --force to continue.")
        return 3
    return execute_commands(commands)


if __name__ == "__main__":
    raise SystemExit(main())

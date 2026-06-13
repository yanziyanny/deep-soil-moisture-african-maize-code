#!/usr/bin/env python3

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "training" / "outputs" / "benchmark_runtime.json"
sys.dont_write_bytecode = True
PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "geopandas",
    "pyogrio",
    "shapely",
    "linearmodels",
    "scikit-learn",
    "xgboost",
    "shap",
    "PyYAML",
]


def package_versions():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def memory_gb():
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        return None


def run_command(command):
    start = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, env=env)
    elapsed = time.perf_counter() - start
    stdout_tail = completed.stdout[-2000:].replace(str(REPO_ROOT), "<repo>")
    stderr_tail = completed.stderr[-2000:].replace(str(REPO_ROOT), "<repo>")
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "runtime_seconds": elapsed,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def main():
    parser = argparse.ArgumentParser(description="Record computational-resource and runtime information.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to write JSON benchmark report")
    parser.add_argument("--skip-run-all-figures", action="store_true", help="Do not time the full figure reproduction command")
    parser.add_argument("--smoke-figures", action="store_true", help="Time run_all_figures.py --smoke instead of full rendering")
    parser.add_argument("--skip-quick-retraining", action="store_true", help="Do not time optional ML quick retraining")
    args = parser.parse_args()

    report = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": memory_gb(),
        "package_versions": package_versions(),
        "commands": [],
    }

    if not args.skip_run_all_figures:
        command = [sys.executable, "run_all_figures.py"]
        if args.smoke_figures:
            command.append("--smoke")
        report["commands"].append(run_command(command))

    if not args.skip_quick_retraining:
        report["commands"].append(run_command([sys.executable, "training/run_optional_ml_retraining.py", "--quick"]))

    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {output.relative_to(REPO_ROOT) if output.is_relative_to(REPO_ROOT) else output}")


if __name__ == "__main__":
    main()

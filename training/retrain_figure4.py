#!/usr/bin/env python3

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import shutil
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from figure_manifest import ML_RETRAIN_TARGETS, package_path


TARGET = ML_RETRAIN_TARGETS["figure4"]


def sync_outputs(temp_output_dir: Path):
    source_map = {
        temp_output_dir / "summary.csv": "figure4/data/summary.csv",
        temp_output_dir / "koppen1" / "results.json": "figure4/data/koppen1/results.json",
        temp_output_dir / "koppen2" / "results.json": "figure4/data/koppen2/results.json",
        temp_output_dir / "koppen3" / "results.json": "figure4/data/koppen3/results.json",
        temp_output_dir / "koppen4" / "results.json": "figure4/data/koppen4/results.json",
        temp_output_dir / "koppen5" / "results.json": "figure4/data/koppen5/results.json",
    }

    for source, relative_dest in source_map.items():
        destination = package_path(relative_dest)
        if not source.exists():
            raise FileNotFoundError(f"Missing retrained Figure 4 artifact: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"[retrain_figure4] Synced {source} -> {destination}")


def main():
    print("[retrain_figure4] Running upstream Figure 4 ML retraining...")
    with TemporaryDirectory(prefix="nature_food_fig4_") as temp_dir:
        temp_output_dir = Path(temp_dir)
        packaged_input = package_path(TARGET["packaged_training_input"])
        if not packaged_input.exists():
            raise FileNotFoundError(f"Missing packaged Figure 4 retraining input: {packaged_input}")
        env = os.environ.copy()
        env["AFRICA_SIF_KOPPEN_BOOTSTRAP_OUTPUT_DIR"] = str(temp_output_dir)
        env["AFRICA_SIF_KOPPEN_BOOTSTRAP_DATA_FILE"] = str(packaged_input)
        subprocess.run([sys.executable, TARGET["upstream_script"]], check=True, cwd=REPO_ROOT, env=env)
        sync_outputs(temp_output_dir)
    print("[retrain_figure4] Done.")


if __name__ == "__main__":
    main()

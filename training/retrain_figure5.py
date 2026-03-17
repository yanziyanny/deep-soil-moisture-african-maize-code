#!/usr/bin/env python3

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import shutil
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nature_food_code_availability.figure_manifest import ML_RETRAIN_TARGETS, package_path


TARGET = ML_RETRAIN_TARGETS["figure5"]


def sync_outputs(temp_monitoring_dir: Path):
    source_map = {
        temp_monitoring_dir / "monitoring_blind_spots_data.csv": "figure5/data/monitoring_blind_spots_data.csv",
    }

    for source, relative_dest in source_map.items():
        destination = package_path(relative_dest)
        if not source.exists():
            raise FileNotFoundError(f"Missing refreshed Figure 5 artifact: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"[retrain_figure5] Synced {source} -> {destination}")


def main():
    print("[retrain_figure5] Running upstream Figure 5 ML driver retraining...")
    with TemporaryDirectory(prefix="nature_food_fig5_") as temp_dir:
        temp_root = Path(temp_dir)
        driver_output_dir = temp_root / "driver_map"
        monitoring_output_dir = temp_root / "monitoring_blind_spots"
        driver_output_dir.mkdir(parents=True, exist_ok=True)
        monitoring_output_dir.mkdir(parents=True, exist_ok=True)

        driver_env = os.environ.copy()
        driver_env["AFRICA_SIF_ADMIN2_DRIVER_OUTPUT_DIR"] = str(driver_output_dir)
        subprocess.run([sys.executable, TARGET["upstream_script"]], check=True, cwd=REPO_ROOT, env=driver_env)

        print("[retrain_figure5] Refreshing downstream monitoring-blind-spots table...")
        monitoring_env = os.environ.copy()
        monitoring_env["MONITORING_DRIVER_FILE"] = str(driver_output_dir / "admin2_driver_dominance.csv")
        monitoring_env["MONITORING_OUTPUT_DIR"] = str(monitoring_output_dir)
        subprocess.run([sys.executable, TARGET["downstream_refresh_script"]], check=True, cwd=REPO_ROOT, env=monitoring_env)

        sync_outputs(monitoring_output_dir)
    print("[retrain_figure5] Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure_manifest import FIGURES, package_path


def smoke_check():
    for figure in FIGURES:
        script = package_path(figure["script"])
        if not script.exists():
            raise FileNotFoundError(f"Missing figure script: {script}")
        spec = importlib.util.spec_from_file_location(f"{figure['name']}_smoke", script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import {script}")
        module = importlib.util.module_from_spec(spec)
        script_dir = str(script.parent)
        inserted = False
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
            inserted = True
        try:
            spec.loader.exec_module(module)
        finally:
            if inserted:
                sys.path.remove(script_dir)
        output = package_path(figure["output"])
        print(f"[run_all_figures] Smoke checked {script.relative_to(ROOT)} -> {output.relative_to(ROOT)}")
    print("[run_all_figures] Smoke check done.")


def main():
    parser = argparse.ArgumentParser(description="Run all main-text figure scripts.")
    parser.add_argument("--smoke", action="store_true", help="import figure modules and verify expected paths without rendering")
    args = parser.parse_args()

    if args.smoke:
        smoke_check()
        return

    for figure in FIGURES:
        script = package_path(figure["script"])
        output = package_path(figure["output"])
        print(f"[run_all_figures] Running {script.relative_to(ROOT)}")
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT, env=env)
        print(f"[run_all_figures] Wrote {output.relative_to(ROOT)}")
    print("[run_all_figures] Done.")


if __name__ == "__main__":
    main()

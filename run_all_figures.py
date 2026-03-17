#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from nature_food_code_availability.figure_manifest import FIGURES, package_path


def main():
    for figure in FIGURES:
        script = package_path(figure["script"])
        output = package_path(figure["output"])
        print(f"[run_all_figures] Running {script.relative_to(ROOT)}")
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)
        print(f"[run_all_figures] Wrote {output.relative_to(ROOT)}")
    print("[run_all_figures] Done.")


if __name__ == "__main__":
    main()

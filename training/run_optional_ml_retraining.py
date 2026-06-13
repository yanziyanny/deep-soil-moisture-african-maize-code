#!/usr/bin/env python3

from pathlib import Path
import sys


sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.retrain_figure4 import main as retrain_figure4_main


def main():
    retrain_figure4_main(sys.argv[1:])


if __name__ == "__main__":
    main()

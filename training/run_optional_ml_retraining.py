#!/usr/bin/env python3

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from retrain_figure4 import main as retrain_figure4_main


def main():
    retrain_figure4_main()


if __name__ == "__main__":
    main()

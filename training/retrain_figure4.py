#!/usr/bin/env python3

from pathlib import Path
import sys


sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.ml_pipeline.figure4 import build_arg_parser, run_from_args


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()

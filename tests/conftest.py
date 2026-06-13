import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def pytest_sessionfinish(session, exitstatus):
    for path in REPO_ROOT.rglob("__pycache__"):
        shutil.rmtree(path, ignore_errors=True)

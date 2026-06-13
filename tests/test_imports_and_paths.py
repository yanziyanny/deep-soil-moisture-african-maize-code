import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_core_modules_importable():
    for module_name in [
        "figure_manifest",
        "run_all_figures",
        "training.retrain_figure4",
        "training.ml_pipeline.figure4",
    ]:
        importlib.import_module(module_name)


def test_no_external_sibling_or_absolute_local_paths_in_text_files():
    slash = chr(47)
    blocked = [
        "".join(["africa_sif_", "cluster80_cvmedian"]),
        slash + "Users" + slash,
        slash + "Volumes" + slash,
        "".join(["REPO_ROOT", ".parent"]),
    ]
    text_suffixes = {
        ".py",
        ".md",
        ".yml",
        ".yaml",
        ".txt",
        ".json",
        ".toml",
        ".cfg",
        ".ini",
        ".sh",
    }
    offenders = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts or "outputs" in path.parts:
            continue
        if path.suffix not in text_suffixes and path.name not in {"LICENSE"}:
            continue
        text = path.read_text(errors="ignore")
        for pattern in blocked:
            if pattern in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains {pattern}")
    assert not offenders, "\n".join(offenders)

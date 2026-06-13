import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OUTPUTS = [
    "model_metrics_by_zone.csv",
    "drop_column_importance_individual.csv",
    "drop_column_importance_group.csv",
    "bootstrap_confidence_intervals.csv",
    "hard_energy_filtering_sensitivity.csv",
    "run_metadata.json",
    "train_test_split_ids.csv",
    "groupkfold_fold_ids.csv",
    "figure4_data/summary.csv",
    "figure4_data/koppen1/results.json",
    "figure4_data/koppen2/results.json",
    "figure4_data/koppen3/results.json",
    "figure4_data/koppen4/results.json",
    "figure4_data/koppen5/results.json",
]


def test_packaged_ml_input_exists_and_has_rows():
    path = REPO_ROOT / "training/data/figure4_retraining_input.csv.gz"
    assert path.exists()
    assert path.stat().st_size > 0
    sample = pd.read_csv(path, nrows=1)
    assert len(sample) == 1
    assert {"VPD_8mean_raw", "PPTa_8sum"}.issubset(sample.columns)
    assert sample["VPD_8mean_raw"].notna().all()


def test_quick_retraining_outputs_and_leakage_checks(tmp_path):
    output_dir = tmp_path / "ml_outputs"
    subprocess.run(
        [
            sys.executable,
            "training/run_optional_ml_retraining.py",
            "--quick",
            "--outputs-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    for relative in EXPECTED_OUTPUTS:
        assert (output_dir / relative).exists(), relative

    split = pd.read_csv(output_dir / "train_test_split_ids.csv")
    fold = pd.read_csv(output_dir / "groupkfold_fold_ids.csv")
    assert split.groupby("group_id")["split"].nunique().max() == 1
    validation = fold.dropna(subset=["validation_fold"])
    assert validation.groupby("group_id")["validation_fold"].nunique().max() == 1
    assert set(split["split"]) == {"train", "test"}

    hard_filter = pd.read_csv(output_dir / "hard_energy_filtering_sensitivity.csv")
    assert "computed" in set(hard_filter["status"])
    assert hard_filter["reason"].str.contains("VPD_8mean_raw").any()

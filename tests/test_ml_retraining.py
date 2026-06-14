import subprocess
import sys
import json
from pathlib import Path

import pandas as pd

from training.ml_pipeline.figure4 import load_config, prepare_data, split_grouped_stratified_by_year


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OUTPUTS = [
    "model_metrics_by_zone.csv",
    "drop_column_importance_individual.csv",
    "drop_column_importance_group.csv",
    "bootstrap_confidence_intervals.csv",
    "s8_model_predictions.csv",
    "hard_energy_filtering_sensitivity.csv",
    "hard_energy_filtering/summary.csv",
    "hard_energy_filtering/figure_dropcol_combined_nature_v3.png",
    "hard_energy_filtering/koppen1/results.json",
    "hard_energy_filtering/koppen2/results.json",
    "hard_energy_filtering/koppen3/results.json",
    "hard_energy_filtering/koppen4/results.json",
    "hard_energy_filtering/koppen5/results.json",
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
    assert {"test_r2", "delta_energy", "delta_surface", "delta_rootwater", "sw_threshold", "tmax_threshold"}.issubset(hard_filter.columns)
    assert set(hard_filter["sw_threshold"]) == {21.6}
    assert set(hard_filter["tmax_threshold"]) == {18.0}
    assert "vpd_threshold" not in {column.lower() for column in hard_filter.columns}

    predictions = pd.read_csv(output_dir / "s8_model_predictions.csv")
    assert {"koppen_id", "observed_sif_anom", "predicted_sif_anom", "group_id"}.issubset(predictions.columns)
    assert len(predictions) > 0

    with (output_dir / "hard_energy_filtering/koppen1/results.json").open("r") as handle:
        hard_result = json.load(handle)
    assert {"drop_column_bootstrap", "drop_group_bootstrap"}.issubset(hard_result)
    assert hard_result["sw_threshold"] == 21.6
    assert hard_result["tmax_threshold"] == 18


def test_quick_s9_shapley_retraining_outputs(tmp_path):
    output_dir = tmp_path / "s9_outputs"
    subprocess.run(
        [
            sys.executable,
            "training/run_s9_shapley_r2.py",
            "--quick",
            "--zones",
            "1",
            "--bootstrap-iters",
            "3",
            "--outputs-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    for relative in [
        "summary.csv",
        "s9_shapley_group_decomposition.csv",
        "koppen1/results.json",
    ]:
        assert (output_dir / relative).exists(), relative
    s9 = pd.read_csv(output_dir / "s9_shapley_group_decomposition.csv")
    check = s9.groupby("koppen_id").agg(
        full_model_r2=("full_model_r2", "first"),
        shapley_sum=("shapley_r2", "sum"),
    )
    assert (check["full_model_r2"] - check["shapley_sum"]).abs().max() < 1e-9


def test_figure4_split_matches_packaged_results():
    config = load_config(REPO_ROOT / "training/config.yml")
    prepared = prepare_data(config, REPO_ROOT / config["data"]["input_file"])
    zone_col = config["data"]["climate_zone_variable"]
    group_col = config["data"]["grouping_variable"]
    year_col = config["data"]["year_stratification_variable"]
    for zone_id in range(1, 6):
        zone = prepared.frame[prepared.frame[zone_col] == zone_id].copy()
        train_mask, test_mask, _ = split_grouped_stratified_by_year(
            zone,
            group_col,
            year_col,
            float(config["test_size"]),
            int(config["random_seed"]),
        )
        with (REPO_ROOT / f"figure4/data/koppen{zone_id}/results.json").open("r") as handle:
            packaged = json.load(handle)
        assert int(train_mask.sum()) == packaged["n_train"]
        assert int(test_mask.sum()) == packaged["n_test"]

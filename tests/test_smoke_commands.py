import subprocess
import sys
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_all_figures_smoke_imports():
    subprocess.run([sys.executable, "run_all_figures.py", "--smoke"], cwd=REPO_ROOT, check=True)


def test_supplement_quick_runs():
    subprocess.run([sys.executable, "supplement/run_all_supplement.py", "--quick"], cwd=REPO_ROOT, check=True)
    report_path = REPO_ROOT / "supplement/outputs/supplement_run_report.json"
    with report_path.open("r") as handle:
        report = json.load(handle)
    status_by_item = {item["item"]: item["status"] for item in report["items"]}
    for item in [
        "S3 GLEAM SM-VPD coupling",
        "S4 GLDAS Noah SM-VPD coupling",
        "S5 GLDAS Noah nonlinear yield response",
        "S6 GLDAS Noah yield sensitivity",
        "S7 GLDAS Noah SIF attribution",
    ]:
        assert status_by_item[item] == "written"
    for relative in [
        "supplement/outputs/s3_gleam_sm_vpd_coupling_summary.csv",
        "supplement/outputs/s4_gldas_sm_vpd_coupling_summary.csv",
        "supplement/outputs/s5_gldas_yield_response_coefficients.csv",
        "supplement/outputs/s6_gldas_yield_sensitivity_r2.csv",
        "supplement/outputs/s7_gldas_sif_attribution_summary.csv",
        "supplement/outputs/s9_shapley_group_decomposition.csv",
        "supplement/outputs/s10_hard_energy_filtering_sensitivity.csv",
    ]:
        assert (REPO_ROOT / relative).exists(), relative

    s9 = pd.read_csv(REPO_ROOT / "supplement/outputs/s9_shapley_group_decomposition.csv")
    s9_check = s9.groupby("koppen_id").agg(
        full_model_r2=("full_model_r2", "first"),
        shapley_sum=("shapley_r2", "sum"),
    )
    assert (s9_check["full_model_r2"] - s9_check["shapley_sum"]).abs().max() < 1e-9

    s10 = pd.read_csv(REPO_ROOT / "supplement/outputs/s10_hard_energy_filtering_sensitivity.csv")
    assert set(s10["sw_threshold"]) == {21.6}
    assert set(s10["tmax_threshold"]) == {18.0}
    assert not any("vpd" in column.lower() and "threshold" in column.lower() for column in s10.columns)

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
        "S8 SIF predicted vs observed",
        "S9 Shapley R2 decomposition",
        "S10 hard energy filtering",
    ]:
        assert status_by_item[item] == "written"
    for relative in [
        "supplement/outputs/supplementary_figure_s3_gleam_sm_vpd_correlation.png",
        "supplement/outputs/supplementary_figure_s3_gleam_sm_vpd_correlation.pdf",
        "supplement/outputs/supplementary_figure_s4_gldas_sm_vpd_correlation.png",
        "supplement/outputs/supplementary_figure_s4_gldas_sm_vpd_correlation.pdf",
        "supplement/outputs/supplementary_figure_s5_gldas_yield_response.png",
        "supplement/outputs/supplementary_figure_s5_gldas_yield_response.pdf",
        "supplement/outputs/supplementary_figure_s6_gldas_yield_sensitivity.png",
        "supplement/outputs/supplementary_figure_s6_gldas_yield_sensitivity.pdf",
        "supplement/outputs/supplementary_figure_s7_gldas_sif_attribution.png",
        "supplement/outputs/supplementary_figure_s7_gldas_sif_attribution.pdf",
        "supplement/outputs/supplementary_figure_s8_sif_pred_vs_obs.png",
        "supplement/outputs/supplementary_figure_s8_sif_pred_vs_obs.pdf",
        "supplement/outputs/supplementary_figure_s9_shapley_r2_decomposition.png",
        "supplement/outputs/supplementary_figure_s9_shapley_r2_decomposition.pdf",
        "supplement/outputs/supplementary_figure_s10_hard_energy_filtering.png",
        "supplement/outputs/supplementary_figure_s10_hard_energy_filtering.pdf",
    ]:
        assert (REPO_ROOT / relative).exists(), relative

    s8 = pd.read_csv(REPO_ROOT / "supplement/data/sif_predictions/s8_sif_pred_vs_obs.csv.gz")
    figure4_summary = pd.read_csv(REPO_ROOT / "figure4/data/summary.csv").set_index("koppen_id")
    for koppen_id, zone in s8.groupby("koppen_id"):
        weights = zone["sample_weight"]
        observed = zone["observed_sif_anom"]
        predicted = zone["predicted_sif_anom"]
        observed_mean = (weights * observed).sum() / weights.sum()
        ss_res = (weights * (observed - predicted) ** 2).sum()
        ss_tot = (weights * (observed - observed_mean) ** 2).sum()
        weighted_r2 = 1 - ss_res / ss_tot
        assert abs(weighted_r2 - figure4_summary.loc[koppen_id, "test_r2"]) < 1e-9

    s9 = pd.read_csv(REPO_ROOT / "supplement/data/shapley_r2/s9_shapley_group_decomposition.csv")
    s9_check = s9.groupby("koppen_id").agg(
        full_model_r2=("full_model_r2", "first"),
        shapley_sum=("shapley_r2", "sum"),
    )
    assert (s9_check["full_model_r2"] - s9_check["shapley_sum"]).abs().max() < 1e-9

    s10 = pd.read_csv(REPO_ROOT / "supplement/data/hard_energy_filtering/s10_hard_energy_filtering_sensitivity.csv")
    assert set(s10["sw_threshold"]) == {21.6}
    assert set(s10["tmax_threshold"]) == {18.0}
    assert not any("vpd" in column.lower() and "threshold" in column.lower() for column in s10.columns)

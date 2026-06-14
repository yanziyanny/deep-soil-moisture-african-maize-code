#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "supplement" / "data"
FIGURE_DIR = DATA_DIR / "figures"
OUTPUT_DIR = REPO_ROOT / "supplement" / "outputs"


FIGURE_SPECS = [
    (
        "S3 GLEAM SM-VPD coupling",
        [
            "supplementary_figure_s3_gleam_sm_vpd_correlation.png",
            "supplementary_figure_s3_gleam_sm_vpd_correlation.pdf",
        ],
    ),
    (
        "S4 GLDAS Noah SM-VPD coupling",
        [
            "supplementary_figure_s4_gldas_sm_vpd_correlation.png",
            "supplementary_figure_s4_gldas_sm_vpd_correlation.pdf",
        ],
    ),
    (
        "S5 GLDAS Noah nonlinear yield response",
        [
            "supplementary_figure_s5_gldas_yield_response.png",
            "supplementary_figure_s5_gldas_yield_response.pdf",
        ],
    ),
    (
        "S6 GLDAS Noah yield sensitivity",
        [
            "supplementary_figure_s6_gldas_yield_sensitivity.png",
            "supplementary_figure_s6_gldas_yield_sensitivity.pdf",
        ],
    ),
    (
        "S7 GLDAS Noah SIF attribution",
        [
            "supplementary_figure_s7_gldas_sif_attribution.png",
            "supplementary_figure_s7_gldas_sif_attribution.pdf",
        ],
    ),
    (
        "S8 XGBoost model performance",
        [
            "supplementary_figure_s8_sif_pred_vs_obs.png",
        ],
    ),
    (
        "S9 Shapley R2 decomposition",
        [
            "supplementary_figure_s9_shapley_r2_decomposition.png",
            "supplementary_figure_s9_shapley_r2_decomposition.pdf",
        ],
    ),
    (
        "S10 hard energy filtering",
        [
            "supplementary_figure_s10_hard_energy_filtering.png",
            "supplementary_figure_s10_hard_energy_filtering.pdf",
        ],
    ),
]


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def copy_processed_figure(source_name: str, output_dir: Path) -> Path:
    source = FIGURE_DIR / source_name
    if not source.exists():
        raise FileNotFoundError(f"Missing packaged supplement figure: {source}")
    destination = output_dir / source_name
    shutil.copy2(source, destination)
    return destination


def validate_s9_source() -> None:
    source = DATA_DIR / "shapley_r2" / "s9_shapley_group_decomposition.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing S9 source data: {source}")
    df = pd.read_csv(source)
    sums = df.groupby("koppen_id", as_index=False).agg(
        full_model_r2=("full_model_r2", "first"),
        shapley_sum=("shapley_r2", "sum"),
    )
    gap = (sums["full_model_r2"] - sums["shapley_sum"]).abs().max()
    if gap > 1e-9:
        raise ValueError(f"S9 Shapley R2 values do not sum to full-model R2; max gap={gap}")


def validate_s10_source() -> None:
    source = DATA_DIR / "hard_energy_filtering" / "s10_hard_energy_filtering_sensitivity.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing S10 source data: {source}")
    df = pd.read_csv(source)
    if set(df["sw_threshold"]) != {21.6}:
        raise ValueError("S10 source data must use SW_8mean_raw >= 21.6 MJ m-2 day-1")
    if set(df["tmax_threshold"]) != {18.0}:
        raise ValueError("S10 source data must use Tmax_8mean_raw >= 18 C")
    vpd_thresholds = [column for column in df.columns if "vpd" in column.lower() and "threshold" in column.lower()]
    if vpd_thresholds:
        raise ValueError(f"S10 source data should not contain a VPD threshold: {vpd_thresholds}")


def validate_sources(label: str) -> None:
    if label.startswith("S9 "):
        validate_s9_source()
    if label.startswith("S10 "):
        validate_s10_source()


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy packaged supplement figure panels to supplement/outputs.")
    parser.add_argument("--quick", action="store_true", help="Accepted for command compatibility; all panels are lightweight.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    for label, source_names in FIGURE_SPECS:
        try:
            validate_sources(label)
            paths = [copy_processed_figure(source_name, OUTPUT_DIR) for source_name in source_names]
            report.append({"item": label, "status": "written", "paths": [rel(path) for path in paths]})
        except Exception as exc:
            report.append({"item": label, "status": "skipped", "reason": str(exc)})

    report_path = OUTPUT_DIR / "supplement_run_report.json"
    with report_path.open("w") as handle:
        json.dump({"quick": bool(args.quick), "items": report}, handle, indent=2)
    for item in report:
        label = item["item"]
        status = item["status"]
        detail = ", ".join(item.get("paths", [])) or item.get("reason", "")
        print(f"[supplement] {label}: {status} {detail}")
    print(f"[supplement] Wrote {rel(report_path)}")


if __name__ == "__main__":
    main()

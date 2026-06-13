#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE4_DATA = REPO_ROOT / "figure4" / "data"
DATA_DIR = REPO_ROOT / "supplement" / "data"
OUTPUT_DIR = REPO_ROOT / "supplement" / "outputs"


def load_figure4_results():
    results = []
    for zone_id in range(1, 6):
        path = FIGURE4_DATA / f"koppen{zone_id}" / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing Figure 4 results file: {path}")
        with path.open("r") as handle:
            results.append(json.load(handle))
    return results


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def copy_processed_figure(source_name: str, output_name: str, output_dir: Path) -> Path:
    source = DATA_DIR / "figures" / source_name
    if not source.exists():
        raise FileNotFoundError(f"Missing packaged supplement figure input: {source}")
    destination = output_dir / output_name
    shutil.copy2(source, destination)
    return destination


def raster_stats(path: Path, label: str, dataset: str, layer: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing packaged coupling raster: {path}")
    arr = np.load(path)
    values = arr[np.isfinite(arr)]
    if len(values) == 0:
        raise ValueError(f"Coupling raster has no finite values: {path}")
    return {
        "dataset": dataset,
        "layer": layer,
        "label": label,
        "n_pixels": int(len(values)),
        "median_r": float(np.nanmedian(values)),
        "p25_r": float(np.nanpercentile(values, 25)),
        "p75_r": float(np.nanpercentile(values, 75)),
        "mean_r": float(np.nanmean(values)),
    }


def write_s3(output_dir: Path):
    rows = [
        raster_stats(DATA_DIR / "coupling" / "gleam_surface_sm_vpd_r.npy", "GLEAM surface SM", "GLEAM v4.2b", "SMs"),
        raster_stats(DATA_DIR / "coupling" / "gleam_rootzone_sm_vpd_r.npy", "GLEAM root-zone SM", "GLEAM v4.2b", "SMrz"),
    ]
    table = output_dir / "s3_gleam_sm_vpd_coupling_summary.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    figures = [
        copy_processed_figure("s3_gleam_sm_vpd_correlation.png", "s3_gleam_sm_vpd_correlation.png", output_dir),
        copy_processed_figure("s3_gleam_sm_vpd_correlation.pdf", "s3_gleam_sm_vpd_correlation.pdf", output_dir),
    ]
    return [table, *figures]


def write_s4(output_dir: Path):
    rows = [
        raster_stats(DATA_DIR / "coupling" / "gldas_l1_sm_vpd_r.npy", "GLDAS L1 SM", "GLDAS Noah v2.1", "0-10 cm"),
        raster_stats(DATA_DIR / "coupling" / "gldas_l2_sm_vpd_r.npy", "GLDAS L2 SM", "GLDAS Noah v2.1", "10-40 cm"),
        raster_stats(DATA_DIR / "coupling" / "gldas_l3_sm_vpd_r.npy", "GLDAS L3 SM", "GLDAS Noah v2.1", "40-100 cm"),
    ]
    table = output_dir / "s4_gldas_sm_vpd_coupling_summary.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    figures = [
        copy_processed_figure("s4_gldas_sm_vpd_correlation.png", "s4_gldas_sm_vpd_correlation.png", output_dir),
        copy_processed_figure("s4_gldas_sm_vpd_correlation.pdf", "s4_gldas_sm_vpd_correlation.pdf", output_dir),
    ]
    return [table, *figures]


def write_s5(output_dir: Path):
    source = DATA_DIR / "gldas_yield" / "s5_gldas_exposure_coefficients.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing packaged S5 coefficient table: {source}")
    table = output_dir / "s5_gldas_yield_response_coefficients.csv"
    pd.read_csv(source).to_csv(table, index=False)
    figures = [
        copy_processed_figure("s5_gldas_yield_response.png", "s5_gldas_yield_response.png", output_dir),
        copy_processed_figure("s5_gldas_yield_response.pdf", "s5_gldas_yield_response.pdf", output_dir),
    ]
    return [table, *figures]


def write_s6(output_dir: Path):
    source = DATA_DIR / "gldas_yield" / "s6_gldas_r2_results.json"
    if not source.exists():
        raise FileNotFoundError(f"Missing packaged S6 R2 results: {source}")
    with source.open("r") as handle:
        data = json.load(handle)
    rows = []
    for model_name, metrics in data.get("pooled_models", {}).items():
        rows.append({"section": "pooled", "model": model_name, **metrics})
    for model_name, metrics in data.get("growth_stage_models", {}).items():
        rows.append({"section": "growth_stage", "model": model_name, **metrics})
    table = output_dir / "s6_gldas_yield_sensitivity_r2.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    json_out = output_dir / "s6_gldas_r2_results.json"
    shutil.copy2(source, json_out)
    figures = [
        copy_processed_figure("s6_gldas_yield_sensitivity.png", "s6_gldas_yield_sensitivity.png", output_dir),
        copy_processed_figure("s6_gldas_yield_sensitivity.pdf", "s6_gldas_yield_sensitivity.pdf", output_dir),
    ]
    return [table, json_out, *figures]


def write_s7(output_dir: Path):
    rows = []
    for zone_id in range(1, 6):
        source = DATA_DIR / "gldas_sif" / f"koppen{zone_id}" / "results.json"
        if not source.exists():
            raise FileNotFoundError(f"Missing packaged S7 GLDAS SIF result: {source}")
        with source.open("r") as handle:
            result = json.load(handle)
        metrics = result["metrics"]
        drop_group = result.get("drop_group_delta", {})
        drop_column = result.get("drop_column_delta", {})
        top_feature = max(drop_column, key=drop_column.get) if drop_column else None
        rows.append(
            {
                "koppen_id": result["koppen_id"],
                "koppen_name": result["koppen_name"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "train_r2": metrics.get("train_r2"),
                "test_r2": metrics.get("test_r2"),
                "delta_energy": drop_group.get("Energy"),
                "delta_surface": drop_group.get("SurfaceWater"),
                "delta_rootwater": drop_group.get("RootWater"),
                "dominance": max(drop_group, key=drop_group.get) if drop_group else None,
                "top_feature": top_feature,
                "top_feature_delta": drop_column.get(top_feature) if top_feature else None,
            }
        )
    table = output_dir / "s7_gldas_sif_attribution_summary.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    figures = [
        copy_processed_figure("s7_gldas_sif_attribution.png", "s7_gldas_sif_attribution.png", output_dir),
        copy_processed_figure("s7_gldas_sif_attribution.pdf", "s7_gldas_sif_attribution.pdf", output_dir),
    ]
    return [table, *figures]


def write_s8(results, output_dir: Path):
    rows = []
    for result in results:
        metrics = result["metrics"]
        rows.append(
            {
                "koppen_id": result["koppen_id"],
                "koppen_name": result["koppen_name"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "train_r2": metrics.get("train_r2"),
                "test_r2": metrics.get("test_r2"),
                "test_mae": metrics.get("test_mae"),
                "test_rmse": metrics.get("test_rmse"),
            }
        )
    path = output_dir / "s8_xgboost_model_performance.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_s9(results, output_dir: Path):
    feature_rows = []
    group_rows = []
    partial_rows = []
    for result in results:
        zone = {"koppen_id": result["koppen_id"], "koppen_name": result["koppen_name"]}
        for feature, value in result.get("shap_feature", {}).items():
            feature_rows.append({**zone, "feature": feature, "shapley_r2": value})
        for group, value in result.get("shap_group", {}).items():
            group_rows.append({**zone, "feature_group": group, "shapley_r2": value})
        for feature, value in result.get("partial_r2", {}).items():
            partial_rows.append({**zone, "feature": feature, "partial_r2": value})

    paths = [
        output_dir / "s9_shapley_feature_decomposition.csv",
        output_dir / "s9_shapley_group_decomposition.csv",
        output_dir / "s9_partial_r2.csv",
    ]
    pd.DataFrame(feature_rows).to_csv(paths[0], index=False)
    pd.DataFrame(group_rows).to_csv(paths[1], index=False)
    pd.DataFrame(partial_rows).to_csv(paths[2], index=False)
    return paths


def inspect_s10(output_dir: Path):
    path = REPO_ROOT / "training" / "outputs" / "hard_energy_filtering_sensitivity.csv"
    if not path.exists():
        return {
            "item": "S10 hard energy filtering",
            "status": "skipped",
            "reason": "Run optional ML retraining first to create training/outputs/hard_energy_filtering_sensitivity.csv.",
        }
    df = pd.read_csv(path)
    out_path = output_dir / "s10_hard_energy_filtering_sensitivity.csv"
    df.to_csv(out_path, index=False)
    if "status" in df.columns and (df["status"] == "computed").any():
        return {"item": "S10 hard energy filtering", "status": "written", "path": str(out_path.relative_to(REPO_ROOT))}
    reason = "; ".join(sorted(set(df.get("reason", pd.Series(["No computed rows."])).dropna().astype(str))))
    return {"item": "S10 hard energy filtering", "status": "skipped", "reason": reason}


def main():
    parser = argparse.ArgumentParser(description="Reproduce supported supplement tables from packaged processed inputs.")
    parser.add_argument("--quick", action="store_true", help="Run available lightweight supplement outputs and report skipped items")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    results = load_figure4_results()

    supplement_writers = [
        ("S3 GLEAM SM-VPD coupling", write_s3),
        ("S4 GLDAS Noah SM-VPD coupling", write_s4),
        ("S5 GLDAS Noah nonlinear yield response", write_s5),
        ("S6 GLDAS Noah yield sensitivity", write_s6),
        ("S7 GLDAS Noah SIF attribution", write_s7),
    ]
    for label, writer in supplement_writers:
        try:
            paths = writer(OUTPUT_DIR)
            report.append({"item": label, "status": "written", "paths": [rel(path) for path in paths]})
        except Exception as exc:
            report.append({"item": label, "status": "skipped", "reason": str(exc)})

    s8_path = write_s8(results, OUTPUT_DIR)
    report.append({"item": "S8 XGBoost model performance", "status": "written", "path": rel(s8_path)})

    for path in write_s9(results, OUTPUT_DIR):
        report.append({"item": "S9 Shapley/partial R2 table", "status": "written", "path": rel(path)})

    report.append(inspect_s10(OUTPUT_DIR))

    report_path = OUTPUT_DIR / "supplement_run_report.json"
    with report_path.open("w") as handle:
        json.dump({"quick": bool(args.quick), "items": report}, handle, indent=2)
    for item in report:
        label = item["item"]
        status = item["status"]
        detail = item.get("path") or ", ".join(item.get("paths", [])) or item.get("reason", "")
        print(f"[supplement] {label}: {status} {detail}")
    print(f"[supplement] Wrote {rel(report_path)}")


if __name__ == "__main__":
    main()

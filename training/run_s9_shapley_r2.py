#!/usr/bin/env python3
"""Recompute Supplementary Fig. S9 Shapley R2 decomposition.

This script ports the original S9 workflow into the packaged repository:
grouped year-stratified train/test split, soft energy weights, XGBoost
coalition models for all non-empty subsets of Energy/SurfaceWater/RootWater,
and held-out bootstrap evaluation of the Shapley R2 values.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from scipy.special import expit
from sklearn.model_selection import GroupKFold


warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = REPO_ROOT / "training" / "config.yml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training" / "outputs" / "shapley_r2"

FEATURES = [
    "VPDa_8mean",
    "SWa_8mean",
    "Tmaxa_8mean",
    "SMa_L1_8mean",
    "PPTa_8sum",
    "SMa_L2_8mean",
    "SMa_L3_8mean",
]

FEATURE_GROUPS = {
    "Energy": ["VPDa_8mean", "SWa_8mean", "Tmaxa_8mean"],
    "SurfaceWater": ["SMa_L1_8mean", "PPTa_8sum"],
    "RootWater": ["SMa_L2_8mean", "SMa_L3_8mean"],
}

GROUP_KEYS = list(FEATURE_GROUPS.keys())
GROUP_ORDER = ["RootWater", "Energy", "SurfaceWater"]
GROUP_LABELS = {"RootWater": "Root", "Energy": "Energy", "SurfaceWater": "Surface"}
SHORT_LABELS = {"Energy": "E", "SurfaceWater": "S", "RootWater": "R"}

KOPPEN_NAMES = {
    1: "Tropical (Af/Am)",
    2: "Savanna (Aw)",
    3: "Desert (BW)",
    4: "Semi-arid (BS)",
    5: "Temperate + Med",
}

KOPPEN_SHORT = {1: "Tropical", 2: "Savanna", 3: "Desert", 4: "Semi-arid", 5: "Temperate"}

COLOR_MAP = {
    "RootWater": "#1B7837",
    "Energy": "#D95F02",
    "SurfaceWater": "#92C5DE",
}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute Supplementary Fig. S9 Shapley R2 decomposition.")
    parser.add_argument("--config", default=str(CONFIG_FILE), help="Path to training/config.yml")
    parser.add_argument("--input", default=None, help="Override the packaged retraining input CSV")
    parser.add_argument("--outputs-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for S9 outputs")
    parser.add_argument("--bootstrap-iters", type=int, default=None, help="Bootstrap iterations; defaults to config value")
    parser.add_argument("--quick", action="store_true", help="Use reduced rows, CV work, tree count, and bootstrap iterations for smoke testing")
    parser.add_argument("--zones", nargs="+", type=int, default=None, help="Optional Koppen zone ids to run, e.g. --zones 1 2")
    return parser


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file did not contain a mapping: {path}")
    return config


def model_params(config: dict[str, Any], seed: int) -> dict[str, Any]:
    es = config["early_stopping"]
    return {
        "objective": "reg:squarederror",
        "tree_method": es.get("tree_method", "hist"),
        "max_depth": int(es["max_depth"]),
        "learning_rate": float(es["learning_rate"]),
        "subsample": float(es["subsample"]),
        "colsample_bytree": float(es["colsample_bytree"]),
        "reg_lambda": float(es.get("reg_lambda", 0.5)),
        "n_jobs": int(es.get("n_jobs", -1)),
        "random_state": int(seed),
    }


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def make_group_key(df: pd.DataFrame, admin_col: str = "admin2_idx", year_col: str = "year") -> np.ndarray:
    return (df[admin_col].astype(str) + "_" + df[year_col].astype(str)).values


def split_grouped_stratified_by_year(
    df: pd.DataFrame,
    group_key: np.ndarray,
    year_col: str,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    group_year_df = pd.DataFrame({"group": group_key, "year": df[year_col].values}).drop_duplicates()
    test_groups = []
    for year in group_year_df["year"].unique():
        year_groups = group_year_df[group_year_df["year"] == year]["group"].values
        n_test = max(1, int(len(year_groups) * test_ratio))
        year_test_groups = np.random.choice(year_groups, size=n_test, replace=False)
        test_groups.extend(year_test_groups)
    test_groups_set = set(test_groups)
    test_mask = np.array([group in test_groups_set for group in group_key])
    return ~test_mask, test_mask


def sigmoid_weight(series: pd.Series, threshold: float) -> np.ndarray:
    iqr = series.quantile(0.75) - series.quantile(0.25)
    sigma = max(float(iqr) / 2.0, 1e-3)
    return expit((series - threshold) / sigma)


def compute_weights(df: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    weight_cfg = config["sample_weight"]
    w_sw = sigmoid_weight(pd.to_numeric(df[weight_cfg["sw_column"]], errors="coerce"), float(weight_cfg["sw_threshold"]))
    w_t = sigmoid_weight(pd.to_numeric(df[weight_cfg["tmax_column"]], errors="coerce"), float(weight_cfg["tmax_threshold"]))
    weights = w_sw * w_t
    weights = weights / (np.mean(weights) + 1e-12)
    return np.clip(weights, 1e-3, 10)


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = weights.sum()
    y_mean = np.sum(weights * y_true) / (weight_sum + 1e-12)
    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_true - y_mean) ** 2)
    if ss_tot <= 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def all_coalitions() -> list[frozenset[str]]:
    coalitions: list[frozenset[str]] = [frozenset()]
    for size in range(1, len(GROUP_KEYS) + 1):
        for combo in combinations(GROUP_KEYS, size):
            coalitions.append(frozenset(combo))
    return coalitions


def coalition_features(coalition: frozenset[str]) -> list[str]:
    features: list[str] = []
    for group in coalition:
        features.extend(FEATURE_GROUPS[group])
    return features


def coalition_label(coalition: frozenset[str]) -> str:
    if not coalition:
        return "empty"
    return "{" + ",".join(sorted(SHORT_LABELS[group] for group in coalition)) + "}"


def compute_shapley_values(v: dict[frozenset[str], float]) -> dict[str, float]:
    energy, surface, root = "Energy", "SurfaceWater", "RootWater"
    full = frozenset([energy, surface, root])
    return {
        energy: (
            (1 / 3) * (v[frozenset([energy])] - v[frozenset()])
            + (1 / 6) * (v[frozenset([energy, surface])] - v[frozenset([surface])])
            + (1 / 6) * (v[frozenset([energy, root])] - v[frozenset([root])])
            + (1 / 3) * (v[full] - v[frozenset([surface, root])])
        ),
        surface: (
            (1 / 3) * (v[frozenset([surface])] - v[frozenset()])
            + (1 / 6) * (v[frozenset([energy, surface])] - v[frozenset([energy])])
            + (1 / 6) * (v[frozenset([surface, root])] - v[frozenset([root])])
            + (1 / 3) * (v[full] - v[frozenset([energy, root])])
        ),
        root: (
            (1 / 3) * (v[frozenset([root])] - v[frozenset()])
            + (1 / 6) * (v[frozenset([surface, root])] - v[frozenset([surface])])
            + (1 / 6) * (v[frozenset([energy, root])] - v[frozenset([energy])])
            + (1 / 3) * (v[full] - v[frozenset([energy, surface])])
        ),
    }


def prepare_df(config: dict[str, Any], input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Missing S9 retraining input: {input_file}")
    df = pd.read_csv(input_file)
    if "PPTa_8sum" not in df.columns and "PPTa_8mean" in df.columns:
        df["PPTa_8sum"] = df["PPTa_8mean"] * 8.0
    df["sample_weight"] = compute_weights(df, config)
    required = FEATURES + [
        config["data"]["target_variable"],
        config["data"]["admin_unit_variable"],
        config["data"]["year_stratification_variable"],
        config["data"]["climate_zone_variable"],
        config["sample_weight"]["sw_column"],
        config["sample_weight"]["tmax_column"],
    ]
    before = len(df)
    df = df.dropna(subset=list(dict.fromkeys(required))).copy()
    if len(df) != before:
        print(f"[s9_shapley_r2] Dropped {before - len(df)} rows with missing required values.")
    return df


def apply_quick_group_sample(df: pd.DataFrame, config: dict[str, Any], seed: int) -> pd.DataFrame:
    max_rows = int(config.get("quick", {}).get("max_rows_per_zone", 0) or 0)
    if max_rows <= 0:
        return df
    rng = np.random.default_rng(seed)
    zone_col = config["data"]["climate_zone_variable"]
    admin_col = config["data"]["admin_unit_variable"]
    year_col = config["data"]["year_stratification_variable"]
    group_col = "quick_group_id"
    df = df.copy()
    df[group_col] = df[admin_col].astype(str) + "_" + df[year_col].astype(str)
    selected_groups: set[str] = set()
    group_sizes = df.groupby([zone_col, group_col], sort=False).size().reset_index(name="n_rows")
    for _, zone_groups in group_sizes.groupby(zone_col, sort=False):
        shuffled = zone_groups.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        cumulative = 0
        for row in shuffled.itertuples(index=False):
            if cumulative >= max_rows and cumulative > 0:
                break
            selected_groups.add(getattr(row, group_col))
            cumulative += int(row.n_rows)
    return df[df[group_col].isin(selected_groups)].drop(columns=[group_col]).copy()


def find_n_fixed(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    group_key: np.ndarray,
    config: dict[str, Any],
    seed: int,
    quick: bool,
) -> int:
    unique_groups = len(pd.unique(group_key))
    if unique_groups < 2:
        return max(50, int(config["quick"]["n_estimators"] if quick else config["early_stopping"]["n_estimators"]))
    n_splits = min(int(config["cross_validation"]["n_splits"]), unique_groups)
    fold_limit = int(config.get("quick", {}).get("cv_folds_to_train", 0) or 0) if quick else 0
    n_estimators = int(config["quick"]["n_estimators"] if quick else config["early_stopping"]["n_estimators"])
    early_rounds = int(config["quick"]["early_stopping_rounds"] if quick else config["early_stopping"]["rounds"])
    best: list[int] = []
    for fold_idx, (train_idx, val_idx) in enumerate(GroupKFold(n_splits=n_splits).split(x_train, y_train, group_key), start=1):
        if fold_limit and fold_idx > fold_limit:
            break
        model = xgb.XGBRegressor(
            **model_params(config, seed),
            n_estimators=n_estimators,
            early_stopping_rounds=early_rounds,
        )
        model.fit(
            x_train[train_idx],
            y_train[train_idx],
            sample_weight=weights[train_idx],
            eval_set=[(x_train[val_idx], y_train[val_idx])],
            sample_weight_eval_set=[weights[val_idx]],
            verbose=False,
        )
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is not None and best_iteration >= 0:
            best.append(int(best_iteration))
    if not best:
        return max(50, n_estimators)
    return max(int(np.median(best)), 50)


def train_coalition_models(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    n_fixed: int,
    config: dict[str, Any],
    seed: int,
) -> dict[frozenset[str], np.ndarray]:
    predictions: dict[frozenset[str], np.ndarray] = {}
    for coalition in all_coalitions():
        if not coalition:
            continue
        features = coalition_features(coalition)
        model = xgb.XGBRegressor(**model_params(config, seed), n_estimators=int(n_fixed))
        model.fit(
            df_train[features].values,
            df_train["sif_anom"].values,
            sample_weight=df_train["sample_weight"].values,
            verbose=False,
        )
        predictions[coalition] = model.predict(df_test[features].values)
    return predictions


def evaluate_coalitions(
    y_test: np.ndarray,
    w_test: np.ndarray,
    predictions: dict[frozenset[str], np.ndarray],
    boot_idx: np.ndarray | None = None,
) -> dict[frozenset[str], float]:
    y = y_test[boot_idx] if boot_idx is not None else y_test
    weights = w_test[boot_idx] if boot_idx is not None else w_test
    v: dict[frozenset[str], float] = {frozenset(): 0.0}
    for coalition, pred in predictions.items():
        pred_values = pred[boot_idx] if boot_idx is not None else pred
        v[coalition] = weighted_r2(y, pred_values, weights)
    return v


def stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
    }


def shapley_r2_with_bootstrap(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    n_fixed: int,
    config: dict[str, Any],
    seed: int,
    bootstrap_iters: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    print("    Training 7 coalition models ...")
    predictions = train_coalition_models(df_train, df_test, n_fixed, config, seed)
    y_test = df_test["sif_anom"].values
    w_test = df_test["sample_weight"].values

    v_point = evaluate_coalitions(y_test, w_test, predictions)
    phi_point = compute_shapley_values(v_point)
    full_coalition = frozenset(GROUP_KEYS)
    gap = sum(phi_point.values()) - (v_point[full_coalition] - v_point[frozenset()])
    print(
        f"    Efficiency check: sum_phi={sum(phi_point.values()):.6f}, "
        f"v_full={v_point[full_coalition]:.6f}, gap={gap:.2e}"
    )

    rng = np.random.RandomState(seed)
    boot_phis = {group: [] for group in GROUP_KEYS}
    boot_vs = {coalition_label(coalition): [] for coalition in all_coalitions()}
    print(f"    Running {bootstrap_iters} bootstrap iterations ...")
    for _ in range(int(bootstrap_iters)):
        idx = rng.choice(len(y_test), size=len(y_test), replace=True)
        v_boot = evaluate_coalitions(y_test, w_test, predictions, boot_idx=idx)
        phi_boot = compute_shapley_values(v_boot)
        for group in GROUP_KEYS:
            boot_phis[group].append(phi_boot[group])
        for coalition in all_coalitions():
            boot_vs[coalition_label(coalition)].append(v_boot[coalition])

    v_readable = {coalition_label(coalition): float(v_point[coalition]) for coalition in all_coalitions()}
    phi_stats = {group: stats(boot_phis[group]) for group in GROUP_KEYS}
    v_stats = {label: stats(values) for label, values in boot_vs.items()}
    return v_readable, {group: float(phi_point[group]) for group in GROUP_KEYS}, phi_stats, v_stats


def analyze_zone(
    df: pd.DataFrame,
    koppen_id: int,
    config: dict[str, Any],
    seed: int,
    quick: bool,
    bootstrap_iters: int,
) -> dict[str, Any]:
    print(f"\n[s9_shapley_r2] Koppen {koppen_id}: {KOPPEN_NAMES[koppen_id]}")
    data_cfg = config["data"]
    df_zone = df[df[data_cfg["climate_zone_variable"]] == koppen_id].copy()
    group_key = make_group_key(df_zone, data_cfg["admin_unit_variable"], data_cfg["year_stratification_variable"])
    train_mask, test_mask = split_grouped_stratified_by_year(
        df_zone,
        group_key,
        data_cfg["year_stratification_variable"],
        test_ratio=float(config["test_size"]),
        seed=seed,
    )
    df_train = df_zone[train_mask].copy()
    df_test = df_zone[test_mask].copy()
    if df_train.empty or df_test.empty:
        raise RuntimeError(f"Koppen zone {koppen_id} has empty train or test partition.")
    print(f"  n_train={len(df_train):,} n_test={len(df_test):,}")

    x_full = df_train[FEATURES].values
    y_full = df_train["sif_anom"].values
    w_full = df_train["sample_weight"].values
    train_group_key = make_group_key(df_train, data_cfg["admin_unit_variable"], data_cfg["year_stratification_variable"])
    n_fixed = find_n_fixed(x_full, y_full, w_full, train_group_key, config, seed, quick)
    print(f"  n_fixed={n_fixed}")

    model_full = xgb.XGBRegressor(**model_params(config, seed), n_estimators=int(n_fixed))
    model_full.fit(x_full, y_full, sample_weight=w_full, verbose=False)
    y_pred = model_full.predict(df_test[FEATURES].values)
    baseline_r2 = weighted_r2(df_test["sif_anom"].values, y_pred, df_test["sample_weight"].values)
    print(f"  baseline_test_r2={baseline_r2:.6f}")

    v_readable, phi_point, phi_stats, v_stats = shapley_r2_with_bootstrap(
        df_train,
        df_test,
        n_fixed,
        config,
        seed,
        bootstrap_iters,
    )

    full_label = coalition_label(frozenset(GROUP_KEYS))
    drop_group_delta = {}
    for group in GROUP_KEYS:
        others = frozenset(key for key in GROUP_KEYS if key != group)
        drop_group_delta[group] = v_readable[full_label] - v_readable[coalition_label(others)]

    return {
        "koppen_id": int(koppen_id),
        "koppen_name": KOPPEN_NAMES[koppen_id],
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "n_fixed": int(n_fixed),
        "baseline_test_r2": float(baseline_r2),
        "coalition_values": v_readable,
        "coalition_bootstrap": v_stats,
        "shapley_phi": phi_point,
        "shapley_bootstrap": phi_stats,
        "drop_group_delta": drop_group_delta,
        "bootstrap_iters": int(bootstrap_iters),
    }


def write_outputs(results: dict[int, dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    long_rows = []
    full_label = coalition_label(frozenset(GROUP_KEYS))

    for koppen_id in sorted(results):
        result = results[koppen_id]
        zone_dir = output_dir / f"koppen{koppen_id}"
        zone_dir.mkdir(parents=True, exist_ok=True)
        with (zone_dir / "results.json").open("w") as handle:
            json.dump(result, handle, indent=2)

        phi = result["shapley_phi"]
        boot = result["shapley_bootstrap"]
        full_r2 = float(result["coalition_values"][full_label])
        summary_rows.append(
            {
                "koppen_id": koppen_id,
                "koppen_name": result["koppen_name"],
                "baseline_r2": result["baseline_test_r2"],
                "v_full": full_r2,
                "phi_Energy": phi["Energy"],
                "phi_SurfaceWater": phi["SurfaceWater"],
                "phi_RootWater": phi["RootWater"],
                "phi_sum": sum(phi.values()),
                "phi_Energy_ci": f"[{boot['Energy']['ci_low']:.4f}, {boot['Energy']['ci_high']:.4f}]",
                "phi_Surface_ci": f"[{boot['SurfaceWater']['ci_low']:.4f}, {boot['SurfaceWater']['ci_high']:.4f}]",
                "phi_Root_ci": f"[{boot['RootWater']['ci_low']:.4f}, {boot['RootWater']['ci_high']:.4f}]",
                "drop_Energy": result["drop_group_delta"]["Energy"],
                "drop_Surface": result["drop_group_delta"]["SurfaceWater"],
                "drop_Root": result["drop_group_delta"]["RootWater"],
                "bootstrap_iters": result["bootstrap_iters"],
            }
        )
        for group in GROUP_KEYS:
            long_rows.append(
                {
                    "koppen_id": koppen_id,
                    "koppen_name": result["koppen_name"],
                    "full_model_r2": full_r2,
                    "feature_group": group,
                    "shapley_r2": phi[group],
                    "share_percent": phi[group] / full_r2 * 100 if full_r2 else np.nan,
                    "ci_low": boot[group]["ci_low"],
                    "ci_high": boot[group]["ci_high"],
                    "drop_delta_r2": result["drop_group_delta"][group],
                    "bootstrap_iters": result["bootstrap_iters"],
                }
            )

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(long_rows).to_csv(output_dir / "s9_shapley_group_decomposition.csv", index=False)
    if set(results) == {1, 2, 3, 4, 5}:
        plot_shapley_stacked(results, output_dir)


def plot_shapley_stacked(results: dict[int, dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    fig.subplots_adjust(left=0.22, right=0.88, top=0.90, bottom=0.15)
    y_pos = np.arange(5)
    zone_labels = [KOPPEN_SHORT[koppen_id] for koppen_id in range(1, 6)]

    left = np.zeros(5)
    for group in GROUP_ORDER:
        values = np.array([results[koppen_id]["shapley_phi"][group] for koppen_id in range(1, 6)])
        ax.barh(
            y_pos,
            values,
            left=left,
            height=0.6,
            color=COLOR_MAP[group],
            edgecolor="white",
            linewidth=0.3,
            label=GROUP_LABELS[group],
        )
        for idx, koppen_id in enumerate(range(1, 6)):
            full_r2 = results[koppen_id]["baseline_test_r2"]
            if values[idx] > 0.015:
                ax.text(
                    left[idx] + values[idx] / 2,
                    y_pos[idx],
                    f"{values[idx] / full_r2 * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        left += values

    for idx, koppen_id in enumerate(range(1, 6)):
        full_r2 = results[koppen_id]["baseline_test_r2"]
        ax.plot(full_r2, y_pos[idx], "k|", markersize=10, markeredgewidth=1.0)
        ax.text(full_r2 + 0.008, y_pos[idx], f"$R^2$={full_r2:.2f}", va="center", fontsize=8, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(zone_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(r"Shapley $\phi$ (contribution to $R^2$)", fontsize=10)
    ax.set_xlim(0, max(results[koppen_id]["baseline_test_r2"] for koppen_id in results) * 1.35)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.3, alpha=0.5, color="#CCCCCC")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=COLOR_MAP["RootWater"], edgecolor="none", label="Root"),
            mpatches.Patch(facecolor=COLOR_MAP["Energy"], edgecolor="none", label="Energy"),
            mpatches.Patch(facecolor=COLOR_MAP["SurfaceWater"], edgecolor="none", label="Surface"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=9,
    )
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"figure_shapley_r2_stacked.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_from_args(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = load_config(config_path)
    seed = int(config["random_seed"])
    input_file = Path(args.input) if args.input else REPO_ROOT / config["data"]["input_file"]
    if not input_file.is_absolute():
        input_file = REPO_ROOT / input_file
    output_dir = Path(args.outputs_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    bootstrap_iters = int(args.bootstrap_iters if args.bootstrap_iters is not None else config["bootstrap"]["iters"])
    if args.quick and args.bootstrap_iters is None:
        bootstrap_iters = int(config["quick"]["bootstrap_iters"])

    df = prepare_df(config, input_file)
    if args.quick:
        df = apply_quick_group_sample(df, config, seed)
    zone_values = args.zones or sorted(int(zone) for zone in df[config["data"]["climate_zone_variable"]].dropna().unique())
    results = {}
    for zone_id in zone_values:
        if zone_id not in KOPPEN_NAMES:
            raise ValueError(f"Unsupported Koppen zone: {zone_id}")
        results[zone_id] = analyze_zone(df, zone_id, config, seed, bool(args.quick), bootstrap_iters)
    write_outputs(results, output_dir)
    print(f"\n[s9_shapley_r2] Wrote outputs to {output_dir}")


def main() -> None:
    run_from_args(build_parser().parse_args())


if __name__ == "__main__":
    main()

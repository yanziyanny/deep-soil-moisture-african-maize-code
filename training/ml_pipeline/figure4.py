from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "training" / "config.yml"
OUTPUT_DIR = REPO_ROOT / "training" / "outputs"
FIGURE4_DATA_DIR = REPO_ROOT / "figure4" / "data"

KOPPEN_NAMES = {
    1: "Tropical (Af/Am)",
    2: "Savanna (Aw)",
    3: "Desert (BW)",
    4: "Semi-arid (BS)",
    5: "Temperate + Med",
}

VAR_LABELS = {
    "VPDa_8mean": "VPD",
    "SWa_8mean": "SW",
    "Tmaxa_8mean": "Tmax",
    "SMa_L1_8mean": "SM L1",
    "PPTa_8sum": "PPT",
    "SMa_L2_8mean": "SM L2",
    "SMa_L3_8mean": "SM L3",
}


class SchemaError(ValueError):
    """Raised when the packaged retraining input cannot satisfy the ML schema."""


@dataclass
class PreparedData:
    frame: pd.DataFrame
    features: list[str]
    feature_groups: dict[str, list[str]]
    source_columns: dict[str, str]
    hard_filter_columns: dict[str, dict[str, Any]]
    notes: list[str]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the self-contained Figure 4 ML retraining pipeline.")
    parser.add_argument("--config", default=str(CONFIG_FILE), help="Path to training/config.yml")
    parser.add_argument("--input", default=None, help="Override the packaged retraining input CSV")
    parser.add_argument("--outputs-dir", default=str(OUTPUT_DIR), help="Directory for machine-readable training outputs")
    parser.add_argument("--bootstrap-iters", type=int, default=None, help="Bootstrap iterations; defaults to config value")
    parser.add_argument("--quick", action="store_true", help="Use reduced settings for CI and smoke tests")
    parser.add_argument(
        "--sync-figure-data",
        action="store_true",
        help="Sync generated Figure 4 summary/results into figure4/data even in quick mode",
    )
    parser.add_argument(
        "--no-sync-figure-data",
        action="store_true",
        help="Do not sync generated Figure 4 summary/results into figure4/data",
    )
    return parser


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file did not contain a mapping: {config_path}")
    return config


def all_features(config: dict[str, Any]) -> list[str]:
    features: list[str] = []
    for group_features in config["feature_groups"].values():
        features.extend(group_features)
    return features


def resolve_column(columns: set[str], canonical: str, aliases: list[str] | None = None) -> str | None:
    candidates = [canonical]
    if aliases:
        candidates.extend(aliases)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def prepare_data(config: dict[str, Any], input_file: Path) -> PreparedData:
    if not input_file.exists():
        raise FileNotFoundError(f"Missing packaged Figure 4 retraining input: {input_file}")

    df = pd.read_csv(input_file)
    columns = set(df.columns)
    data_cfg = config["data"]
    feature_groups = {group: list(features) for group, features in config["feature_groups"].items()}
    features = all_features(config)
    source_columns: dict[str, str] = {}
    missing: list[str] = []
    notes: list[str] = []

    required_columns = [
        data_cfg["target_variable"],
        data_cfg["admin_unit_variable"],
        data_cfg["year_stratification_variable"],
        data_cfg["climate_zone_variable"],
    ]
    for column in required_columns:
        if column not in columns:
            missing.append(column)

    aliases = config.get("feature_aliases", {})
    for feature in features:
        source = resolve_column(columns, feature, aliases.get(feature, []))
        if source is None:
            missing.append(f"{feature} (aliases: {', '.join(aliases.get(feature, []))})")
            continue
        source_columns[feature] = source
        if source != feature:
            if feature == "PPTa_8sum" and source in {"PPTa_8mean", "PPT_8mean_anom"}:
                df[feature] = df[source] * 8.0
                notes.append(f"Converted input column {source} to canonical feature {feature} by multiplying by 8.")
            else:
                df[feature] = df[source]
                notes.append(f"Using input column {source} for canonical feature {feature}.")

    if missing:
        raise SchemaError(
            "The packaged ML input is missing required columns: "
            + "; ".join(missing)
            + ". See training/data/README.md for the expected schema."
        )

    admin_col = data_cfg["admin_unit_variable"]
    year_col = data_cfg["year_stratification_variable"]
    group_col = data_cfg["grouping_variable"]
    df[group_col] = df[admin_col].astype(str) + "-" + df[year_col].astype(str)

    keep_cols = [data_cfg["target_variable"], admin_col, year_col, data_cfg["climate_zone_variable"], group_col] + features
    weight_col = data_cfg.get("sample_weight_variable")
    if weight_col:
        keep_cols.append(weight_col)
    before = len(df)
    df = df.dropna(subset=keep_cols).copy()
    if len(df) != before:
        notes.append(f"Dropped {before - len(df)} rows with missing target, grouping, climate-zone, or feature values.")

    hard_columns = resolve_hard_filter_columns(df, config)
    return PreparedData(df, features, feature_groups, source_columns, hard_columns, notes)


def resolve_hard_filter_columns(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    columns = set(df.columns)
    for canonical, spec in config["hard_energy_filter"]["thresholds"].items():
        source = resolve_column(columns, canonical, spec.get("aliases", []))
        if source is None:
            resolved[canonical] = {
                "source_column": None,
                "available": False,
                "reason": f"Missing raw threshold column for {spec['label']}: {canonical}",
                "threshold": spec["value"],
                "units": spec["units"],
            }
            continue

        threshold = spec["value"]
        units = spec["units"]
        values = pd.to_numeric(df[source], errors="coerce")
        alternate = spec.get("alternate_if_values_look_like_mj_m2_day")
        if alternate and values.max() < 100 and threshold >= 100:
            threshold = alternate["value"]
            units = alternate["units"]
        alternate_hpa = spec.get("alternate_if_values_look_like_hpa")
        if alternate_hpa and values.max() > 20 and threshold <= 2:
            threshold = alternate_hpa["value"]
            units = alternate_hpa["units"]
        resolved[canonical] = {
            "source_column": source,
            "available": True,
            "threshold": threshold,
            "units": units,
            "label": spec["label"],
        }
    return resolved


def apply_quick_group_sample(df: pd.DataFrame, config: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    quick_cfg = config.get("quick", {})
    max_rows = int(quick_cfg.get("max_rows_per_zone", 0) or 0)
    if max_rows <= 0:
        return df

    zone_col = config["data"]["climate_zone_variable"]
    group_col = config["data"]["grouping_variable"]
    selected_groups: set[str] = set()
    group_sizes = df.groupby([zone_col, group_col], sort=True).size().reset_index(name="n_rows")
    for zone, zone_groups in group_sizes.groupby(zone_col, sort=True):
        shuffled = zone_groups.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        cumulative = 0
        for row in shuffled.itertuples(index=False):
            if cumulative >= max_rows and cumulative > 0:
                break
            selected_groups.add(getattr(row, group_col))
            cumulative += int(row.n_rows)
    return df[df[group_col].isin(selected_groups)].copy()


def make_group_table(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    group_col = data_cfg["grouping_variable"]
    admin_col = data_cfg["admin_unit_variable"]
    year_col = data_cfg["year_stratification_variable"]
    zone_col = data_cfg["climate_zone_variable"]
    return (
        df[[group_col, admin_col, year_col, zone_col]]
        .drop_duplicates()
        .rename(columns={group_col: "group_id", admin_col: "admin2_idx", year_col: "year", zone_col: "koppen5"})
        .sort_values(["year", "koppen5", "admin2_idx"])
        .reset_index(drop=True)
    )


def assign_train_test(groups: pd.DataFrame, config: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    groups = groups.copy()
    groups["split"] = "train"
    test_size = float(config["test_size"])
    strata = ["year", "koppen5"] if "koppen5" in groups.columns else ["year"]
    for _, stratum in groups.groupby(strata, sort=True):
        indices = stratum.index.to_numpy()
        if len(indices) <= 1:
            continue
        n_test = max(1, int(round(len(indices) * test_size)))
        n_test = min(n_test, len(indices) - 1)
        test_indices = rng.choice(indices, size=n_test, replace=False)
        groups.loc[test_indices, "split"] = "test"
    return groups


def assign_groupkfolds(groups: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    groups = groups.copy()
    groups["validation_fold"] = pd.NA
    n_splits_config = int(config["cross_validation"]["n_splits"])
    for zone, zone_groups in groups[groups["split"] == "train"].groupby("koppen5", sort=True):
        if len(zone_groups) < 2:
            continue
        n_splits = min(n_splits_config, len(zone_groups))
        splitter = GroupKFold(n_splits=n_splits)
        row_positions = zone_groups.index.to_numpy()
        dummy = np.zeros(len(zone_groups))
        group_ids = zone_groups["group_id"].to_numpy()
        for fold_idx, (_, val_pos) in enumerate(splitter.split(dummy, groups=group_ids), start=1):
            groups.loc[row_positions[val_pos], "validation_fold"] = fold_idx
    return groups


def validate_splits(groups: pd.DataFrame) -> None:
    split_counts = groups.groupby("group_id")["split"].nunique()
    leaked = split_counts[split_counts > 1]
    if not leaked.empty:
        raise RuntimeError(f"Train/test leakage detected for groups: {leaked.index[:5].tolist()}")

    fold_counts = groups.dropna(subset=["validation_fold"]).groupby("group_id")["validation_fold"].nunique()
    repeated = fold_counts[fold_counts > 1]
    if not repeated.empty:
        raise RuntimeError(f"GroupKFold leakage detected for groups: {repeated.index[:5].tolist()}")


def save_split_files(groups: pd.DataFrame, outputs_dir: Path) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    split_cols = ["group_id", "admin2_idx", "year", "koppen5", "split"]
    fold_cols = ["group_id", "admin2_idx", "year", "koppen5", "split", "validation_fold"]
    groups[split_cols].to_csv(outputs_dir / "train_test_split_ids.csv", index=False)
    groups[fold_cols].to_csv(outputs_dir / "groupkfold_fold_ids.csv", index=False)


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
    weight_sum = np.sum(sample_weight)
    if weight_sum <= 0 or len(y_true) == 0:
        return float("nan")
    y_bar = np.average(y_true, weights=sample_weight)
    ss_tot = np.sum(sample_weight * (y_true - y_bar) ** 2)
    if ss_tot <= 0:
        return float("nan")
    ss_res = np.sum(sample_weight * (y_true - y_pred) ** 2)
    return float(1.0 - ss_res / ss_tot)


def model_params(config: dict[str, Any], seed: int, n_estimators: int | None = None, early_stopping: bool = False) -> dict[str, Any]:
    es = config["early_stopping"]
    params = {
        "objective": "reg:squarederror",
        "n_estimators": int(n_estimators or es["n_estimators"]),
        "learning_rate": float(es["learning_rate"]),
        "max_depth": int(es["max_depth"]),
        "min_child_weight": float(es.get("min_child_weight", 1)),
        "subsample": float(es["subsample"]),
        "colsample_bytree": float(es["colsample_bytree"]),
        "reg_lambda": float(es.get("reg_lambda", 1.0)),
        "tree_method": es.get("tree_method", "hist"),
        "random_state": int(seed),
        "n_jobs": int(es.get("n_jobs", 1)),
        "eval_metric": es.get("eval_metric", "rmse"),
    }
    if early_stopping:
        params["early_stopping_rounds"] = int(es["rounds"])
    return params


def fit_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
    seed: int,
    n_estimators: int | None = None,
    sample_weight: np.ndarray | None = None,
) -> XGBRegressor:
    model = XGBRegressor(**model_params(config, seed, n_estimators=n_estimators, early_stopping=False))
    model.fit(X, y, sample_weight=sample_weight, verbose=False)
    return model


def cv_early_stopping(
    zone_df: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
    fold_groups: pd.DataFrame,
    seed: int,
    quick: bool,
) -> tuple[int, dict[str, float]]:
    target_col = config["data"]["target_variable"]
    group_col = config["data"]["grouping_variable"]
    fold_limit = config.get("quick", {}).get("cv_folds_to_train") if quick else None
    fold_values = sorted(v for v in fold_groups["validation_fold"].dropna().unique())
    if fold_limit:
        fold_values = fold_values[: int(fold_limit)]

    best_iterations: list[int] = []
    val_scores: list[float] = []
    for fold in fold_values:
        val_group_ids = set(fold_groups.loc[fold_groups["validation_fold"] == fold, "group_id"])
        val_mask = zone_df[group_col].isin(val_group_ids)
        train_mask = ~val_mask
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        model = XGBRegressor(**model_params(config, seed + int(fold), early_stopping=True))
        X_train = zone_df.loc[train_mask, features]
        y_train = zone_df.loc[train_mask, target_col]
        X_val = zone_df.loc[val_mask, features]
        y_val = zone_df.loc[val_mask, target_col]
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = model.predict(X_val)
        val_scores.append(weighted_r2(y_val.to_numpy(), pred))
        best = getattr(model, "best_iteration", None)
        if best is None:
            best = model.get_booster().best_iteration
        if best is not None and best >= 0:
            best_iterations.append(int(best) + 1)

    configured_estimators = int(config["early_stopping"]["n_estimators"])
    if quick:
        configured_estimators = int(config["quick"]["n_estimators"])
    if best_iterations:
        final_estimators = max(10, min(configured_estimators, int(np.median(best_iterations))))
    else:
        final_estimators = configured_estimators

    cv_summary = {
        "cv_folds_trained": float(len(val_scores)),
        "cv_mean_r2": float(np.nanmean(val_scores)) if val_scores else float("nan"),
        "cv_std_r2": float(np.nanstd(val_scores)) if val_scores else float("nan"),
        "best_iteration_median": float(np.median(best_iterations)) if best_iterations else float("nan"),
    }
    return final_estimators, cv_summary


def bootstrap_delta_stats(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    alternate_predictions: dict[str, np.ndarray],
    actual_deltas: dict[str, float],
    group_ids: np.ndarray,
    bootstrap_iters: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if bootstrap_iters <= 0:
        return {
            key: {"mean": float(delta), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
            for key, delta in actual_deltas.items()
        }

    rng = np.random.default_rng(seed)
    unique_groups = pd.unique(group_ids)
    positions_by_group = {group: np.flatnonzero(group_ids == group) for group in unique_groups}
    draws: dict[str, list[float]] = {key: [] for key in alternate_predictions}
    for _ in range(bootstrap_iters):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        positions = np.concatenate([positions_by_group[group] for group in sampled_groups])
        base_r2 = weighted_r2(y_true[positions], baseline_pred[positions])
        for key, pred in alternate_predictions.items():
            draws[key].append(base_r2 - weighted_r2(y_true[positions], pred[positions]))

    stats: dict[str, dict[str, float]] = {}
    for key, values in draws.items():
        arr = np.asarray(values, dtype=float)
        stats[key] = {
            "mean": float(actual_deltas[key]),
            "std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else float("nan"),
            "ci_low": float(np.nanpercentile(arr, 2.5)),
            "ci_high": float(np.nanpercentile(arr, 97.5)),
        }
    return stats


def compute_shapley_summary(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    baseline_r2: float,
    feature_groups: dict[str, list[str]],
    sample_size: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    if len(X_test) == 0:
        return {col: 0.0 for col in X_test.columns}, {group: 0.0 for group in feature_groups}
    if len(X_test) > sample_size:
        sampled = X_test.sample(n=sample_size, random_state=seed)
    else:
        sampled = X_test
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sampled)
        mean_abs = np.abs(np.asarray(values)).mean(axis=0)
    except Exception:
        mean_abs = np.asarray(model.feature_importances_, dtype=float)

    if mean_abs.sum() > 0 and math.isfinite(baseline_r2):
        feature_values = {
            feature: float(max(baseline_r2, 0.0) * value / mean_abs.sum())
            for feature, value in zip(sampled.columns, mean_abs)
        }
    else:
        feature_values = {feature: 0.0 for feature in sampled.columns}
    group_values = {
        group: float(sum(feature_values.get(feature, 0.0) for feature in features))
        for group, features in feature_groups.items()
    }
    return feature_values, group_values


def compute_hard_filter_rows(
    zone_id: int,
    zone_test: pd.DataFrame,
    y_test: np.ndarray,
    baseline_pred: np.ndarray,
    drop_group_predictions: dict[str, np.ndarray],
    hard_columns: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    missing = [spec["reason"] for spec in hard_columns.values() if not spec.get("available")]
    if missing:
        return [
            {
                "koppen_id": zone_id,
                "koppen_name": KOPPEN_NAMES.get(zone_id, str(zone_id)),
                "status": "skipped",
                "reason": "; ".join(missing),
                "n_test_filtered": 0,
                "test_r2_filtered": np.nan,
                "delta_energy_filtered": np.nan,
                "delta_surface_filtered": np.nan,
                "delta_rootwater_filtered": np.nan,
            }
        ]

    mask = np.ones(len(zone_test), dtype=bool)
    threshold_notes = []
    for spec in hard_columns.values():
        source = spec["source_column"]
        threshold = spec["threshold"]
        mask &= pd.to_numeric(zone_test[source], errors="coerce").to_numpy() > threshold
        threshold_notes.append(f"{source}>{threshold} {spec['units']}")

    if mask.sum() < 5:
        return [
            {
                "koppen_id": zone_id,
                "koppen_name": KOPPEN_NAMES.get(zone_id, str(zone_id)),
                "status": "skipped",
                "reason": "Fewer than five held-out rows pass the hard energy filter: " + ", ".join(threshold_notes),
                "n_test_filtered": int(mask.sum()),
                "test_r2_filtered": np.nan,
                "delta_energy_filtered": np.nan,
                "delta_surface_filtered": np.nan,
                "delta_rootwater_filtered": np.nan,
            }
        ]

    base_r2 = weighted_r2(y_test[mask], baseline_pred[mask])
    return [
        {
            "koppen_id": zone_id,
            "koppen_name": KOPPEN_NAMES.get(zone_id, str(zone_id)),
            "status": "computed",
            "reason": ", ".join(threshold_notes),
            "n_test_filtered": int(mask.sum()),
            "test_r2_filtered": base_r2,
            "delta_energy_filtered": base_r2 - weighted_r2(y_test[mask], drop_group_predictions["Energy"][mask]),
            "delta_surface_filtered": base_r2 - weighted_r2(y_test[mask], drop_group_predictions["SurfaceWater"][mask]),
            "delta_rootwater_filtered": base_r2 - weighted_r2(y_test[mask], drop_group_predictions["RootWater"][mask]),
        }
    ]


def run_zone(
    zone_id: int,
    df: pd.DataFrame,
    groups: pd.DataFrame,
    prepared: PreparedData,
    config: dict[str, Any],
    bootstrap_iters: int,
    quick: bool,
    seed: int,
) -> dict[str, Any]:
    data_cfg = config["data"]
    target_col = data_cfg["target_variable"]
    group_col = data_cfg["grouping_variable"]
    zone_col = data_cfg["climate_zone_variable"]

    zone_df = df[df[zone_col] == zone_id].copy()
    zone_group_ids = set(zone_df[group_col])
    zone_groups = groups[groups["group_id"].isin(zone_group_ids)].copy()
    train_group_ids = set(zone_groups.loc[zone_groups["split"] == "train", "group_id"])
    test_group_ids = set(zone_groups.loc[zone_groups["split"] == "test", "group_id"])

    train_df = zone_df[zone_df[group_col].isin(train_group_ids)]
    test_df = zone_df[zone_df[group_col].isin(test_group_ids)]
    if train_df.empty or test_df.empty:
        raise RuntimeError(f"Koppen zone {zone_id} has empty train or test partition.")

    features = prepared.features
    final_estimators, cv_summary = cv_early_stopping(train_df, features, config, zone_groups, seed, quick)
    baseline_model = fit_model(train_df[features], train_df[target_col], config, seed, n_estimators=final_estimators)
    train_pred = baseline_model.predict(train_df[features])
    test_pred = baseline_model.predict(test_df[features])
    y_train = train_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()
    baseline_train_r2 = weighted_r2(y_train, train_pred)
    baseline_test_r2 = weighted_r2(y_test, test_pred)

    metrics = {
        "n_fixed": int(train_df[group_col].nunique()),
        "train_r2": baseline_train_r2,
        "test_r2": baseline_test_r2,
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "test_rmse": float(mean_squared_error(y_test, test_pred, squared=False)),
    }

    drop_column_delta: dict[str, float] = {}
    drop_column_r2: dict[str, float] = {}
    drop_column_pred: dict[str, np.ndarray] = {}
    for index, feature in enumerate(features):
        keep = [col for col in features if col != feature]
        model = fit_model(train_df[keep], train_df[target_col], config, seed + 100 + index, n_estimators=final_estimators)
        pred = model.predict(test_df[keep])
        r2 = weighted_r2(y_test, pred)
        drop_column_r2[feature] = r2
        drop_column_delta[feature] = baseline_test_r2 - r2
        drop_column_pred[feature] = pred

    drop_group_delta: dict[str, float] = {}
    drop_group_r2: dict[str, float] = {}
    drop_group_pred: dict[str, np.ndarray] = {}
    for index, (group, group_features) in enumerate(prepared.feature_groups.items()):
        keep = [col for col in features if col not in group_features]
        model = fit_model(train_df[keep], train_df[target_col], config, seed + 200 + index, n_estimators=final_estimators)
        pred = model.predict(test_df[keep])
        r2 = weighted_r2(y_test, pred)
        drop_group_r2[group] = r2
        drop_group_delta[group] = baseline_test_r2 - r2
        drop_group_pred[group] = pred

    column_bootstrap = bootstrap_delta_stats(
        y_test,
        test_pred,
        drop_column_pred,
        drop_column_delta,
        test_df[group_col].to_numpy(),
        bootstrap_iters,
        seed + 300,
    )
    group_bootstrap = bootstrap_delta_stats(
        y_test,
        test_pred,
        drop_group_pred,
        drop_group_delta,
        test_df[group_col].to_numpy(),
        bootstrap_iters,
        seed + 400,
    )

    shap_sample_size = int(config.get("shapley", {}).get("sample_size", 5000))
    if quick:
        shap_sample_size = int(config.get("quick", {}).get("shap_sample_size", min(500, shap_sample_size)))
    shap_feature, shap_group = compute_shapley_summary(
        baseline_model,
        test_df[features],
        baseline_test_r2,
        prepared.feature_groups,
        shap_sample_size,
        seed + 500,
    )

    partial_r2: dict[str, float] = {}
    for index, feature in enumerate(features):
        model = fit_model(
            train_df[[feature]],
            train_df[target_col],
            config,
            seed + 600 + index,
            n_estimators=max(10, min(final_estimators, int(config.get("quick", {}).get("n_estimators", final_estimators)) if quick else final_estimators)),
        )
        pred = model.predict(test_df[[feature]])
        partial_r2[feature] = weighted_r2(y_test, pred)

    hard_rows = compute_hard_filter_rows(zone_id, test_df, y_test, test_pred, drop_group_pred, prepared.hard_filter_columns)

    result_json = {
        "koppen_id": int(zone_id),
        "koppen_name": KOPPEN_NAMES.get(int(zone_id), str(zone_id)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "metrics": metrics,
        "drop_group_delta": drop_group_delta,
        "drop_group_bootstrap": group_bootstrap,
        "drop_column_delta": drop_column_delta,
        "drop_column_bootstrap": column_bootstrap,
        "shap_feature": shap_feature,
        "shap_group": shap_group,
        "partial_r2": partial_r2,
    }

    return {
        "zone_id": int(zone_id),
        "result_json": result_json,
        "cv_summary": cv_summary,
        "final_estimators": final_estimators,
        "drop_column_r2": drop_column_r2,
        "drop_group_r2": drop_group_r2,
        "hard_filter_rows": hard_rows,
    }


def write_artifacts(
    zone_outputs: list[dict[str, Any]],
    prepared: PreparedData,
    groups: pd.DataFrame,
    config: dict[str, Any],
    outputs_dir: Path,
    sync_figure_data: bool,
    metadata: dict[str, Any],
) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    figure_output_root = outputs_dir / "figure4_data"
    figure_output_root.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    drop_individual_rows = []
    drop_group_rows = []
    bootstrap_rows = []
    hard_rows = []
    summary_rows = []

    for zone_output in zone_outputs:
        result = zone_output["result_json"]
        zone_id = result["koppen_id"]
        zone_name = result["koppen_name"]
        metrics = result["metrics"]
        metrics_rows.append(
            {
                "koppen_id": zone_id,
                "koppen_name": zone_name,
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "n_train_groups": metrics["n_fixed"],
                "train_r2": metrics["train_r2"],
                "test_r2": metrics["test_r2"],
                "test_mae": metrics["test_mae"],
                "test_rmse": metrics["test_rmse"],
                "cv_mean_r2": zone_output["cv_summary"]["cv_mean_r2"],
                "cv_std_r2": zone_output["cv_summary"]["cv_std_r2"],
                "cv_folds_trained": zone_output["cv_summary"]["cv_folds_trained"],
                "best_iteration_median": zone_output["cv_summary"]["best_iteration_median"],
                "final_n_estimators": zone_output["final_estimators"],
            }
        )

        for feature, delta in result["drop_column_delta"].items():
            group = next(group for group, features in prepared.feature_groups.items() if feature in features)
            drop_individual_rows.append(
                {
                    "koppen_id": zone_id,
                    "koppen_name": zone_name,
                    "feature": feature,
                    "feature_label": VAR_LABELS.get(feature, feature),
                    "feature_group": group,
                    "source_column": prepared.source_columns.get(feature, feature),
                    "baseline_test_r2": metrics["test_r2"],
                    "dropped_test_r2": zone_output["drop_column_r2"][feature],
                    "delta_r2": delta,
                }
            )
            stats = result["drop_column_bootstrap"][feature]
            bootstrap_rows.append(
                {
                    "koppen_id": zone_id,
                    "koppen_name": zone_name,
                    "scope": "individual",
                    "term": feature,
                    "mean_delta_r2": stats["mean"],
                    "std": stats["std"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "bootstrap_iters": metadata["bootstrap_iters"],
                }
            )

        for group, delta in result["drop_group_delta"].items():
            drop_group_rows.append(
                {
                    "koppen_id": zone_id,
                    "koppen_name": zone_name,
                    "feature_group": group,
                    "features": ",".join(prepared.feature_groups[group]),
                    "baseline_test_r2": metrics["test_r2"],
                    "dropped_test_r2": zone_output["drop_group_r2"][group],
                    "delta_r2": delta,
                }
            )
            stats = result["drop_group_bootstrap"][group]
            bootstrap_rows.append(
                {
                    "koppen_id": zone_id,
                    "koppen_name": zone_name,
                    "scope": "group",
                    "term": group,
                    "mean_delta_r2": stats["mean"],
                    "std": stats["std"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "bootstrap_iters": metadata["bootstrap_iters"],
                }
            )

        hard_rows.extend(zone_output["hard_filter_rows"])
        dominance = max(result["drop_group_delta"], key=result["drop_group_delta"].get)
        top_feature = max(result["drop_column_delta"], key=result["drop_column_delta"].get)
        summary_rows.append(
            {
                "koppen_id": zone_id,
                "koppen_name": zone_name,
                "test_r2": metrics["test_r2"],
                "delta_energy": result["drop_group_delta"].get("Energy", np.nan),
                "delta_surface": result["drop_group_delta"].get("SurfaceWater", np.nan),
                "delta_rootwater": result["drop_group_delta"].get("RootWater", np.nan),
                "dominance": dominance,
                "top_var": VAR_LABELS.get(top_feature, top_feature),
                "top_var_delta": result["drop_column_delta"][top_feature],
            }
        )

        zone_dir = figure_output_root / f"koppen{zone_id}"
        zone_dir.mkdir(parents=True, exist_ok=True)
        with (zone_dir / "results.json").open("w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)

    pd.DataFrame(metrics_rows).to_csv(outputs_dir / "model_metrics_by_zone.csv", index=False)
    pd.DataFrame(drop_individual_rows).to_csv(outputs_dir / "drop_column_importance_individual.csv", index=False)
    pd.DataFrame(drop_group_rows).to_csv(outputs_dir / "drop_column_importance_group.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(outputs_dir / "bootstrap_confidence_intervals.csv", index=False)
    pd.DataFrame(hard_rows).to_csv(outputs_dir / "hard_energy_filtering_sensitivity.csv", index=False)
    summary = pd.DataFrame(summary_rows).sort_values("koppen_id")
    summary.to_csv(figure_output_root / "summary.csv", index=False)

    metadata = dict(metadata)
    metadata["output_files"] = [
        "training/outputs/model_metrics_by_zone.csv",
        "training/outputs/drop_column_importance_individual.csv",
        "training/outputs/drop_column_importance_group.csv",
        "training/outputs/bootstrap_confidence_intervals.csv",
        "training/outputs/hard_energy_filtering_sensitivity.csv",
        "training/outputs/train_test_split_ids.csv",
        "training/outputs/groupkfold_fold_ids.csv",
        "training/outputs/figure4_data/summary.csv",
        "training/outputs/figure4_data/koppen*/results.json",
    ]
    metadata["schema_notes"] = prepared.notes
    metadata["feature_source_columns"] = prepared.source_columns
    metadata["hard_filter_columns"] = prepared.hard_filter_columns
    metadata["split_summary"] = groups["split"].value_counts().to_dict()
    with (outputs_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    if sync_figure_data:
        FIGURE4_DATA_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(FIGURE4_DATA_DIR / "summary.csv", index=False)
        for zone_output in zone_outputs:
            zone_id = zone_output["result_json"]["koppen_id"]
            destination = FIGURE4_DATA_DIR / f"koppen{zone_id}" / "results.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w") as handle:
                json.dump(zone_output["result_json"], handle, indent=2, sort_keys=True)


def run_pipeline(
    config_path: Path,
    input_file: Path | None,
    outputs_dir: Path,
    bootstrap_iters: int | None,
    quick: bool,
    sync_figure_data: bool,
) -> None:
    start = time.perf_counter()
    config = load_config(config_path)
    seed = int(config["random_seed"])
    rng = np.random.default_rng(seed)
    if input_file is None:
        input_file = REPO_ROOT / config["data"]["input_file"]
    elif not input_file.is_absolute():
        input_file = REPO_ROOT / input_file

    if bootstrap_iters is None:
        bootstrap_iters = int(config["bootstrap"]["iters"])
    if quick:
        bootstrap_iters = int(config["quick"]["bootstrap_iters"]) if bootstrap_iters == int(config["bootstrap"]["iters"]) else bootstrap_iters
        config["early_stopping"]["n_estimators"] = int(config["quick"]["n_estimators"])
        config["early_stopping"]["rounds"] = int(config["quick"]["early_stopping_rounds"])

    prepared = prepare_data(config, input_file)
    df = prepared.frame
    if quick:
        df = apply_quick_group_sample(df, config, rng)

    groups = make_group_table(df, config)
    groups = assign_train_test(groups, config, rng)
    groups = assign_groupkfolds(groups, config)
    validate_splits(groups)
    save_split_files(groups, outputs_dir)

    zone_outputs = []
    for zone_id in sorted(df[config["data"]["climate_zone_variable"]].dropna().unique()):
        print(f"[retrain_figure4] Training Koppen zone {int(zone_id)}")
        zone_outputs.append(
            run_zone(
                int(zone_id),
                df,
                groups,
                prepared,
                config,
                bootstrap_iters,
                quick,
                seed + int(zone_id) * 1000,
            )
        )

    metadata = {
        "quick": bool(quick),
        "bootstrap_iters": int(bootstrap_iters),
        "random_seed": seed,
        "input_file": str(input_file.relative_to(REPO_ROOT) if input_file.is_relative_to(REPO_ROOT) else input_file),
        "n_rows": int(len(df)),
        "n_groups": int(groups["group_id"].nunique()),
        "test_size": float(config["test_size"]),
        "grouping_variable": config["data"]["grouping_variable"],
        "year_stratification_variable": config["data"]["year_stratification_variable"],
        "cv_method": config["cross_validation"]["method"],
        "cv_n_splits": int(config["cross_validation"]["n_splits"]),
        "sync_figure_data": bool(sync_figure_data),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "runtime_seconds": float(time.perf_counter() - start),
    }
    write_artifacts(zone_outputs, prepared, groups, config, outputs_dir, sync_figure_data, metadata)
    print(f"[retrain_figure4] Wrote machine-readable outputs to {outputs_dir}")
    if sync_figure_data:
        print("[retrain_figure4] Synced Figure 4 packaged outputs to figure4/data")
    else:
        print("[retrain_figure4] Figure 4 packaged outputs were left unchanged; generated copies are in training/outputs/figure4_data")


def run_from_args(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    input_file = Path(args.input) if args.input else None
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.is_absolute():
        outputs_dir = REPO_ROOT / outputs_dir
    if args.no_sync_figure_data:
        sync = False
    elif args.sync_figure_data:
        sync = True
    else:
        sync = not args.quick
    run_pipeline(config_path, input_file, outputs_dir, args.bootstrap_iters, args.quick, sync)


def main() -> None:
    parser = build_arg_parser()
    run_from_args(parser.parse_args())


if __name__ == "__main__":
    main()

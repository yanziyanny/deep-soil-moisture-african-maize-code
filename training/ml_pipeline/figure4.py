from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
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
    parser.add_argument("--quick", action="store_true", help="Use reduced settings for smoke tests")
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


def sigmoid_weight(series: pd.Series, threshold: float) -> np.ndarray:
    iqr = series.quantile(0.75) - series.quantile(0.25)
    sigma = max(float(iqr) / 2.0, 1e-3)
    z = np.clip((series.to_numpy(dtype=float) - threshold) / sigma, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def compute_sample_weight(df: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    weight_cfg = config.get("sample_weight", {})
    if weight_cfg.get("method") != "energy_soft_threshold":
        return np.ones(len(df), dtype=float)
    sw = sigmoid_weight(pd.to_numeric(df[weight_cfg["sw_column"]], errors="coerce"), float(weight_cfg["sw_threshold"]))
    tmax = sigmoid_weight(pd.to_numeric(df[weight_cfg["tmax_column"]], errors="coerce"), float(weight_cfg["tmax_threshold"]))
    weights = sw * tmax
    weights = weights / (np.nanmean(weights) + 1e-12)
    return np.clip(weights, 1e-3, 10.0)


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
    df[group_col] = df[admin_col].astype(str) + "_" + df[year_col].astype(str)

    raw_weight_cols = []
    weight_cfg = config.get("sample_weight", {})
    if weight_cfg.get("method") == "energy_soft_threshold":
        raw_weight_cols = [weight_cfg["sw_column"], weight_cfg["tmax_column"]]

    hard_columns = resolve_hard_filter_columns(df, config)
    hard_raw_cols = [spec["source_column"] for spec in hard_columns.values() if spec.get("source_column")]

    subset_cols = [
        data_cfg["target_variable"],
        admin_col,
        year_col,
        data_cfg["climate_zone_variable"],
        group_col,
        *features,
        *raw_weight_cols,
        *hard_raw_cols,
    ]
    before = len(df)
    df = df.dropna(subset=list(dict.fromkeys(subset_cols))).copy()
    if len(df) != before:
        notes.append(f"Dropped {before - len(df)} rows with missing target, grouping, climate-zone, feature, or raw energy values.")

    df["sample_weight"] = compute_sample_weight(df, config)
    return PreparedData(df, features, feature_groups, source_columns, hard_columns, notes)


def apply_quick_group_sample(df: pd.DataFrame, config: dict[str, Any], seed: int) -> pd.DataFrame:
    max_rows = int(config.get("quick", {}).get("max_rows_per_zone", 0) or 0)
    if max_rows <= 0:
        return df
    rng = np.random.default_rng(seed)
    zone_col = config["data"]["climate_zone_variable"]
    group_col = config["data"]["grouping_variable"]
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
    return df[df[group_col].isin(selected_groups)].copy()


def split_grouped_stratified_by_year(
    df_zone: pd.DataFrame,
    group_col: str,
    year_col: str,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    group_year = df_zone[[group_col, year_col]].drop_duplicates()
    test_groups: list[str] = []
    for year in group_year[year_col].unique():
        year_groups = group_year.loc[group_year[year_col] == year, group_col].to_numpy()
        if len(year_groups) == 0:
            continue
        n_test = max(1, int(len(year_groups) * test_ratio))
        if len(year_groups) > 1:
            n_test = min(n_test, len(year_groups) - 1)
        year_test_groups = rng.choice(year_groups, size=n_test, replace=False)
        test_groups.extend(year_test_groups.tolist())

    test_set = set(test_groups)
    test_mask = df_zone[group_col].isin(test_set).to_numpy()
    train_mask = ~test_mask
    split_rows = group_year.copy()
    split_rows["split"] = np.where(split_rows[group_col].isin(test_set), "test", "train")
    return train_mask, test_mask, split_rows


def assign_validation_folds(split_groups: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    split_groups = split_groups.copy()
    group_col = config["data"]["grouping_variable"]
    zone_col = config["data"]["climate_zone_variable"]
    split_groups["validation_fold"] = pd.NA
    n_splits_config = int(config["cross_validation"]["n_splits"])
    for _, zone_groups in split_groups[split_groups["split"] == "train"].groupby(zone_col, sort=False):
        if len(zone_groups) < 2:
            continue
        n_splits = min(n_splits_config, len(zone_groups))
        row_positions = zone_groups.index.to_numpy()
        dummy = np.zeros(len(zone_groups))
        group_ids = zone_groups[group_col].to_numpy()
        splitter = GroupKFold(n_splits=n_splits)
        for fold_idx, (_, val_pos) in enumerate(splitter.split(dummy, groups=group_ids), start=1):
            split_groups.loc[row_positions[val_pos], "validation_fold"] = fold_idx
    return split_groups


def validate_splits(groups: pd.DataFrame, config: dict[str, Any]) -> None:
    group_col = config["data"]["grouping_variable"]
    split_counts = groups.groupby(group_col)["split"].nunique()
    leaked = split_counts[split_counts > 1]
    if not leaked.empty:
        raise RuntimeError(f"Train/test leakage detected for groups: {leaked.index[:5].tolist()}")
    fold_counts = groups.dropna(subset=["validation_fold"]).groupby(group_col)["validation_fold"].nunique()
    repeated = fold_counts[fold_counts > 1]
    if not repeated.empty:
        raise RuntimeError(f"GroupKFold leakage detected for groups: {repeated.index[:5].tolist()}")


def save_split_files(groups: pd.DataFrame, config: dict[str, Any], outputs_dir: Path) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = config["data"]
    group_col = data_cfg["grouping_variable"]
    admin_col = data_cfg["admin_unit_variable"]
    year_col = data_cfg["year_stratification_variable"]
    zone_col = data_cfg["climate_zone_variable"]
    rename = {group_col: "group_id", admin_col: "admin2_idx", year_col: "year", zone_col: "koppen5"}
    split_cols = [group_col, admin_col, year_col, zone_col, "split"]
    fold_cols = [group_col, admin_col, year_col, zone_col, "split", "validation_fold"]
    groups[split_cols].rename(columns=rename).to_csv(outputs_dir / "train_test_split_ids.csv", index=False)
    groups[fold_cols].rename(columns=rename).to_csv(outputs_dir / "groupkfold_fold_ids.csv", index=False)


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if weights is None:
        weights = np.ones_like(y_true, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return 0.0
    y_mean = np.sum(weights * y_true) / (weight_sum + 1e-12)
    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_true - y_mean) ** 2)
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * np.abs(y_true - y_pred)) / (weights.sum() + 1e-12))


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * (y_true - y_pred) ** 2) / (weights.sum() + 1e-12)))


def model_params(config: dict[str, Any], seed: int) -> dict[str, Any]:
    es = config["early_stopping"]
    return {
        "objective": "reg:squarederror",
        "tree_method": es.get("tree_method", "hist"),
        "max_depth": int(es["max_depth"]),
        "learning_rate": float(es["learning_rate"]),
        "min_child_weight": float(es.get("min_child_weight", 1)),
        "subsample": float(es["subsample"]),
        "colsample_bytree": float(es["colsample_bytree"]),
        "reg_lambda": float(es.get("reg_lambda", 1.0)),
        "n_jobs": int(es.get("n_jobs", -1)),
        "random_state": int(seed),
    }


def find_n_fixed(df_train: pd.DataFrame, features: list[str], config: dict[str, Any], seed: int, quick: bool) -> int:
    X = df_train[features].to_numpy()
    y = df_train[config["data"]["target_variable"]].to_numpy()
    weights = df_train["sample_weight"].to_numpy()
    groups = df_train[config["data"]["grouping_variable"]].to_numpy()
    splitter = GroupKFold(n_splits=int(config["cross_validation"]["n_splits"]))
    best: list[int] = []
    n_estimators = int(config["quick"]["n_estimators"] if quick else config["early_stopping"]["n_estimators"])
    early_rounds = int(config["quick"]["early_stopping_rounds"] if quick else config["early_stopping"]["rounds"])
    fold_limit = int(config.get("quick", {}).get("cv_folds_to_train", 0) or 0) if quick else 0
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        if fold_limit and fold_idx > fold_limit:
            break
        model = XGBRegressor(
            **model_params(config, seed),
            n_estimators=n_estimators,
            early_stopping_rounds=early_rounds,
        )
        model.fit(
            X[train_idx],
            y[train_idx],
            sample_weight=weights[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            sample_weight_eval_set=[weights[val_idx]],
            verbose=False,
        )
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is not None and best_iteration >= 0:
            best.append(int(best_iteration))
    if not best:
        return max(50, n_estimators)
    return max(int(np.median(best)), 50)


def train_model(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
    seed: int,
    n_fixed: int | None = None,
    quick: bool = False,
) -> tuple[XGBRegressor, dict[str, float]]:
    if n_fixed is None:
        n_fixed = find_n_fixed(df_train, features, config, seed, quick)
    target = config["data"]["target_variable"]
    X_train = df_train[features].to_numpy()
    y_train = df_train[target].to_numpy()
    w_train = df_train["sample_weight"].to_numpy()
    X_test = df_test[features].to_numpy()
    y_test = df_test[target].to_numpy()
    w_test = df_test["sample_weight"].to_numpy()

    model = XGBRegressor(**model_params(config, seed), n_estimators=int(n_fixed))
    model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    metrics = {
        "n_fixed": int(n_fixed),
        "train_r2": weighted_r2(y_train, train_pred, w_train),
        "test_r2": weighted_r2(y_test, test_pred, w_test),
        "test_mae": weighted_mae(y_test, test_pred, w_test),
        "test_rmse": weighted_rmse(y_test, test_pred, w_test),
    }
    return model, metrics


def bootstrap_stats(
    y_true: np.ndarray,
    weights: np.ndarray,
    baseline_pred: np.ndarray,
    alternate_predictions: dict[str, np.ndarray],
    bootstrap_iters: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.RandomState(seed)
    n_test = len(y_true)
    draws: dict[str, list[float]] = {key: [] for key in alternate_predictions}
    if bootstrap_iters <= 0:
        baseline_r2 = weighted_r2(y_true, baseline_pred, weights)
        return {
            key: {
                "mean": float(baseline_r2 - weighted_r2(y_true, pred, weights)),
                "std": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
            }
            for key, pred in alternate_predictions.items()
        }
    for _ in range(bootstrap_iters):
        boot_idx = rng.choice(n_test, size=n_test, replace=True)
        y_boot = y_true[boot_idx]
        w_boot = weights[boot_idx]
        baseline_r2 = weighted_r2(y_boot, baseline_pred[boot_idx], w_boot)
        for key, pred in alternate_predictions.items():
            draws[key].append(baseline_r2 - weighted_r2(y_boot, pred[boot_idx], w_boot))
    stats: dict[str, dict[str, float]] = {}
    for key, values in draws.items():
        arr = np.asarray(values, dtype=float)
        stats[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
        }
    return stats


def compute_shap(model: XGBRegressor, df_train: pd.DataFrame, df_test: pd.DataFrame, features: list[str], feature_groups: dict[str, list[str]], config: dict[str, Any], quick: bool) -> tuple[dict[str, float], dict[str, float]]:
    sample_size = int(config.get("shapley", {}).get("sample_size", 0) or 0)
    if quick:
        sample_size = int(config.get("quick", {}).get("shap_sample_size", 500))
    if sample_size > 0 and len(df_test) > sample_size:
        df_test = df_test.sample(n=sample_size, random_state=int(config["random_seed"]))
    try:
        import shap

        explainer = shap.TreeExplainer(
            model,
            data=df_train[features].to_numpy(),
            feature_perturbation="interventional",
        )
        shap_vals = explainer.shap_values(df_test[features].to_numpy())
        weights = df_test["sample_weight"].to_numpy()
        mean_abs = np.average(np.abs(shap_vals), axis=0, weights=weights)
    except Exception:
        mean_abs = np.asarray(model.feature_importances_, dtype=float)
    feature_values = {feature: float(value) for feature, value in zip(features, mean_abs)}
    group_values = {
        group: float(sum(feature_values.get(feature, 0.0) for feature in group_features))
        for group, group_features in feature_groups.items()
    }
    return feature_values, group_values


def compute_partial_r2(df_train: pd.DataFrame, features: list[str], config: dict[str, Any], seed: int, quick: bool) -> dict[str, float]:
    splitter = GroupKFold(n_splits=int(config["cross_validation"]["n_splits"]))
    target = config["data"]["target_variable"]
    group_col = config["data"]["grouping_variable"]
    X = df_train[features].to_numpy()
    y = df_train[target].to_numpy()
    weights = df_train["sample_weight"].to_numpy()
    groups = df_train[group_col].to_numpy()
    n_estimators = 80 if quick else 200
    partial: dict[str, float] = {}
    for var in features:
        others = [feature for feature in features if feature != var]
        if not others:
            partial[var] = 0.0
            continue
        y_resid = np.zeros_like(y, dtype=float)
        x_resid = np.zeros_like(y, dtype=float)
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            model_y = XGBRegressor(**model_params(config, seed + fold_idx), n_estimators=n_estimators)
            model_y.fit(
                df_train.iloc[train_idx][others].to_numpy(),
                y[train_idx],
                sample_weight=weights[train_idx],
                verbose=False,
            )
            y_resid[test_idx] = y[test_idx] - model_y.predict(df_train.iloc[test_idx][others].to_numpy())

            model_x = XGBRegressor(**model_params(config, seed + 100 + fold_idx), n_estimators=n_estimators)
            model_x.fit(
                df_train.iloc[train_idx][others].to_numpy(),
                df_train.iloc[train_idx][var].to_numpy(),
                sample_weight=weights[train_idx],
                verbose=False,
            )
            x_resid[test_idx] = df_train.iloc[test_idx][var].to_numpy() - model_x.predict(df_train.iloc[test_idx][others].to_numpy())

        y_mean = np.sum(weights * y_resid) / (weights.sum() + 1e-12)
        x_mean = np.sum(weights * x_resid) / (weights.sum() + 1e-12)
        cov = np.sum(weights * (y_resid - y_mean) * (x_resid - x_mean))
        var_y = np.sum(weights * (y_resid - y_mean) ** 2)
        var_x = np.sum(weights * (x_resid - x_mean) ** 2)
        if var_y <= 0 or var_x <= 0:
            partial[var] = 0.0
        else:
            corr = cov / np.sqrt(var_y * var_x)
            partial[var] = max(0.0, float(corr**2))
    return partial


def compute_hard_filter_rows(
    zone_id: int,
    df_test: pd.DataFrame,
    baseline_pred: np.ndarray,
    drop_group_predictions: dict[str, np.ndarray],
    hard_columns: dict[str, dict[str, Any]],
    config: dict[str, Any],
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

    mask = np.ones(len(df_test), dtype=bool)
    threshold_notes = []
    for spec in hard_columns.values():
        source = spec["source_column"]
        threshold = spec["threshold"]
        mask &= pd.to_numeric(df_test[source], errors="coerce").to_numpy() > threshold
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

    target = config["data"]["target_variable"]
    y_test = df_test[target].to_numpy()
    weights = df_test["sample_weight"].to_numpy()
    base_r2 = weighted_r2(y_test[mask], baseline_pred[mask], weights[mask])
    return [
        {
            "koppen_id": zone_id,
            "koppen_name": KOPPEN_NAMES.get(zone_id, str(zone_id)),
            "status": "computed",
            "reason": ", ".join(threshold_notes),
            "n_test_filtered": int(mask.sum()),
            "test_r2_filtered": base_r2,
            "delta_energy_filtered": base_r2 - weighted_r2(y_test[mask], drop_group_predictions["Energy"][mask], weights[mask]),
            "delta_surface_filtered": base_r2 - weighted_r2(y_test[mask], drop_group_predictions["SurfaceWater"][mask], weights[mask]),
            "delta_rootwater_filtered": base_r2 - weighted_r2(y_test[mask], drop_group_predictions["RootWater"][mask], weights[mask]),
        }
    ]


def run_zone(
    zone_id: int,
    df: pd.DataFrame,
    prepared: PreparedData,
    config: dict[str, Any],
    bootstrap_iters: int,
    quick: bool,
    seed: int,
) -> dict[str, Any]:
    data_cfg = config["data"]
    zone_col = data_cfg["climate_zone_variable"]
    group_col = data_cfg["grouping_variable"]
    year_col = data_cfg["year_stratification_variable"]
    target = data_cfg["target_variable"]
    features = prepared.features

    df_zone = df[df[zone_col] == zone_id].copy()
    train_mask, test_mask, split_groups = split_grouped_stratified_by_year(
        df_zone,
        group_col,
        year_col,
        float(config["test_size"]),
        seed=int(config["random_seed"]),
    )
    split_groups[data_cfg["admin_unit_variable"]] = split_groups[group_col].str.split("_").str[0].astype(df_zone[data_cfg["admin_unit_variable"]].dtype)
    split_groups[zone_col] = zone_id

    df_train = df_zone[train_mask].copy()
    df_test = df_zone[test_mask].copy()
    if df_train.empty or df_test.empty:
        raise RuntimeError(f"Koppen zone {zone_id} has empty train or test partition.")

    baseline_model, metrics = train_model(df_train, df_test, features, config, seed, quick=quick)
    y_test = df_test[target].to_numpy()
    weights = df_test["sample_weight"].to_numpy()
    baseline_pred = baseline_model.predict(df_test[features].to_numpy())

    drop_column_predictions: dict[str, np.ndarray] = {}
    drop_column_r2: dict[str, float] = {}
    for feature in features:
        keep = [col for col in features if col != feature]
        model, _ = train_model(df_train, df_test, keep, config, seed, n_fixed=int(metrics["n_fixed"]), quick=quick)
        pred = model.predict(df_test[keep].to_numpy())
        drop_column_predictions[feature] = pred
        drop_column_r2[feature] = weighted_r2(y_test, pred, weights)

    drop_group_predictions: dict[str, np.ndarray] = {}
    drop_group_r2: dict[str, float] = {}
    for group, group_features in prepared.feature_groups.items():
        keep = [col for col in features if col not in group_features]
        model, _ = train_model(df_train, df_test, keep, config, seed, n_fixed=int(metrics["n_fixed"]), quick=quick)
        pred = model.predict(df_test[keep].to_numpy())
        drop_group_predictions[group] = pred
        drop_group_r2[group] = weighted_r2(y_test, pred, weights)

    column_bootstrap = bootstrap_stats(y_test, weights, baseline_pred, drop_column_predictions, bootstrap_iters, seed=int(config["random_seed"]))
    group_bootstrap = bootstrap_stats(y_test, weights, baseline_pred, drop_group_predictions, bootstrap_iters, seed=int(config["random_seed"]))
    drop_column_delta = {feature: stats["mean"] for feature, stats in column_bootstrap.items()}
    drop_group_delta = {group: stats["mean"] for group, stats in group_bootstrap.items()}

    shap_feature, shap_group = compute_shap(baseline_model, df_train, df_test, features, prepared.feature_groups, config, quick)
    partial_r2 = compute_partial_r2(df_train, features, config, seed, quick)
    hard_rows = compute_hard_filter_rows(zone_id, df_test, baseline_pred, drop_group_predictions, prepared.hard_filter_columns, config)

    result_json = {
        "koppen_id": int(zone_id),
        "koppen_name": KOPPEN_NAMES[int(zone_id)],
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
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
        "split_groups": split_groups,
        "drop_column_r2": drop_column_r2,
        "drop_group_r2": drop_group_r2,
        "hard_filter_rows": hard_rows,
    }


def write_artifacts(
    zone_outputs: list[dict[str, Any]],
    prepared: PreparedData,
    split_groups: pd.DataFrame,
    config: dict[str, Any],
    outputs_dir: Path,
    sync_figure_data: bool,
    metadata: dict[str, Any],
) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    figure_output_root = outputs_dir / "figure4_data"
    figure_output_root.mkdir(parents=True, exist_ok=True)

    split_groups = assign_validation_folds(split_groups, config)
    validate_splits(split_groups, config)
    save_split_files(split_groups, config, outputs_dir)

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
                "n_fixed": metrics["n_fixed"],
                "train_r2": metrics["train_r2"],
                "test_r2": metrics["test_r2"],
                "test_mae": metrics["test_mae"],
                "test_rmse": metrics["test_rmse"],
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
            json.dump(result, handle, indent=2)

    pd.DataFrame(metrics_rows).to_csv(outputs_dir / "model_metrics_by_zone.csv", index=False)
    pd.DataFrame(drop_individual_rows).to_csv(outputs_dir / "drop_column_importance_individual.csv", index=False)
    pd.DataFrame(drop_group_rows).to_csv(outputs_dir / "drop_column_importance_group.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(outputs_dir / "bootstrap_confidence_intervals.csv", index=False)
    hard_table = pd.DataFrame(hard_rows)
    hard_table.to_csv(outputs_dir / "hard_energy_filtering_sensitivity.csv", index=False)
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
    metadata["split_summary"] = split_groups["split"].value_counts().to_dict()
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
                json.dump(zone_output["result_json"], handle, indent=2)


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
    if input_file is None:
        input_file = REPO_ROOT / config["data"]["input_file"]
    elif not input_file.is_absolute():
        input_file = REPO_ROOT / input_file

    if bootstrap_iters is None:
        bootstrap_iters = int(config["bootstrap"]["iters"])
    if quick and bootstrap_iters == int(config["bootstrap"]["iters"]):
        bootstrap_iters = int(config["quick"]["bootstrap_iters"])

    prepared = prepare_data(config, input_file)
    df = prepared.frame
    if quick:
        df = apply_quick_group_sample(df, config, seed)

    zone_outputs = []
    split_frames = []
    for zone_id in sorted(df[config["data"]["climate_zone_variable"]].dropna().unique()):
        print(f"[retrain_figure4] Training Koppen zone {int(zone_id)}")
        output = run_zone(int(zone_id), df, prepared, config, int(bootstrap_iters), quick, seed)
        zone_outputs.append(output)
        split_frames.append(output["split_groups"])
    split_groups = pd.concat(split_frames, ignore_index=True)

    metadata = {
        "quick": bool(quick),
        "bootstrap_iters": int(bootstrap_iters),
        "random_seed": seed,
        "input_file": str(input_file.relative_to(REPO_ROOT) if input_file.is_relative_to(REPO_ROOT) else input_file),
        "n_rows": int(len(df)),
        "n_groups": int(split_groups[config["data"]["grouping_variable"]].nunique()),
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
    write_artifacts(zone_outputs, prepared, split_groups, config, outputs_dir, sync_figure_data, metadata)
    print(f"[retrain_figure4] Wrote machine-readable outputs to {outputs_dir}")
    if sync_figure_data:
        print("[retrain_figure4] Synced Figure 4 packaged outputs to figure4/data")
    else:
        print(f"[retrain_figure4] Figure 4 packaged outputs were left unchanged; generated copies are in {outputs_dir / 'figure4_data'}")


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

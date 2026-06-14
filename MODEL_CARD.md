# Model Card: Figure 4 XGBoost Attribution Model

## Model Overview

- Model type: XGBoost regression.
- Purpose: attribution of 8-day SIF anomaly variability across African cropland climate zones, not operational forecasting.
- Target variable: `sif_anom`, an 8-day deseasonalized SIF anomaly.
- Spatial and temporal unit: admin2/county-like unit by 8-day growing-season observations, grouped by `admin2_idx-year` for splitting.
- Pretrained model: none.
- Deployed or reusable forecasting model: none.

## Predictors

Functional groups are defined in `training/config.yml`.

- Energy: `VPDa_8mean`, `Tmaxa_8mean`, `SWa_8mean`.
- Surface Water: `SMa_L1_8mean`, `PPTa_8sum`.
- Root Water: `SMa_L2_8mean`, `SMa_L3_8mean`.

The packaged retraining input contains the canonical `PPTa_8sum` feature used by the packaged Figure 4 JSON files. If an alternate input supplies only `PPTa_8mean`, the retraining code converts it to `PPTa_8sum` by multiplying by 8.

## Train/Test Split and Validation

- Held-out split: grouped 80/20 train/test split.
- Grouping variable: `county_year`, constructed from `admin2_idx` and `year`.
- Stratification: by year within each Koppen climate zone.
- Cross-validation: 5-fold `GroupKFold` within each Koppen zone.
- Early stopping: configured in `training/config.yml`.
- Reproducibility: all random behavior uses `random_seed` from `training/config.yml`.

The retraining workflow writes `training/outputs/train_test_split_ids.csv` and `training/outputs/groupkfold_fold_ids.csv`, and checks that no county-year group appears in both train and test or in more than one validation fold.

## Metrics and Interpretation

- Primary metric: weighted held-out test R2.
- Sample weights: soft energy-availability weights derived from raw shortwave radiation and maximum temperature, as configured in `training/config.yml`.
- Additional metrics: train R2, test MAE, test RMSE, cross-validation R2 summaries.
- Ablation design: drop-column Delta R2 for individual predictors and functional groups.
- Bootstrap CIs: bootstrap resampling of held-out rows with replacement.
- Shapley sensitivity: Tree SHAP-derived feature and group contributions are written to the Figure 4 JSON outputs.
- Partial R2: single-feature XGBoost fits are reported as an additional sensitivity.

## Benchmark and Baseline Context

This analysis is designed for scientific attribution rather than a public benchmark prediction task. There is no standard external benchmark dataset for the specific task of attributing African maize SIF anomaly variability to soil-water depth and energy-limitation predictors.

Model evaluation and comparison are based on:

- grouped held-out test performance and 5-fold grouped cross-validation;
- drop-column and group ablation relative to the full XGBoost model;
- single-feature partial R2 fits;
- hard energy-filtering sensitivity;
- robustness checks using alternative processed soil-moisture products in the supplement.

## Hard Energy-Filtering Sensitivity

The configured hard energy thresholds are:

- Tmax > 15 C.
- VPD > 0.5 kPa.
- SW > 200 W m-2.

The packaged analysis-ready retraining panel contains raw VPD, Tmax, and SW columns for this sensitivity. The retraining code converts thresholds when the packaged raw-column range indicates an equivalent unit such as hPa for VPD or MJ m-2 day-1 for SW.

## Intended Uses

- Reproduce and audit the model-dependent attribution analysis behind Figure 4.
- Inspect split IDs, validation folds, ablation design, bootstrap uncertainty, and sensitivity summaries.
- Support Nature Portfolio ML checklist assessment of the submitted code package.

## Non-Intended Uses

- Operational yield, SIF, drought, or crop-stress forecasting.
- Transfer to regions, crops, sensors, or time periods outside the processed analysis panel without revalidation.
- Policy or farm management decisions without independent validation and uncertainty analysis.

## Limitations

- The repository contains processed figure-ready data and an analysis-ready ML retraining panel, not raw remote-sensing archives.
- Exact hard energy-filtering sensitivity depends on the documented unit conversion for raw VPD and SW thresholds in `training/config.yml`.
- Quick mode is a smoke-test mode and should not be used for manuscript numerical values.
- The processed panel inherits coverage, retrieval, and representativeness limits from satellite SIF, gridded meteorology, soil-moisture products, crop maps, and administrative yield data. The model should therefore be interpreted as an attribution analysis for the packaged African rainfed-maize analysis domain, not as a bias-free representation of all African cropping systems.

# Figure 4 XGBoost Model Details

## Model

- Model type: XGBoost regression.
- Purpose: attribution of 8-day SIF anomaly variability across African cropland climate zones.
- Target variable: `sif_anom`, an 8-day deseasonalized SIF anomaly.
- Observation unit: admin2/county-like unit by 8-day growing-season window.
- Pretrained model: none.

## Predictors

Functional groups are defined in `training/config.yml`.

- Energy: `VPDa_8mean`, `Tmaxa_8mean`, `SWa_8mean`.
- Surface Water: `SMa_L1_8mean`, `PPTa_8sum`.
- Root Water: `SMa_L2_8mean`, `SMa_L3_8mean`.

The retraining input is `training/data/figure4_retraining_input.csv.gz`.

## Train/Test Split and Validation

- Held-out split: grouped 80/20 train/test split.
- Grouping variable: `county_year`, constructed from `admin2_idx` and `year`.
- Stratification: by year within each Koppen climate zone.
- Cross-validation: 5-fold `GroupKFold` within each Koppen zone.
- Early stopping: configured in `training/config.yml`.
- Random seed: `random_seed` in `training/config.yml`.

The retraining workflow writes `training/outputs/train_test_split_ids.csv` and `training/outputs/groupkfold_fold_ids.csv`, and checks that no county-year group appears in both train and test or in more than one validation fold.
It also writes `training/outputs/s8_model_predictions.csv`, containing held-out observed and predicted SIF anomalies for reproducing the Supplementary Figure S8 predicted-versus-observed performance panel.

## Metrics and Interpretation

- Primary metric: weighted held-out test R2.
- Sample weights: soft energy-availability weights derived from raw shortwave radiation and maximum temperature, as configured in `training/config.yml`.
- Additional metrics: train R2, test MAE, test RMSE, cross-validation R2 summaries.
- Ablation design: drop-column Delta R2 for individual predictors and functional groups.
- Bootstrap CIs: bootstrap resampling of held-out rows with replacement.
- Shapley R2 sensitivity: group-level values are packaged in `supplement/data/shapley_r2/`.
- Partial R2: single-feature XGBoost fits are reported as an additional sensitivity.

## Baseline and Sensitivity Context

No community benchmark dataset is used for this attribution task. Model evaluation and sensitivity checks use:

- grouped held-out test performance and 5-fold grouped cross-validation;
- drop-column and group ablation relative to the full XGBoost model;
- hard energy-filtering sensitivity;
- GLDAS/GLEAM processed-input robustness checks in the supplement.

## Hard Energy-Filtering Sensitivity

The configured hard energy thresholds are:

- Tmax >= 18 C.
- SW >= 21.6 MJ m-2 day-1.

The hard-filter sensitivity filters observations before splitting and retrains unweighted models.

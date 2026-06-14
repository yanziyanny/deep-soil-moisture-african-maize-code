# Optional ML Retraining

This directory contains the optional Figure 4 XGBoost retraining workflow.

## Scope

- `config.yml`: split, fold, feature-group, early-stopping, bootstrap, and hard-filter settings.
- `ml_pipeline/`: in-repository XGBoost retraining implementation.
- `data/figure4_retraining_input.csv.gz`: packaged analysis-ready 8-day panel used for retraining.
- `run_optional_ml_retraining.py`: command-line entry point.

The workflow uses `training/data/figure4_retraining_input.csv.gz`.

## Commands

Quick smoke run:

```bash
python training/run_optional_ml_retraining.py --quick
```

Full run from the packaged analysis-ready 8-day training panel:

```bash
python training/run_optional_ml_retraining.py --bootstrap-iters 1000
```

Quick mode reduces data size, tree count, cross-validation work, and bootstrap iterations for smoke testing.

The full command uses `training/data/figure4_retraining_input.csv.gz` and the settings in `training/config.yml`. In the config, the default bootstrap setting is:

- `bootstrap.iters: 1000`
- `bootstrap.resample_unit: held_out_rows`
- `bootstrap.confidence_level: 0.95`

The command-line `--bootstrap-iters` argument overrides `bootstrap.iters`.

## Outputs

The workflow writes:

- `training/outputs/model_metrics_by_zone.csv`
- `training/outputs/drop_column_importance_individual.csv`
- `training/outputs/drop_column_importance_group.csv`
- `training/outputs/bootstrap_confidence_intervals.csv`
- `training/outputs/s8_model_predictions.csv`
- `training/outputs/hard_energy_filtering_sensitivity.csv`
- `training/outputs/hard_energy_filtering/summary.csv`
- `training/outputs/hard_energy_filtering/koppen*/results.json`
- `training/outputs/hard_energy_filtering/figure_dropcol_combined_nature_v3.png`
- `training/outputs/train_test_split_ids.csv`
- `training/outputs/groupkfold_fold_ids.csv`
- `training/outputs/run_metadata.json`
- `training/outputs/figure4_data/summary.csv`
- `training/outputs/figure4_data/koppen*/results.json`

By default, retraining leaves the packaged Figure 4 inputs in `figure4/data/` unchanged and writes regenerated files under `training/outputs/figure4_data/`.

Run full retraining and also replace the packaged Figure 4 inputs:

```bash
python training/run_optional_ml_retraining.py --bootstrap-iters 1000 --sync-figure-data
```

## Split Controls

- The split group is `county_year`, constructed from `admin2_idx` and `year`.
- Train/test split is grouped, with all observations from the same county-year kept together.
- Split assignment is stratified by year within each Koppen climate zone.
- GroupKFold validation folds are assigned within each Koppen zone.
- The workflow checks that no county-year group appears in both train and test or in more than one validation fold.

## Sample Weights

The Figure 4 attribution model uses soft energy-availability sample weights defined in `training/config.yml` from `SW_8mean_raw` and `Tmax_8mean_raw`.

## Hard Energy Filtering

The hard-filter sensitivity filters observations before splitting, then retrains unweighted models. Thresholds are configured in `training/config.yml`: `SW_8mean_raw >= 21.6` MJ m-2 day-1 and `Tmax_8mean_raw >= 18` C.

## Shapley R2 Retraining

Supplementary Figure S9 uses the original Shapley R2 coalition workflow: train all seven non-empty group coalitions, evaluate held-out R2, and bootstrap held-out rows without retraining the coalition models.

Run:

```bash
python training/run_s9_shapley_r2.py --bootstrap-iters 1000
```

Outputs:

- `training/outputs/shapley_r2/summary.csv`
- `training/outputs/shapley_r2/s9_shapley_group_decomposition.csv`
- `training/outputs/shapley_r2/koppen*/results.json`
- `training/outputs/shapley_r2/figure_shapley_r2_stacked.png`

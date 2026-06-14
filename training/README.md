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
- `training/outputs/hard_energy_filtering_sensitivity.csv`
- `training/outputs/train_test_split_ids.csv`
- `training/outputs/groupkfold_fold_ids.csv`
- `training/outputs/run_metadata.json`
- `training/outputs/figure4_data/summary.csv`
- `training/outputs/figure4_data/koppen*/results.json`

The full non-quick workflow syncs regenerated Figure 4 packaged outputs into `figure4/data/`. Quick mode leaves `figure4/data/` unchanged unless `--sync-figure-data` is explicitly passed.

Run full retraining without replacing the packaged Figure 4 inputs:

```bash
python training/run_optional_ml_retraining.py --bootstrap-iters 1000 --no-sync-figure-data
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

The analysis-ready input contains raw VPD, Tmax, and SW columns for the hard energy-filtering sensitivity. Thresholds are configured in `training/config.yml`; the retraining code converts threshold units when the packaged raw-column range indicates an equivalent unit such as hPa for VPD or MJ m-2 day-1 for SW.

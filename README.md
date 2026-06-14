# Deep soil moisture reveals hidden water stress in African rainfed maize systems

This repository contains code and analysis-ready processed data required to reproduce the five main-text figures for the manuscript *Deep soil moisture reveals hidden water stress in African rainfed maize systems*.

The repository is self-contained for reproducing the submitted figures, supplement summaries, and Figure 4 machine-learning retraining from packaged analysis-ready inputs. It does not include the large upstream raw remote-sensing archives used to construct these processed panels.

## Repository Purpose

Included:

- one plotting script for each main-text figure
- processed figure-ready inputs required to regenerate the final figures
- a top-level `run_all_figures.py` entry point
- optional XGBoost retraining support for the Figure 4 attribution analysis
- split/fold leakage checks, a model card, supplement table support, tests, and smoke checks

Not included:

- full raw remote-sensing, climate, or yield-source archives
- heavy upstream preprocessing pipelines
- manuscript PDFs or manuscript editing scripts
- mandatory retraining for default figure reproduction

## Analysis-Ready Data

The Figure 4 retraining input is:

`training/data/figure4_retraining_input.csv.gz`

This file is an 8-day processed panel with 1,000,127 rows. Each row represents an admin2/county-like unit in a given year and 8-day growing-season window. It contains:

- ID and grouping fields: `admin2_idx`, `year`, `koppen5`
- target variable: `sif_anom`
- energy predictors: `VPDa_8mean`, `SWa_8mean`, `Tmaxa_8mean`
- water predictors: `SMa_L1_8mean`, `SMa_L2_8mean`, `SMa_L3_8mean`, `PPTa_8sum`
- raw energy fields for hard-filter sensitivity: `VPD_8mean_raw`, `SW_8mean_raw`, `Tmax_8mean_raw`

The default figure scripts also use packaged figure-ready inputs under `figure*/data/`, shared map layers under `common/map_layers/`, and supplement processed inputs under `supplement/data/`. See `DATA_DICTIONARY.md` for file-level documentation.

## Install

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
```

For the optional ML workflow:

```bash
pip install -r training/requirements-ml.txt
```

A conda-style environment file is also provided:

```bash
conda env create -f environment.yml
conda activate deep-soil-moisture-maize
```

`requirements-lock.txt` records versions observed in the development environment.

## Main-Text Figure Reproduction

Run all main-text figures:

```bash
python run_all_figures.py
```

Expected outputs:

- `figure1/outputs/figure1_vpd_soil_moisture_correlation.png`
- `figure2/outputs/figure2_yield_response_and_mismatch_maps.png`
- `figure3/outputs/figure3_panel_effect_estimates.png`
- `figure4/outputs/figure4_climate_zone_driver_importance.png`
- `figure5/outputs/figure5_monitoring_blind_spot_risk.png`

Run a lightweight import/path smoke check without rendering figures:

```bash
python run_all_figures.py --smoke
```

Figure 4 is plotted by default from the packaged model-result summaries in `figure4/data/summary.csv` and `figure4/data/koppen*/results.json`. This keeps the default figure-reproduction path lightweight and deterministic.

## Optional Figure 4 ML Retraining

Quick smoke run:

```bash
python training/run_optional_ml_retraining.py --quick
```

Full optional run from the packaged analysis-ready 8-day training panel:

```bash
python training/run_optional_ml_retraining.py --bootstrap-iters 1000
```

The retraining input is `training/data/figure4_retraining_input.csv.gz`. Bootstrap, split, cross-validation, feature-group, sample-weight, and hard-filter settings are defined in `training/config.yml`. The default bootstrap setting is `bootstrap.iters: 1000`, resampling held-out rows with replacement. The command-line `--bootstrap-iters` value overrides the config value.

Expected machine-readable outputs:

- `training/outputs/model_metrics_by_zone.csv`
- `training/outputs/drop_column_importance_individual.csv`
- `training/outputs/drop_column_importance_group.csv`
- `training/outputs/bootstrap_confidence_intervals.csv`
- `training/outputs/train_test_split_ids.csv`
- `training/outputs/groupkfold_fold_ids.csv`
- `training/outputs/run_metadata.json`
- `training/outputs/figure4_data/summary.csv`
- `training/outputs/figure4_data/koppen*/results.json`

The full non-quick workflow also syncs regenerated packaged Figure 4 data to:

- `figure4/data/summary.csv`
- `figure4/data/koppen1/results.json`
- `figure4/data/koppen2/results.json`
- `figure4/data/koppen3/results.json`
- `figure4/data/koppen4/results.json`
- `figure4/data/koppen5/results.json`

`--quick` intentionally leaves `figure4/data/` unchanged unless `--sync-figure-data` is provided. Full optional ML retraining is heavier than default figure reproduction.

To audit retraining without replacing the packaged Figure 4 inputs, run:

```bash
python training/run_optional_ml_retraining.py --bootstrap-iters 1000 --no-sync-figure-data
```

The default `python run_all_figures.py` path uses packaged processed outputs and does not rerun model training. Reviewers who want to reproduce Figure 4 from training can run `python training/run_optional_ml_retraining.py --bootstrap-iters 1000`; that command writes split IDs, validation folds, bootstrap confidence intervals, and model metrics.

## Supplement Support

Run supplement table reproduction:

```bash
python supplement/run_all_supplement.py --quick
```

Expected outputs:

- `supplement/outputs/s3_gleam_sm_vpd_coupling_summary.csv`
- `supplement/outputs/s4_gldas_sm_vpd_coupling_summary.csv`
- `supplement/outputs/s5_gldas_yield_response_coefficients.csv`
- `supplement/outputs/s6_gldas_yield_sensitivity_r2.csv`
- `supplement/outputs/s7_gldas_sif_attribution_summary.csv`
- `supplement/outputs/s8_xgboost_model_performance.csv`
- `supplement/outputs/s9_shapley_feature_decomposition.csv`
- `supplement/outputs/s9_shapley_group_decomposition.csv`
- `supplement/outputs/s9_partial_r2.csv`
- `supplement/outputs/s10_hard_energy_filtering_sensitivity.csv`
- `supplement/outputs/supplement_run_report.json`

Supplement processed inputs are listed in `DATA_DICTIONARY.md`.

## Benchmarking

Run:

```bash
python scripts/benchmark_runtime.py
```

Expected output:

- `training/outputs/benchmark_runtime.json`

Record runtime and hardware values in `COMPUTATIONAL_RESOURCES.md`. To time the smoke workflow only, run:

```bash
python scripts/benchmark_runtime.py --smoke-figures
```

## Nature ML Checklist Support

| Checklist item | Repository support |
| --- | --- |
| Source code | `figure*/run_figure*.py`, `run_all_figures.py`, `training/ml_pipeline/`, `supplement/run_all_supplement.py` |
| Test/processed dataset | `figure*/data/`, `common/map_layers/`, `supplement/data/`, `training/data/figure4_retraining_input.csv.gz` |
| README instructions | this file, `training/README.md`, `training/data/README.md`, `supplement/README.md` |
| Train/test split | `training/config.yml`; generated by optional retraining as `training/outputs/train_test_split_ids.csv` |
| Validation folds | `training/config.yml`; generated by optional retraining as `training/outputs/groupkfold_fold_ids.csv` |
| Model card | `MODEL_CARD.md` |
| Ablation/drop-column | generated by optional retraining as `training/outputs/drop_column_importance_individual.csv` and `training/outputs/drop_column_importance_group.csv` |
| Bootstrap CIs | `training/config.yml`; generated by optional retraining as `training/outputs/bootstrap_confidence_intervals.csv` |
| Shapley sensitivity | `figure4/data/koppen*/results.json`, `supplement/outputs/s9_*` |
| Hard energy filtering | `training/config.yml`; generated by optional retraining as `training/outputs/hard_energy_filtering_sensitivity.csv`; packaged S10 input in `supplement/data/hard_energy_filtering/` |
| Benchmark/baseline context | `MODEL_CARD.md` |
| Computational resources | `COMPUTATIONAL_RESOURCES.md`, `scripts/benchmark_runtime.py` |
| Tests/smoke checks | `tests/` |
| License | `LICENSE` |

## Reproducibility Scope

This repository supports reproduction from packaged analysis-ready processed data. It is not intended to rerun upstream acquisition, reprojection, remote-sensing extraction, or administrative aggregation from raw source archives.

## License

This code package is released under the MIT License. See `LICENSE`.

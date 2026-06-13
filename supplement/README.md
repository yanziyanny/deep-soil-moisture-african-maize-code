# Supplement Reproduction Support

This directory contains lightweight scripts for supplement figures and tables that can be reproduced from packaged processed inputs.

Run:

```bash
python supplement/run_all_supplement.py --quick
```

Currently supported from packaged inputs:

- S3 GLEAM SM-VPD coupling summary table and figure copy.
- S4 GLDAS Noah SM-VPD coupling summary table and figure copy.
- S5 GLDAS Noah nonlinear yield-response coefficient table and figure copy.
- S6 GLDAS Noah yield-sensitivity R2 table and figure copy.
- S7 GLDAS Noah SIF attribution summary table and figure copy.
- S8 XGBoost model performance table from `figure4/data/koppen*/results.json`.
- S9 Shapley feature/group and partial R2 tables from `figure4/data/koppen*/results.json`.
- S10 hard energy-filtering table after optional ML retraining has produced `training/outputs/hard_energy_filtering_sensitivity.csv`.

Skipped items, if any required processed input is missing, are reported in `supplement/outputs/supplement_run_report.json`.

The S3-S7 outputs use processed arrays, CSV/JSON summaries, and figure-ready images under `supplement/data/`; they do not require raw GLDAS, GLEAM, or remote-sensing archives.

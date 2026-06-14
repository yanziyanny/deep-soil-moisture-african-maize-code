# Supplement Reproduction

This directory contains scripts and packaged inputs for reproducing supplement summary tables and copied figure panels.

Run:

```bash
python supplement/run_all_supplement.py --quick
```

Outputs:

- S3 GLEAM SM-VPD coupling summary table and figure copy.
- S4 GLDAS Noah SM-VPD coupling summary table and figure copy.
- S5 GLDAS Noah nonlinear yield-response coefficient table and figure copy.
- S6 GLDAS Noah yield-sensitivity R2 table and figure copy.
- S7 GLDAS Noah SIF attribution summary table and figure copy.
- S8 XGBoost model performance table from `figure4/data/koppen*/results.json`.
- S9 Shapley R2 group decomposition from `supplement/data/shapley_r2/`.
- S10 hard energy-filtering table from `supplement/data/hard_energy_filtering/` or from `training/outputs/hard_energy_filtering_sensitivity.csv` if optional ML retraining has been run.
- `supplement/outputs/supplement_run_report.json`.

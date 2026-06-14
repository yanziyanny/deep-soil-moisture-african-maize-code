# Supplement Figure Reproduction

This directory contains the packaged supplement figure panels and a lightweight copy script.

Run:

```bash
python supplement/run_all_supplement.py --quick
```

Outputs:

- `supplement/outputs/supplementary_figure_s3_gleam_sm_vpd_correlation.png`
- `supplement/outputs/supplementary_figure_s4_gldas_sm_vpd_correlation.png`
- `supplement/outputs/supplementary_figure_s5_gldas_yield_response.png`
- `supplement/outputs/supplementary_figure_s6_gldas_yield_sensitivity.png`
- `supplement/outputs/supplementary_figure_s7_gldas_sif_attribution.png`
- `supplement/outputs/supplementary_figure_s8_sif_pred_vs_obs.png`
- `supplement/outputs/supplementary_figure_s9_shapley_r2_decomposition.png`
- `supplement/outputs/supplementary_figure_s10_hard_energy_filtering.png`
- `supplement/outputs/supplement_run_report.json`.

Default supplement reproduction copies figure-ready panels from `supplement/data/figures/`. Source data for the S9 and S10 panels are provided in `supplement/data/shapley_r2/` and `supplement/data/hard_energy_filtering/`; full retraining commands are documented in `training/README.md`.

Packaged inputs for S8-S10:

| Figure | Default figure input | Source data or retraining output |
| --- | --- | --- |
| S8 | `supplement/data/figures/supplementary_figure_s8_sif_pred_vs_obs.png` | `training/outputs/s8_model_predictions.csv` from optional retraining |
| S9 | `supplement/data/figures/supplementary_figure_s9_shapley_r2_decomposition.png` | `supplement/data/shapley_r2/s9_shapley_group_decomposition.csv`; retraining writes `training/outputs/shapley_r2/` |
| S10 | `supplement/data/figures/supplementary_figure_s10_hard_energy_filtering.png` | `supplement/data/hard_energy_filtering/s10_hard_energy_filtering_sensitivity.csv`; retraining writes `training/outputs/hard_energy_filtering/` |

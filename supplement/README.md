# Supplement Figure Reproduction

This directory contains packaged supplement figure inputs and the runner used to write Supplementary Figures S3-S10.

Run:

```bash
python supplement/run_all_supplement.py --quick
```

Outputs:

- `supplement/outputs/supplementary_figure_s3_gleam_sm_vpd_correlation.png`
- `supplement/outputs/supplementary_figure_s3_gleam_sm_vpd_correlation.pdf`
- `supplement/outputs/supplementary_figure_s4_gldas_sm_vpd_correlation.png`
- `supplement/outputs/supplementary_figure_s4_gldas_sm_vpd_correlation.pdf`
- `supplement/outputs/supplementary_figure_s5_gldas_yield_response.png`
- `supplement/outputs/supplementary_figure_s5_gldas_yield_response.pdf`
- `supplement/outputs/supplementary_figure_s6_gldas_yield_sensitivity.png`
- `supplement/outputs/supplementary_figure_s6_gldas_yield_sensitivity.pdf`
- `supplement/outputs/supplementary_figure_s7_gldas_sif_attribution.png`
- `supplement/outputs/supplementary_figure_s7_gldas_sif_attribution.pdf`
- `supplement/outputs/supplementary_figure_s8_sif_pred_vs_obs.png`
- `supplement/outputs/supplementary_figure_s8_sif_pred_vs_obs.pdf`
- `supplement/outputs/supplementary_figure_s9_shapley_r2_decomposition.png`
- `supplement/outputs/supplementary_figure_s9_shapley_r2_decomposition.pdf`
- `supplement/outputs/supplementary_figure_s10_hard_energy_filtering.png`
- `supplement/outputs/supplementary_figure_s10_hard_energy_filtering.pdf`
- `supplement/outputs/supplement_run_report.json`.

Default supplement reproduction redraws S3-S10 from packaged processed source data.

Packaged inputs:

| Figure | Default source data | Related retraining output |
| --- | --- | --- |
| S3-S4 | `supplement/data/coupling/*.npy` | not applicable |
| S5-S6 | `supplement/data/gldas_yield/` | not applicable |
| S7 | `supplement/data/gldas_sif/koppen*/results.json` | not applicable |
| S8 | `supplement/data/sif_predictions/s8_sif_pred_vs_obs.csv.gz` | `training/outputs/s8_model_predictions.csv` |
| S9 | `supplement/data/shapley_r2/s9_shapley_group_decomposition.csv` | `training/outputs/shapley_r2/` |
| S10 | `supplement/data/hard_energy_filtering/koppen*/results.json` | `training/outputs/hard_energy_filtering/` |

Full retraining commands are documented in `training/README.md`.

# Data Dictionary

This file lists the packaged inputs used by the figure, supplement, and optional retraining commands.

## Analysis-Ready Dataset

The packaged analysis-ready dataset consists of:

- `figure*/data/`: figure-ready main-text inputs.
- `common/map_layers/`: shared processed map layers used by the plotting scripts.
- `training/data/figure4_retraining_input.csv.gz`: 8-day processed panel for optional Figure 4 ML retraining.

## Packaged Inputs

| Path | Purpose | Source analysis/figure | Key columns or arrays | Units | Data level | Required for |
| --- | --- | --- | --- | --- | --- | --- |
| `figure1/data/vpd_sm_correlation_layers.npz` | Correlation rasters for VPD vs soil moisture by layer | Figure 1 | arrays `r1`, `r2`, `r3` | correlation r, dimensionless | processed summary array | default figure reproduction |
| `common/map_layers/africa_coastline.geojson` | Africa coastline overlay | Figures 1, 2, 5 | geometry | geographic coordinates | processed map layer | default figure reproduction |
| `common/map_layers/admin2_boundaries.geojson` | Admin2 boundaries | Figures 2, 5 | `admin2_idx`, `ADMIN0`, geometry | geographic coordinates | processed map layer | default figure reproduction |
| `figure2/data/panel_a_exposure_coefficients.csv` | Precomputed exposure coefficients used for panel A display | Figure 2 | `decile`, `L1_*`, `L2L3_*`, `VPD_*` | log points and percent change | summary data | default figure reproduction |
| `figure2/data/panel_a_regression_input.csv.gz` | Processed yield/exposure panel for recomputing panel A coefficients | Figure 2 | `admin2_idx`, `country_id`, `year`, `log_yield`, `L1_pct*`, `L2L3_pct*`, `VPD_pct*` | log yield, exposure percent by decile | processed panel | default figure reproduction |
| `figure2/data/panel_b_case_chad_2010.csv` | Chad 2010 case-study time series | Figure 2 | `day_of_season`, `sm_l1`, `sm_l2l3`, `vpd` | standardized/processed seasonal series | processed case-study data | default figure reproduction |
| `figure2/data/panel_c_case_malawi_2002.csv` | Malawi 2002 case-study time series | Figure 2 | `day_of_season`, `sm_l1`, `sm_l2l3`, `vpd` | standardized/processed seasonal series | processed case-study data | default figure reproduction |
| `figure2/data/panel_c_case_metadata.json` | Metadata for Malawi case-study annotation | Figure 2 | `country`, `year`, `admin2_idx`, `vpd_root_corr` | mixed | summary metadata | default figure reproduction |
| `figure2/data/panel_de_map_frequencies.csv` | Admin2 mismatch frequencies for maps | Figure 2 | `admin2_idx`, `ineffective_freq`, `opposite_freq` | frequency/proportion | processed summary table | default figure reproduction |
| `figure3/data/admin2_year_panel_v3.csv` | Annual admin2 yield and growing-season hydroclimate panel | Figure 3 | `admin2_idx`, `country_id`, `year`, `log_yield`, `yield_kg_ha`, `vpd_gs_mean`, `ppt_gs_sum`, `tmax_gs_mean`, `rad_gs_mean`, `sm_*_gs_mean` | yield kg ha-1, log yield, processed climate/soil-moisture summaries | processed panel | default figure reproduction |
| `figure3/data/admin2_year_panel_stages_v3.csv` | Annual panel with crop-stage variables | Figure 3 | annual fields plus `*_veg`, `*_rep`, `*_mat` stage columns | processed growing-stage summaries | processed panel | default figure reproduction |
| `figure4/data/summary.csv` | Packaged Figure 4 climate-zone model summary | Figure 4 | `koppen_id`, `test_r2`, `delta_energy`, `delta_surface`, `delta_rootwater`, `dominance`, `top_var` | R2 and Delta R2 | summary data | default figure reproduction |
| `figure4/data/koppen*/results.json` | Packaged model results by climate zone | Figure 4, Supplement S8-S9 | `metrics`, `drop_group_delta`, `drop_column_delta`, bootstrap CI fields, `shap_feature`, `shap_group`, `partial_r2` | R2, Delta R2, model metrics | summary data | default figure and supplement reproduction |
| `figure5/data/monitoring_blind_spots_data.csv` | Monitoring blind-spot risk summaries by admin2 | Figure 5 | `admin2_idx`, `r2_full`, `delta_*`, `dominance_class`, `koppen_id`, `total_mismatch_freq`, `risk_class`, `avg_production` | R2, Delta R2, frequencies, production summary | processed summary table | default figure reproduction |
| `training/data/figure4_retraining_input.csv.gz` | Analysis-ready 8-day panel for optional ML retraining | Figure 4 retraining | `admin2_idx`, `year`, `koppen5`, `sif_anom`, `VPDa_8mean`, `SWa_8mean`, `Tmaxa_8mean`, `SMa_L1_8mean`, `SMa_L2_8mean`, `SMa_L3_8mean`, `PPTa_8sum`, `VPD_8mean_raw`, `Tmax_8mean_raw`, `SW_8mean_raw` | SIF anomaly, environmental anomalies, precipitation-sum anomaly, raw VPD/Tmax/SW for hard filtering | processed 8-day panel | optional ML retraining |
| `supplement/data/coupling/*.npy` | Processed SM-VPD correlation rasters for GLEAM and GLDAS robustness checks | Supplementary Figures S3-S4 | finite Pearson correlation arrays | correlation r, dimensionless | processed summary arrays | supplement reproduction |
| `supplement/data/figures/s3_*` through `s7_*` | Figure-ready robustness panels copied by the supplement runner | Supplementary Figures S3-S7 | PNG/PDF image files | figure-ready visual outputs | processed figure outputs | supplement reproduction |
| `supplement/data/gldas_yield/s5_gldas_exposure_coefficients.csv` | GLDAS Noah nonlinear yield-response coefficients | Supplementary Figure S5 | `decile`, `L1_pct`, `L2L3_pct`, `VPD_pct`, standard-error columns | percent yield change | processed summary table | supplement reproduction |
| `supplement/data/gldas_yield/s6_gldas_r2_results.json` | GLDAS Noah yield-regression R2 summaries | Supplementary Figure S6 | `pooled_models`, `growth_stage_models`, `within_r2`, `overall_r2`, `n_observations` | R2, observation counts | processed summary JSON | supplement reproduction |
| `supplement/data/gldas_sif/koppen*/results.json` | GLDAS Noah SIF attribution summaries by climate zone | Supplementary Figure S7 | `metrics`, `drop_group_delta`, `drop_column_delta`, `shap_feature`, `shap_group` | R2, Delta R2, attribution summaries | processed summary JSON | supplement reproduction |
| `supplement/data/hard_energy_filtering/s10_hard_energy_filtering_sensitivity.csv` | Hard energy-filtering sensitivity summary | Supplementary Figure S10 | `status`, `n_test_filtered`, `test_r2_filtered`, `delta_*_filtered` | R2, Delta R2, observation counts | processed summary table | supplement reproduction |

## Generated Outputs

Generated outputs are written under `figure*/outputs/`, `training/outputs/`, and `supplement/outputs/`.

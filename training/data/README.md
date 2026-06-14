# Training Data README

## Packaged File

Path: `training/data/figure4_retraining_input.csv.gz`

Purpose: analysis-ready 8-day panel for optional XGBoost retraining of the Figure 4 SIF anomaly attribution analysis.

Rows: 1,000,127.

Data level: processed 8-day retraining panel.

Each row represents an admin2/county-like unit in one year and one 8-day growing-season window.

Required for: optional ML retraining.

## Required Schema

| Column | Role | Units/meaning |
| --- | --- | --- |
| `admin2_idx` | admin/county-like unit identifier | integer ID |
| `year` | year and split stratification variable | calendar year |
| `koppen5` | climate zone | 1-5 Koppen grouping |
| `sif_anom` | target | deseasonalized SIF anomaly |
| `VPDa_8mean` | Energy predictor | VPD anomaly |
| `SWa_8mean` | Energy predictor | shortwave radiation anomaly |
| `Tmaxa_8mean` | Energy predictor | maximum temperature anomaly |
| `PPTa_8sum` | Surface Water predictor | 8-day precipitation-sum anomaly used by the retraining code |
| `SMa_L1_8mean` | Surface Water predictor | layer-1 soil moisture anomaly |
| `SMa_L2_8mean` | Root Water predictor | layer-2 soil moisture anomaly |
| `SMa_L3_8mean` | Root Water predictor | layer-3 soil moisture anomaly |
| `Tmax_8mean_raw` | hard-filter support | raw Tmax in C |
| `VPD_8mean_raw` | hard-filter support | raw VPD; values are treated as hPa when the range indicates hPa-like units |
| `SW_8mean_raw` | hard-filter support | raw shortwave radiation; values are treated as convertible to MJ m-2 day-1 when needed |

The workflow constructs `county_year` from `admin2_idx` and `year`.

## Hard Energy Filtering

The packaged panel includes the raw VPD, Tmax, and SW columns needed by the hard energy-filtering sensitivity. `training/config.yml` stores the manuscript thresholds as Tmax > 15 C, VPD > 0.5 kPa, and SW > 200 W m-2. The retraining code converts thresholds to the packaged raw-column units when the value range indicates an equivalent unit such as hPa for VPD or MJ m-2 day-1 for SW.

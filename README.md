# Nature Food Code Availability Package

This folder is a hybrid code-availability package for the five main-text figures used in the Nature Food submission. The default path is a lightweight figure-reproduction package; an optional `training/` path is included for retraining the machine-learning-dependent analyses behind Figure 4 and Figure 5.

Only code, documentation, shared map layers, and packaged processed data are intended to stay in this folder. Figure PNGs are generated on demand, and the optional retraining wrappers use temporary directories for upstream artifacts before syncing only the final packaged data back into this folder.

## Scope

- `figure1/`: VPD-soil moisture correlation figure
- `figure2/`: Yield response, case studies, and mismatch maps
- `figure3/`: Panel fixed-effects coefficient figure
- `figure4/`: Drop-column importance across Koppen zones
- `figure5/`: Monitoring blind spots map
- `training/`: optional ML retraining wrappers and data-sync scripts for Figure 4 and Figure 5
- `common/map_layers/`: Shared Africa admin2 boundaries and coastline layers used by map panels

## What Is Included

- One plotting script per figure
- Only the minimum processed data needed to regenerate the final figure
- A top-level `run_all_figures.py` entry point
- PNG outputs only
- Optional ML retraining wrappers for the model-dependent figures

## What Is Not Included

- Full raw upstream remote-sensing archives
- Heavy preprocessing steps that are not required to reproduce the final submitted figures
- Mandatory retraining for default figure reproduction

## Run

Create an environment with the dependencies in `requirements.txt`, then run:

```bash
python run_all_figures.py
```

You can also run figures individually:

```bash
python figure1/run_figure1.py
python figure2/run_figure2.py
python figure3/run_figure3.py
python figure4/run_figure4.py
python figure5/run_figure5.py
```

For optional ML retraining of the model-dependent figures:

```bash
python training/run_optional_ml_retraining.py
```

## Outputs

- `figure1/outputs/figure1_vpd_soil_moisture_correlation.png`
- `figure2/outputs/figure2_yield_response_and_mismatch_maps.png`
- `figure3/outputs/figure3_panel_effect_estimates.png`
- `figure4/outputs/figure4_climate_zone_driver_importance.png`
- `figure5/outputs/figure5_monitoring_blind_spot_risk.png`

## Data Provenance

The packaged inputs are reduced from the original project files:

- Figure 1: precomputed correlation arrays from `vpd_sm_correlation/`
- Figure 2: processed coefficient table, extracted case-study time series, and precomputed mismatch frequencies
- Figure 3: panel tables from `nature_code/data/`
- Figure 4: bootstrap result summaries from `africa_sif_cluster80_cvmedian/outputs_koppen5_dropcol_bootstrap/`
- Figure 5: precomputed monitoring-blind-spot table from `nature_code/monitoring_blind_spots/`

The default package is suitable for a code-availability repository because it provides figure-ready code plus the minimum processed data needed to regenerate the final published figures. The optional `training/` layer exposes a reproducible retraining path for the ML-dependent results without forcing every reader to run the full heavy analysis.

# Deep soil moisture reveals hidden water stress in African rainfed maize systems

This repository contains the code and processed data required to reproduce the five main-text figures for the manuscript *Deep soil moisture reveals hidden water stress in African rainfed maize systems*.

The repository is organized as a lightweight figure-reproduction package by default. It also includes an optional `training/` workflow for retraining the machine-learning-dependent analysis underlying Fig. 4. Figure outputs are generated on demand as PNG files and are not stored in the repository.

## Repository purpose

This package is designed for code availability and figure reproduction rather than full upstream data processing. It keeps only the scripts, shared map layers and processed inputs needed to regenerate the final figures reported in the manuscript.

Included in this repository:

- one plotting script for each main-text figure
- processed figure-ready inputs required to regenerate the final figures
- a top-level `run_all_figures.py` entry point
- optional retraining support for the machine-learning analysis behind Fig. 4

Not included in this repository:

- full raw remote-sensing archives and upstream intermediate products
- heavy preprocessing pipelines not required to reproduce the submitted figures
- mandatory retraining for default figure reproduction

## Repository structure

- `figure1/`: VPD and soil moisture correlation figure
- `figure2/`: yield response, case studies and mismatch maps
- `figure3/`: panel fixed-effects estimates
- `figure4/`: climate-zone driver importance from drop-column analysis
- `figure5/`: monitoring blind-spot risk maps
- `training/`: optional retraining wrapper and packaged input for Fig. 4
- `common/map_layers/`: shared Africa admin2 boundaries and coastline layers

## Quick start

Install the base dependencies and run all main-text figures:

```bash
pip install -r requirements.txt
python run_all_figures.py
```

To run figures individually:

```bash
python figure1/run_figure1.py
python figure2/run_figure2.py
python figure3/run_figure3.py
python figure4/run_figure4.py
python figure5/run_figure5.py
```

## Optional retraining

The default workflow reproduces the final figures from packaged processed data. For readers who want to rerun the model-dependent analysis behind Fig. 4, an optional retraining workflow is provided in `training/`.

```bash
python training/run_optional_ml_retraining.py
```

The optional retraining path:

- reruns the machine-learning-dependent analysis for Fig. 4 from a packaged reduced 8-day panel table
- uses temporary directories for upstream artifacts so that intermediate files are not written into the repository
- syncs only the final packaged data products back into the figure folders
- retains the manuscript setting of `1000` bootstrap iterations for the Fig. 4 analysis

Additional dependencies for the optional retraining workflow are listed in `training/requirements-ml.txt`.

## Generated outputs

Running the figure scripts generates the following PNG files:

- `figure1/outputs/figure1_vpd_soil_moisture_correlation.png`
- `figure2/outputs/figure2_yield_response_and_mismatch_maps.png`
- `figure3/outputs/figure3_panel_effect_estimates.png`
- `figure4/outputs/figure4_climate_zone_driver_importance.png`
- `figure5/outputs/figure5_monitoring_blind_spot_risk.png`

## Processed data sources

The packaged inputs were reduced from the original project analyses:

- Fig. 1: precomputed correlation arrays derived from the `vpd_sm_correlation/` workflow
- Fig. 2: packaged panel-a regression input, extracted case-study time series and precomputed mismatch frequencies
- Fig. 3: packaged `admin2 × year` annual and stage-specific panel tables used for the fixed-effects regressions
- Fig. 4: packaged bootstrap summaries plus a reduced retraining input table for the Koppen-zone drop-column analysis
- Fig. 5: packaged admin2 monitoring blind-spot summaries derived from the monitoring workflow

## Reproducibility note

This repository is intended to satisfy code-availability and figure-reproduction requirements for journal submission by combining figure-ready scripts with the minimum processed data needed to regenerate the published main-text figures. The optional retraining layer provides a reproducible path for the key model-dependent result in Fig. 4 without requiring every reader to rerun the full heavy workflow.

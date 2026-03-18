# Optional ML Retraining

This directory contains the optional retraining path for the machine-learning-dependent Fig. 4 analysis in the submission package.

## Scope

- `retrain_figure4.py`: reruns the upstream Köppen-zone drop-column bootstrap analysis and syncs the refreshed outputs into `figure4/data/`
- `training/data/figure4_retraining_input.csv.gz`: packaged reduced 8-day panel table used to rerun Fig. 4
- `run_optional_ml_retraining.py`: runs the Fig. 4 retraining step

The upstream Figure 4 script currently uses `1000` bootstrap iterations to match the manuscript description.

## Command

```bash
python training/run_optional_ml_retraining.py
```

## Notes

- This path is optional and much heavier than the default figure-only reproduction run
- It reuses the original project scripts rather than duplicating the ML training logic
- It does not require the original `T_panel_sif_all_8day.csv` file, because the packaged reduced training table is used instead
- Typical heavy dependencies are listed in `training/requirements-ml.txt`
- The retraining wrappers use temporary output directories and sync back only the final packaged data needed by the submission folder

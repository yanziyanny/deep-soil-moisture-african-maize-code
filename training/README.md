# Optional ML Retraining

This directory contains the optional retraining path for the machine-learning-dependent parts of the submission package.

## Scope

- `retrain_figure4.py`: reruns the upstream Köppen-zone drop-column bootstrap analysis and syncs the refreshed outputs into `figure4/data/`
- `retrain_figure5.py`: reruns the upstream Admin2 driver mapping, refreshes the downstream monitoring-blind-spots table, and syncs the refreshed CSV into `figure5/data/`
- `run_optional_ml_retraining.py`: runs both steps in sequence

The upstream Figure 4 script currently uses `1000` bootstrap iterations to match the manuscript description.

## Command

```bash
python training/run_optional_ml_retraining.py
```

## Notes

- This path is optional and much heavier than the default figure-only reproduction run
- It reuses the original project scripts rather than duplicating the ML training logic
- It expects the upstream project data files to remain available in their original locations
- Typical heavy dependencies are listed in `training/requirements-ml.txt`
- The retraining wrappers use temporary output directories and sync back only the final packaged data needed by the submission folder

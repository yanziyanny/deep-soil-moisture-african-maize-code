# Computational Resources

Recorded benchmark environment and commands.

Run:

```bash
python scripts/benchmark_runtime.py
```

The script writes a machine-readable benchmark report to `training/outputs/benchmark_runtime.json`.

## Recorded Environment

- Hardware: Apple Silicon workstation, arm64.
- Operating system: macOS-26.1-arm64.
- Python version: 3.11.8.
- CPU count: 10.
- RAM: 32 GB.
- Runtime for `python run_all_figures.py`: 20.93 s.
- Runtime for `python training/run_optional_ml_retraining.py --quick`: 50.46 s.
- Full optional retraining input size: 1,000,127 rows and 22,735 county-year groups.

## Benchmark Commands

- Main figures: `python run_all_figures.py`
- Optional quick retraining: `python training/run_optional_ml_retraining.py --quick`
- Optional full retraining: `python training/run_optional_ml_retraining.py --bootstrap-iters 1000 --no-sync-figure-data`

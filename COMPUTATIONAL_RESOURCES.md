# Computational Resources

This file records the computing information requested by the Nature Portfolio machine-learning checklist.

Run:

```bash
python scripts/benchmark_runtime.py
```

The script writes a machine-readable benchmark report to `training/outputs/benchmark_runtime.json`.

## Author-Filled Benchmark Summary

- Hardware: Apple Silicon workstation, arm64.
- Operating system: macOS-26.1-arm64.
- Python version: 3.11.8.
- CPU count: 10.
- RAM: 32 GB.
- Runtime for `python run_all_figures.py`: 20.93 s.
- Runtime for `python training/run_optional_ml_retraining.py --quick`: 8.09 s.
- Full optional retraining runtime with `--bootstrap-iters 1000 --no-sync-figure-data`: 92.03 s.
- Full optional retraining input size: 1,000,127 rows and 22,735 county-year groups.

## Notes

- Full optional ML retraining is expected to be heavier than default figure reproduction.
- The benchmark helper records package versions and command return codes.
- If full figure rendering is not run, record the command that was timed.

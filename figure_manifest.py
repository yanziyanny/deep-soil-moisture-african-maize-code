from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


FIGURES = [
    {
        "name": "figure1",
        "script": "figure1/run_figure1.py",
        "output": "figure1/outputs/figure1_vpd_soil_moisture_correlation.png",
    },
    {
        "name": "figure2",
        "script": "figure2/run_figure2.py",
        "output": "figure2/outputs/figure2_yield_response_and_mismatch_maps.png",
    },
    {
        "name": "figure3",
        "script": "figure3/run_figure3.py",
        "output": "figure3/outputs/figure3_panel_effect_estimates.png",
    },
    {
        "name": "figure4",
        "script": "figure4/run_figure4.py",
        "output": "figure4/outputs/figure4_climate_zone_driver_importance.png",
    },
    {
        "name": "figure5",
        "script": "figure5/run_figure5.py",
        "output": "figure5/outputs/figure5_monitoring_blind_spot_risk.png",
    },
]


def package_path(relative_path: str) -> Path:
    return PACKAGE_ROOT / relative_path


ML_RETRAIN_TARGETS = {
    "figure4": {
        "entry_point": "training/run_optional_ml_retraining.py",
        "packaged_training_input": "training/data/figure4_retraining_input.csv.gz",
        "sync_outputs": [
            "figure4/data/summary.csv",
            "figure4/data/koppen1/results.json",
            "figure4/data/koppen2/results.json",
            "figure4/data/koppen3/results.json",
            "figure4/data/koppen4/results.json",
            "figure4/data/koppen5/results.json",
        ],
    },
}

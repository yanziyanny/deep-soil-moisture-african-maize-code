from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


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


_FIGURE4_OUTPUT_ROOT = REPO_ROOT / "africa_sif_cluster80_cvmedian" / "outputs_koppen5_dropcol_bootstrap"
_FIGURE5_OUTPUT_ROOT = REPO_ROOT / "nature_code" / "monitoring_blind_spots"


ML_RETRAIN_TARGETS = {
    "figure4": {
        "upstream_script": str(
            (REPO_ROOT / "africa_sif_cluster80_cvmedian" / "scripts" / "run_koppen5_dropcol_bootstrap.py").resolve()
        ),
        "sync_outputs": [
            "figure4/data/summary.csv",
            "figure4/data/koppen1/results.json",
            "figure4/data/koppen2/results.json",
            "figure4/data/koppen3/results.json",
            "figure4/data/koppen4/results.json",
            "figure4/data/koppen5/results.json",
        ],
        "sync_map": {
            str((_FIGURE4_OUTPUT_ROOT / "summary.csv").resolve()): "figure4/data/summary.csv",
            str((_FIGURE4_OUTPUT_ROOT / "koppen1" / "results.json").resolve()): "figure4/data/koppen1/results.json",
            str((_FIGURE4_OUTPUT_ROOT / "koppen2" / "results.json").resolve()): "figure4/data/koppen2/results.json",
            str((_FIGURE4_OUTPUT_ROOT / "koppen3" / "results.json").resolve()): "figure4/data/koppen3/results.json",
            str((_FIGURE4_OUTPUT_ROOT / "koppen4" / "results.json").resolve()): "figure4/data/koppen4/results.json",
            str((_FIGURE4_OUTPUT_ROOT / "koppen5" / "results.json").resolve()): "figure4/data/koppen5/results.json",
        },
    },
    "figure5": {
        "upstream_script": str(
            (REPO_ROOT / "africa_sif_cluster80_cvmedian" / "scripts" / "run_admin2_driver_mapping.py").resolve()
        ),
        "downstream_refresh_script": str(
            (REPO_ROOT / "nature_code" / "monitoring_blind_spots" / "figure_monitoring_blind_spots.py").resolve()
        ),
        "sync_outputs": [
            "figure5/data/monitoring_blind_spots_data.csv",
        ],
        "sync_map": {
            str((_FIGURE5_OUTPUT_ROOT / "monitoring_blind_spots_data.csv").resolve()): "figure5/data/monitoring_blind_spots_data.csv",
        },
    },
}

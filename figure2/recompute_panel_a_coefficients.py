#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "data" / "panel_a_regression_input.csv.gz"
DEFAULT_OUTPUT = HERE / "data" / "panel_a_exposure_coefficients.csv"

N_BINS = 10
REF_BIN_SM = 0
REF_BIN_VPD = 0


def read_exposure_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["admin2_idx"] = df["admin2_idx"].astype(int)
    df["year"] = df["year"].astype(int)
    return df


def _exog_columns() -> list[str]:
    bins_sm = [i for i in range(N_BINS) if i != REF_BIN_SM]
    bins_vpd = [i for i in range(N_BINS) if i != REF_BIN_VPD]
    return (
        [f"L1_pct{i}" for i in bins_sm]
        + [f"L2L3_pct{i}" for i in bins_sm]
        + [f"VPD_pct{i}" for i in bins_vpd]
    )


def _extract_coefs(results, prefix: str, ref_bin: int) -> tuple[np.ndarray, np.ndarray]:
    coefs = []
    ses = []
    for i in range(N_BINS):
        if i == ref_bin:
            coefs.append(0.0)
            ses.append(0.0)
            continue

        var_name = f"{prefix}_pct{i}"
        coefs.append(float(results.params[var_name]))
        ses.append(float(results.std_errors[var_name]))

    return np.array(coefs), np.array(ses)


def compute_panel_a_coefficients(df_exposure: pd.DataFrame) -> pd.DataFrame:
    df_panel = df_exposure.set_index(["admin2_idx", "year"]).sort_index()
    model = PanelOLS(
        df_panel["log_yield"],
        df_panel[_exog_columns()],
        entity_effects=True,
        time_effects=True,
    )
    results = model.fit(cov_type="clustered", cluster_entity=True)

    coefs_l1, ses_l1 = _extract_coefs(results, "L1", REF_BIN_SM)
    coefs_l2l3, ses_l2l3 = _extract_coefs(results, "L2L3", REF_BIN_SM)
    coefs_vpd, ses_vpd = _extract_coefs(results, "VPD", REF_BIN_VPD)

    pct_l1 = (np.exp(coefs_l1) - 1.0) * 100.0
    pct_se_l1 = np.exp(coefs_l1) * ses_l1 * 100.0
    pct_l2l3 = (np.exp(coefs_l2l3) - 1.0) * 100.0
    pct_se_l2l3 = np.exp(coefs_l2l3) * ses_l2l3 * 100.0
    pct_vpd = (np.exp(coefs_vpd) - 1.0) * 100.0
    pct_se_vpd = np.exp(coefs_vpd) * ses_vpd * 100.0

    return pd.DataFrame(
        {
            "decile": range(1, N_BINS + 1),
            "L1_coef_logpoint": coefs_l1,
            "L1_se_logpoint": ses_l1,
            "L1_pct": pct_l1,
            "L1_pct_se": pct_se_l1,
            "L2L3_coef_logpoint": coefs_l2l3,
            "L2L3_se_logpoint": ses_l2l3,
            "L2L3_pct": pct_l2l3,
            "L2L3_pct_se": pct_se_l2l3,
            "VPD_coef_logpoint": coefs_vpd,
            "VPD_se_logpoint": ses_vpd,
            "VPD_pct": pct_vpd,
            "VPD_pct_se": pct_se_vpd,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute Figure 2 panel-a coefficients from packaged exposure data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Packaged panel-a regression input table.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path for recomputed coefficients.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df_exposure = read_exposure_input(args.input)
    df_coef = compute_panel_a_coefficients(df_exposure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_coef.to_csv(args.output, index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

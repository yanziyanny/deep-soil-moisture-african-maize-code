#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_FILE = HERE / "outputs" / "figure4_climate_zone_driver_importance.png"

FEATURE_GROUP = {
    "VPDa_8mean": "Energy",
    "SWa_8mean": "Energy",
    "Tmaxa_8mean": "Energy",
    "SMa_L1_8mean": "SurfaceWater",
    "PPTa_8mean": "SurfaceWater",
    "PPTa_8sum": "SurfaceWater",
    "SMa_L2_8mean": "RootWater",
    "SMa_L3_8mean": "RootWater",
}

COLOR_MAP = {
    "RootWater": "#1B7837",
    "SurfaceWater": "#92C5DE",
    "Energy": "#D95F02",
}

VAR_LABELS = {
    "VPDa_8mean": "VPD",
    "SWa_8mean": "SW",
    "Tmaxa_8mean": "Tmax",
    "SMa_L1_8mean": "SM L1",
    "PPTa_8mean": "PPT",
    "PPTa_8sum": "PPT",
    "SMa_L2_8mean": "SM L2",
    "SMa_L3_8mean": "SM L3",
}

KOPPEN_NAMES = {
    1: "Tropical",
    2: "Savanna",
    3: "Desert",
    4: "Semi-arid",
    5: "Temperate",
}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_results():
    results = {}
    for k_id in range(1, 6):
        with open(DATA_DIR / f"koppen{k_id}" / "results.json", "r") as handle:
            results[k_id] = json.load(handle)
    return results


def has_bootstrap_data(results):
    return "drop_column_bootstrap" in results.get(1, {})


def main():
    HERE.joinpath("outputs").mkdir(parents=True, exist_ok=True)

    results = load_results()
    with_bootstrap = has_bootstrap_data(results)

    fig = plt.figure(figsize=(9.0, 6.2))
    gs = fig.add_gridspec(
        2,
        5,
        height_ratios=[1, 0.50],
        hspace=0.38,
        wspace=0.55,
        left=0.08,
        right=0.98,
        top=0.92,
        bottom=0.10,
    )
    panel_labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

    fig.text(-0.025, 0.66, "Individual variables", fontsize=16, fontweight="bold", ha="center", va="center", rotation=90, color="#333333")
    for idx, k_id in enumerate(range(1, 6)):
        ax = fig.add_subplot(gs[0, idx])
        if with_bootstrap:
            boot = results[k_id]["drop_column_bootstrap"]
            items = sorted(boot.items(), key=lambda x: x[1]["mean"], reverse=True)
            labels = [VAR_LABELS[var] for var, _ in items]
            values = [stats["mean"] for _, stats in items]
            errors = [(stats["mean"] - stats["ci_low"], stats["ci_high"] - stats["mean"]) for _, stats in items]
            errors = np.array(errors).T
            colors = [COLOR_MAP[FEATURE_GROUP[var]] for var, _ in items]
        else:
            raw = results[k_id]["drop_column_delta"]
            items = sorted(raw.items(), key=lambda x: x[1], reverse=True)
            labels = [VAR_LABELS[var] for var, _ in items]
            values = [val for _, val in items]
            errors = None
            colors = [COLOR_MAP[FEATURE_GROUP[var]] for var, _ in items]

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, color=colors, height=0.68, edgecolor="none")
        if with_bootstrap and errors is not None:
            ax.errorbar(values, y_pos, xerr=errors, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2, capthick=0.6, clip_on=False)

        baseline = results[k_id].get("baseline", results[k_id].get("metrics", {}))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=16)
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linestyle="-", linewidth=0.3, alpha=0.5, color="#CCCCCC")
        ax.set_title(f"{KOPPEN_NAMES[k_id]}\n($R^2$={baseline.get('test_r2', float('nan')):.2f})", fontsize=17, pad=8)
        ax.text(-0.12, 1.12, panel_labels[idx], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
        ax.set_xlim(0, 0.08)
        ax.set_xticks([0, 0.04, 0.08])
        ax.set_xticklabels(["0", "0.04", "0.08"], fontsize=14)

    fig.text(-0.025, 0.22, "Variable groups", fontsize=16, fontweight="bold", ha="center", va="center", rotation=90, color="#333333")
    group_order = ["RootWater", "Energy", "SurfaceWater"]
    group_labels = ["Root", "Energy", "Surface"]

    for idx, k_id in enumerate(range(1, 6)):
        ax = fig.add_subplot(gs[1, idx])
        if with_bootstrap:
            boot = results[k_id]["drop_group_bootstrap"]
            values = [boot.get(group, {}).get("mean", 0) for group in group_order]
            errors = np.array(
                [
                    [boot.get(group, {}).get("mean", 0) - boot.get(group, {}).get("ci_low", 0) for group in group_order],
                    [boot.get(group, {}).get("ci_high", 0) - boot.get(group, {}).get("mean", 0) for group in group_order],
                ]
            )
        else:
            raw = results[k_id]["drop_group_delta"]
            values = [raw.get(group, 0) for group in group_order]
            errors = None

        y_pos = np.arange(len(group_order))
        ax.barh(y_pos, values, color=[COLOR_MAP[group] for group in group_order], height=0.65, edgecolor="none")
        if with_bootstrap and errors is not None:
            ax.errorbar(values, y_pos, xerr=errors, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2.5, capthick=0.6, clip_on=False)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(group_labels if idx == 0 else [], fontsize=16)
        ax.invert_yaxis()
        ax.set_xlim(0, 0.15)
        ax.set_xticks([0, 0.075, 0.15])
        ax.set_xticklabels(["0", "0.075", "0.15"], fontsize=14)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linestyle="-", linewidth=0.3, alpha=0.5, color="#CCCCCC")
        ax.text(-0.12, 1.20, panel_labels[idx + 5], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    fig.text(0.54, 0.005, r"$\Delta R^2$", fontsize=18, ha="center", va="bottom")
    fig.legend(
        handles=[
            mpatches.Patch(facecolor=COLOR_MAP["RootWater"], edgecolor="none", label="Root Water (SM L2, L3)"),
            mpatches.Patch(facecolor=COLOR_MAP["Energy"], edgecolor="none", label="Energy (VPD, SW, Tmax)"),
            mpatches.Patch(facecolor=COLOR_MAP["SurfaceWater"], edgecolor="none", label="Surface Water (SM L1, PPT)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, 1.12),
        frameon=False,
        fontsize=16,
        ncol=3,
        handlelength=1.3,
        handletextpad=0.5,
        columnspacing=1.8,
    )

    fig.savefig(OUT_FILE, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()

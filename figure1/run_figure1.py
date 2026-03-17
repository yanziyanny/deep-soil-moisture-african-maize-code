#!/usr/bin/env python3

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_FILE = HERE / "data" / "vpd_sm_correlation_layers.npz"
COAST_FILE = ROOT / "common" / "map_layers" / "africa_coastline.geojson"
OUT_FILE = HERE / "outputs" / "figure1_vpd_soil_moisture_correlation.png"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 0.6,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }
)


def load_arrays():
    loaded = np.load(DATA_FILE)
    arrays = []
    for key in ["r1", "r2", "r3"]:
        arr = loaded[key].astype(float)
        arr[~np.isfinite(arr)] = np.nan
        np.clip(arr, -1, 1, out=arr)
        arrays.append(arr)
    return arrays


def sample_values(arr, rng, n_sample=30000):
    values = arr[np.isfinite(arr)].ravel()
    if len(values) == 0:
        return np.array([])
    if len(values) <= n_sample:
        return values
    return rng.choice(values, size=n_sample, replace=False)


def main():
    HERE.joinpath("outputs").mkdir(parents=True, exist_ok=True)

    r1, r2, r3 = load_arrays()
    coast = gpd.read_file(COAST_FILE)

    rng = np.random.default_rng(42)
    violin_data = [sample_values(arr, rng) for arr in [r1, r2, r3]]
    medians = [float(np.nanmedian(arr)) for arr in [r1, r2, r3]]

    lat = np.linspace(40, -40, r1.shape[0])
    lon = np.linspace(-20, 55, r1.shape[1])

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.2, 2.8),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.04], "wspace": 0.05},
        facecolor="white",
    )

    map_axes = axes[:3]
    cax = axes[3]
    titles = ["L1: 0-7 cm", "L2: 7-28 cm", "L3: 28-100 cm"]
    labels = ["a", "b", "c"]
    violin_colors = ["#08519c", "#3182bd", "#6baed6"]
    cmap = plt.cm.Blues_r
    vmin, vmax = -0.8, 0.0

    for i, (ax, arr, vals, med) in enumerate(zip(map_axes, [r1, r2, r3], violin_data, medians)):
        mesh = ax.pcolormesh(
            lon,
            lat,
            arr,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            rasterized=True,
        )
        coast.plot(ax=ax, color="none", edgecolor="#333333", linewidth=0.35, zorder=3)

        ax.set_xlim(-20, 55)
        ax.set_ylim(-40, 40)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.set_title(titles[i], fontsize=10, pad=5, style="italic")
        ax.text(-0.08, 1.10, labels[i], transform=ax.transAxes, fontsize=12, fontweight="bold")
        ax.set_xticks([0, 30])
        ax.set_yticks([-30, 0, 30])
        if i > 0:
            ax.set_yticklabels([])
        ax.tick_params(length=2.5, width=0.5, colors="#444444")

        inset = ax.inset_axes([0.17, 0.06, 0.15, 0.45])
        inset.patch.set_facecolor("white")
        inset.patch.set_alpha(0.9)

        vp = inset.violinplot([vals], positions=[0], widths=0.8, showextrema=False, showmedians=False)
        for body in vp["bodies"]:
            body.set_facecolor(violin_colors[i])
            body.set_edgecolor("white")
            body.set_alpha(0.9)
            body.set_linewidth(1.0)

        inset.hlines(med, -0.3, 0.3, colors="white", linewidth=2.4, zorder=4)
        inset.hlines(med, -0.3, 0.3, colors="#222222", linewidth=1.4, zorder=5)
        inset.axhline(0, color="#666666", linestyle="--", linewidth=0.6, zorder=2)
        inset.text(0.45, med, f"{med:.2f}", fontsize=7.5, fontweight="bold", va="center", ha="left")

        inset.set_xlim(-0.6, 0.9)
        inset.set_ylim(-0.95, 0.15)
        inset.set_yticks([-0.8, -0.4, 0.0])
        inset.set_xticks([])
        inset.tick_params(axis="y", labelsize=7, length=2, width=0.5, pad=2)
        inset.spines["top"].set_visible(False)
        inset.spines["right"].set_visible(False)
        inset.spines["bottom"].set_visible(False)
        inset.spines["left"].set_linewidth(0.5)

    cbar = fig.colorbar(mesh, cax=cax, orientation="vertical", extend="min")
    cbar.set_ticks([0.0, -0.2, -0.4, -0.6, -0.8])
    cbar.set_label("Correlation (r)", fontsize=9, labelpad=5)
    cbar.ax.tick_params(labelsize=8, width=0.6, length=3)
    cbar.outline.set_linewidth(0.6)

    fig.savefig(OUT_FILE, dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()

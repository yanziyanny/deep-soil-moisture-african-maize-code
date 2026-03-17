#!/usr/bin/env python3

from pathlib import Path

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BOUNDARY_FILE = ROOT / "common" / "map_layers" / "admin2_boundaries.geojson"
COAST_FILE = ROOT / "common" / "map_layers" / "africa_coastline.geojson"
DATA_FILE = HERE / "data" / "monitoring_blind_spots_data.csv"
OUT_FILE = HERE / "outputs" / "figure5_monitoring_blind_spot_risk.png"


FONT_SIZE = {
    "title": 20,
    "axis_label": 18,
    "tick_label": 16,
    "legend": 18,
    "panel_label": 26,
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE["tick_label"],
        "axes.linewidth": 0.8,
        "axes.labelsize": FONT_SIZE["axis_label"],
        "axes.titlesize": FONT_SIZE["title"],
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": FONT_SIZE["legend"],
    }
)


def main():
    HERE.joinpath("outputs").mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(BOUNDARY_FILE)
    coast = gpd.read_file(COAST_FILE)
    df = pd.read_csv(DATA_FILE)

    merged = gdf.merge(df, on="admin2_idx", how="left")
    merged["dominance_class"] = pd.to_numeric(merged["dominance_class"], errors="coerce")
    merged["risk_class"] = pd.to_numeric(merged["risk_class"], errors="coerce")
    merged["total_mismatch_freq"] = pd.to_numeric(merged["total_mismatch_freq"], errors="coerce")
    merged["avg_production"] = pd.to_numeric(merged["avg_production"], errors="coerce")

    gdf_with_data = merged[merged["risk_class"].notna()]
    country_boundaries = gdf_with_data.dissolve(by="ADMIN0").boundary

    fig = plt.figure(figsize=(18, 5.5), facecolor="white")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=-0.40, left=0.02, right=0.98, top=0.95, bottom=0.22)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    driver_colors = {0: "#FF9800", 1: "#388E3C", 2: "#AB47BC", 3: "#8D6E63", 4: "#2196F3"}
    driver_labels = {0: "Energy", 1: "Root Water", 2: "Mixed", 3: "Other", 4: "Surface Water"}

    gdf_driver = merged[(merged["dominance_class"].notna()) & (merged["risk_class"].notna())]
    class_counts = gdf_driver["dominance_class"].value_counts().to_dict()
    for class_id in sorted(class_counts):
        subset = gdf_driver[gdf_driver["dominance_class"] == class_id]
        subset.plot(ax=ax_a, color=driver_colors[int(class_id)], edgecolor="#777777", linewidth=0.15, alpha=0.95)
    country_boundaries.plot(ax=ax_a, color="#555555", linewidth=0.35, alpha=0.6)
    coast.plot(ax=ax_a, color="none", edgecolor="#303030", linewidth=1.0, alpha=0.9, zorder=10)
    ax_a.set_xlim(-20, 55)
    ax_a.set_ylim(-40, 40)
    ax_a.set_aspect("equal")
    ax_a.set_axis_off()
    ax_a.text(-0.02, 1.00, "a", transform=ax_a.transAxes, fontsize=FONT_SIZE["panel_label"], fontweight="bold", va="top", ha="left")
    legend_order = [0, 1, 4, 2, 3]
    ax_a.legend(
        handles=[Patch(facecolor=driver_colors[class_id], edgecolor="#555555", linewidth=0.5, label=driver_labels[class_id]) for class_id in legend_order if class_id in class_counts],
        loc="lower center",
        fontsize=FONT_SIZE["legend"],
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#999999",
        framealpha=0.98,
        handlelength=1.8,
        handleheight=1.2,
        borderpad=0.4,
        labelspacing=0.35,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, -0.27),
    )

    bin_edges = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 1.0]
    bin_colors = ["#FFF3E0", "#FFE0B2", "#FFB74D", "#E65100", "#BF360C", "#7B1A0A"]
    bin_labels = ["0-2%", "2-5%", "5-10%", "10-15%", "15-20%", ">20%"]
    gdf_mismatch = merged[merged["total_mismatch_freq"].notna()].copy()
    gdf_mismatch["mismatch_bin"] = pd.cut(gdf_mismatch["total_mismatch_freq"], bins=bin_edges, labels=range(6), include_lowest=True, right=False)
    for bin_id in range(6):
        subset = gdf_mismatch[gdf_mismatch["mismatch_bin"] == bin_id]
        if len(subset):
            subset.plot(ax=ax_b, color=bin_colors[bin_id], edgecolor="#888888", linewidth=0.15, alpha=0.95)
    country_boundaries.plot(ax=ax_b, color="#555555", linewidth=0.35, alpha=0.6)
    coast.plot(ax=ax_b, color="none", edgecolor="#303030", linewidth=1.0, alpha=0.9, zorder=10)
    ax_b.set_xlim(-20, 55)
    ax_b.set_ylim(-40, 40)
    ax_b.set_aspect("equal")
    ax_b.set_axis_off()
    ax_b.text(-0.02, 1.00, "b", transform=ax_b.transAxes, fontsize=FONT_SIZE["panel_label"], fontweight="bold", va="top", ha="left")
    ax_b.legend(
        handles=[Patch(facecolor=bin_colors[i], edgecolor="#555555", linewidth=0.5, label=bin_labels[i]) for i in range(6) if (gdf_mismatch["mismatch_bin"] == i).sum() > 0],
        loc="lower center",
        fontsize=FONT_SIZE["legend"],
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#999999",
        framealpha=0.98,
        handlelength=1.5,
        handleheight=1.0,
        borderpad=0.3,
        labelspacing=0.25,
        columnspacing=0.6,
        title="Mismatch frequency",
        title_fontsize=FONT_SIZE["legend"],
        bbox_to_anchor=(0.5, -0.27),
    )

    risk_colors = {0: "#81C784", 1: "#FFB74D", 2: "#E57373"}
    gdf_risk = merged[merged["risk_class"].notna()]
    for risk_level in [0, 1, 2]:
        subset = gdf_risk[gdf_risk["risk_class"] == risk_level]
        if len(subset):
            subset.plot(ax=ax_c, color=risk_colors[risk_level], edgecolor="#888888", linewidth=0.15, alpha=0.9)
    country_boundaries.plot(ax=ax_c, color="#555555", linewidth=0.35, alpha=0.6)
    coast.plot(ax=ax_c, color="none", edgecolor="#303030", linewidth=1.0, alpha=0.9, zorder=10)
    ax_c.set_xlim(-20, 55)
    ax_c.set_ylim(-40, 40)
    ax_c.set_aspect("equal")
    ax_c.set_axis_off()
    ax_c.text(-0.02, 1.00, "c", transform=ax_c.transAxes, fontsize=FONT_SIZE["panel_label"], fontweight="bold", va="top", ha="left")

    gdf_prod = gdf_risk[gdf_risk["avg_production"].notna()]
    prod_total = gdf_prod["avg_production"].sum()
    prod_high = gdf_prod[gdf_prod["risk_class"] == 2]["avg_production"].sum()
    prod_mod = gdf_prod[gdf_prod["risk_class"] == 1]["avg_production"].sum()
    prod_low = gdf_prod[gdf_prod["risk_class"] == 0]["avg_production"].sum()

    ax_c.legend(
        handles=[
            Patch(facecolor=risk_colors[2], edgecolor="#555555", linewidth=0.5, label=f"High risk ({(prod_high / prod_total * 100):.1f}%)"),
            Patch(facecolor=risk_colors[1], edgecolor="#555555", linewidth=0.5, label=f"Moderate risk ({(prod_mod / prod_total * 100):.1f}%)"),
            Patch(facecolor=risk_colors[0], edgecolor="#555555", linewidth=0.5, label=f"Low risk ({(prod_low / prod_total * 100):.1f}%)"),
        ],
        loc="lower center",
        fontsize=FONT_SIZE["legend"],
        frameon=True,
        facecolor="white",
        edgecolor="#999999",
        framealpha=0.98,
        handlelength=1.8,
        handleheight=1.2,
        borderpad=0.4,
        labelspacing=0.35,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, -0.27),
    )

    fig.savefig(OUT_FILE, dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.05, edgecolor="none")
    plt.close(fig)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()

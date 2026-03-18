#!/usr/bin/env python3

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from recompute_panel_a_coefficients import compute_panel_a_coefficients, read_exposure_input


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_FILE = HERE / "outputs" / "figure2_yield_response_and_mismatch_maps.png"

BOUNDARY_FILE = ROOT / "common" / "map_layers" / "admin2_boundaries.geojson"
COAST_FILE = ROOT / "common" / "map_layers" / "africa_coastline.geojson"

REGRESSION_INPUT_FILE = HERE / "data" / "panel_a_regression_input.csv.gz"
CASE_B_FILE = HERE / "data" / "panel_b_case_chad_2010.csv"
CASE_C_FILE = HERE / "data" / "panel_c_case_malawi_2002.csv"
CASE_C_META = HERE / "data" / "panel_c_case_metadata.json"
MAP_FILE = HERE / "data" / "panel_de_map_frequencies.csv"


FONT_SIZE = {
    "axis_label": 17,
    "tick_label": 15,
    "legend": 18,
    "panel_label": 20,
    "annotation": 15,
    "stats": 14,
}

COLORS = {
    "L1": "#4A90D9",
    "L2L3": "#2E7D32",
    "L2L3_light": "#A5D6A7",
    "VPD": "#E65100",
    "annotation": "#424242",
    "ref_line": "#BDBDBD",
    "false_signal": "#C62828",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE["tick_label"],
        "axes.linewidth": 0.5,
        "axes.labelsize": FONT_SIZE["axis_label"],
        "axes.labelpad": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }
)


def build_map_frame():
    boundaries = gpd.read_file(BOUNDARY_FILE)[["admin2_idx", "ADMIN0", "geometry"]]
    freqs = pd.read_csv(MAP_FILE)
    coast = gpd.read_file(COAST_FILE)
    merged = boundaries.merge(freqs, on="admin2_idx", how="left")
    return merged, coast


def significance_mask(coef, se):
    with np.errstate(divide="ignore", invalid="ignore"):
        t_value = np.abs(coef / se)
    sig = t_value > 1.96
    if len(sig):
        sig[0] = False
    return sig


def main():
    HERE.joinpath("outputs").mkdir(parents=True, exist_ok=True)

    df_coef = compute_panel_a_coefficients(read_exposure_input(REGRESSION_INPUT_FILE))
    case_b = pd.read_csv(CASE_B_FILE)
    case_c = pd.read_csv(CASE_C_FILE)
    case_c_meta = json.load(open(CASE_C_META))
    gdf, coast = build_map_frame()

    fig = plt.figure(figsize=(12, 8.5), facecolor="white")

    top_y0, top_h = 0.62, 0.26
    ax_a = fig.add_axes([0.07, top_y0, 0.24, top_h])
    ax_b = fig.add_axes([0.365, top_y0, 0.22, top_h])
    ax_c = fig.add_axes([0.70, top_y0, 0.22, top_h])

    gs_bot = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1, 1],
        wspace=0.05,
        left=0.00,
        right=0.91,
        top=0.47,
        bottom=0.02,
    )
    ax_d = fig.add_subplot(gs_bot[0])
    ax_e = fig.add_subplot(gs_bot[1])

    x = np.arange(1, 11)
    ax_a.axhline(y=0, color=COLORS["ref_line"], linestyle="-", linewidth=0.5, zorder=0)

    pct_L1 = df_coef["L1_pct"].to_numpy()
    pct_se_L1 = df_coef["L1_pct_se"].to_numpy()
    pct_L2L3 = df_coef["L2L3_pct"].to_numpy()
    pct_se_L2L3 = df_coef["L2L3_pct_se"].to_numpy()
    pct_VPD = df_coef["VPD_pct"].to_numpy()
    pct_se_VPD = df_coef["VPD_pct_se"].to_numpy()

    coef_L1 = df_coef["L1_coef_logpoint"].to_numpy()
    se_L1 = df_coef["L1_se_logpoint"].to_numpy()
    coef_L2L3 = df_coef["L2L3_coef_logpoint"].to_numpy()
    se_L2L3 = df_coef["L2L3_se_logpoint"].to_numpy()
    coef_VPD = df_coef["VPD_coef_logpoint"].to_numpy()
    se_VPD = df_coef["VPD_se_logpoint"].to_numpy()

    sig_L1 = significance_mask(coef_L1, se_L1)
    sig_L2L3 = significance_mask(coef_L2L3, se_L2L3)
    sig_VPD = significance_mask(coef_VPD, se_VPD)

    ax_a.fill_between(x, pct_L1 - 1.96 * pct_se_L1, pct_L1 + 1.96 * pct_se_L1, color=COLORS["L1"], alpha=0.06)
    ax_a.plot(x, pct_L1, "--", color=COLORS["L1"], linewidth=1.5, zorder=3)
    ax_a.scatter(x[sig_L1], pct_L1[sig_L1], s=36, marker="o", facecolor=COLORS["L1"], edgecolor=COLORS["L1"], zorder=5)
    ax_a.scatter(x[~sig_L1], pct_L1[~sig_L1], s=36, marker="o", facecolor="white", edgecolor=COLORS["L1"], linewidth=1.2, zorder=5)

    ax_a.fill_between(
        x,
        pct_L2L3 - 1.96 * pct_se_L2L3,
        pct_L2L3 + 1.96 * pct_se_L2L3,
        color=COLORS["L2L3_light"],
        alpha=0.2,
    )
    ax_a.plot(x, pct_L2L3, "-", color=COLORS["L2L3"], linewidth=2.0, zorder=4)
    ax_a.scatter(x[sig_L2L3], pct_L2L3[sig_L2L3], s=49, marker="s", facecolor=COLORS["L2L3"], edgecolor=COLORS["L2L3"], zorder=6)
    ax_a.scatter(x[~sig_L2L3], pct_L2L3[~sig_L2L3], s=49, marker="s", facecolor="white", edgecolor=COLORS["L2L3"], linewidth=1.2, zorder=6)

    ax_a.fill_between(x, pct_VPD - 1.96 * pct_se_VPD, pct_VPD + 1.96 * pct_se_VPD, color=COLORS["VPD"], alpha=0.06)
    ax_a.plot(x, pct_VPD, ":", color=COLORS["VPD"], linewidth=1.5, zorder=2)
    ax_a.scatter(x[sig_VPD], pct_VPD[sig_VPD], s=36, marker="^", facecolor=COLORS["VPD"], edgecolor=COLORS["VPD"], zorder=5)
    ax_a.scatter(x[~sig_VPD], pct_VPD[~sig_VPD], s=36, marker="^", facecolor="white", edgecolor=COLORS["VPD"], linewidth=1.2, zorder=5)

    ax_a.set_xlabel("Exposure decile", fontsize=FONT_SIZE["axis_label"])
    ax_a.set_ylabel("Yield response (%)", fontsize=FONT_SIZE["axis_label"])
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"D{i}" for i in range(1, 11)], rotation=45, ha="center", fontsize=FONT_SIZE["tick_label"])
    ax_a.set_xlim(0.3, 10.7)

    all_upper = np.concatenate([pct_L1 + 1.96 * pct_se_L1, pct_L2L3 + 1.96 * pct_se_L2L3, pct_VPD + 1.96 * pct_se_VPD])
    all_lower = np.concatenate([pct_L1 - 1.96 * pct_se_L1, pct_L2L3 - 1.96 * pct_se_L2L3, pct_VPD - 1.96 * pct_se_VPD])
    ax_a.set_ylim(np.nanmin(all_lower) - 0.3, np.nanmax(all_upper) + 0.3)

    ax_b_vpd = ax_b.twinx()
    ax_b_vpd.spines["right"].set_visible(True)
    ax_b_vpd.spines["top"].set_visible(False)
    ax_b_vpd.plot(case_b["day_of_season"], case_b["vpd"], "--", color=COLORS["VPD"], linewidth=0.9, alpha=0.5, zorder=1)
    ax_b_vpd.set_ylabel("VPD (hPa)", fontsize=FONT_SIZE["axis_label"], color=COLORS["VPD"])
    ax_b_vpd.tick_params(axis="y", labelcolor=COLORS["VPD"], labelsize=FONT_SIZE["tick_label"])
    ax_b_vpd.spines["right"].set_color(COLORS["VPD"])
    ax_b_vpd.spines["right"].set_linewidth(0.5)

    ax_b.plot(case_b["day_of_season"], case_b["sm_l1"], "-", color=COLORS["L1"], linewidth=1.3, alpha=0.85, zorder=3)
    ax_b.plot(case_b["day_of_season"], case_b["sm_l2l3"], "-", color=COLORS["L2L3"], linewidth=1.5, zorder=4)

    gap_b = (case_b["sm_l1"] - case_b["sm_l2l3"]).to_numpy()
    spike_idx = int(np.argmax(gap_b[20:])) + 20
    spike_day = float(case_b.iloc[spike_idx]["day_of_season"])
    spike_l1 = float(case_b.iloc[spike_idx]["sm_l1"])
    spike_l2l3 = float(case_b.iloc[spike_idx]["sm_l2l3"])
    y_max_b = max(case_b["sm_l1"].max(), case_b["sm_l2l3"].max()) + 0.14
    y_min_b = min(case_b["sm_l1"].min(), case_b["sm_l2l3"].min()) - 0.02

    ax_b.annotate(
        "Surface wets\n(Ineffective wetness)",
        xy=(spike_day, spike_l1 + 0.01),
        xytext=(25, y_max_b - 0.07),
        fontsize=FONT_SIZE["annotation"],
        fontweight="semibold",
        style="italic",
        color=COLORS["false_signal"],
        ha="left",
        arrowprops=dict(arrowstyle="->", color=COLORS["false_signal"], lw=0.7, shrinkB=2),
    )
    ax_b.annotate(
        "Root zone\nremains dry",
        xy=(spike_day, spike_l2l3),
        xytext=(30, 0.24),
        fontsize=FONT_SIZE["annotation"],
        fontweight="semibold",
        color=COLORS["annotation"],
        ha="left",
        arrowprops=dict(arrowstyle="->", color=COLORS["annotation"], lw=0.7, shrinkB=2),
    )

    ax_b.set_xlim(0, 120)
    ax_b.set_ylim(y_min_b, y_max_b)
    vpd_range_b = case_b["vpd"].max() - case_b["vpd"].min()
    ax_b_vpd.set_ylim(case_b["vpd"].min() - 0.15 * vpd_range_b, case_b["vpd"].max() + 0.15 * vpd_range_b)
    ax_b.set_xlabel("Day of growing season", fontsize=FONT_SIZE["axis_label"])
    ax_b.set_ylabel(r"Soil moisture (m$^3$/m$^3$)", fontsize=FONT_SIZE["axis_label"])
    ax_b.text(
        0.97,
        0.03,
        "Example: Chad, 2010",
        transform=ax_b.transAxes,
        fontsize=FONT_SIZE["stats"],
        ha="right",
        va="bottom",
        color="#444444",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
    )

    ax_c_vpd = ax_c.twinx()
    ax_c_vpd.spines["right"].set_visible(True)
    ax_c_vpd.spines["top"].set_visible(False)
    ax_c_vpd.plot(case_c["day_of_season"], case_c["vpd"], "--", color=COLORS["VPD"], linewidth=0.9, alpha=0.5, zorder=1)
    ax_c_vpd.set_ylabel("VPD (hPa)", fontsize=FONT_SIZE["axis_label"], color=COLORS["VPD"])
    ax_c_vpd.tick_params(axis="y", labelcolor=COLORS["VPD"], labelsize=FONT_SIZE["tick_label"])
    ax_c_vpd.spines["right"].set_color(COLORS["VPD"])
    ax_c_vpd.spines["right"].set_linewidth(0.5)

    ax_c.plot(case_c["day_of_season"], case_c["sm_l1"], "-", color=COLORS["L1"], linewidth=1.3, alpha=0.85, zorder=3)
    ax_c.plot(case_c["day_of_season"], case_c["sm_l2l3"], "-", color=COLORS["L2L3"], linewidth=1.5, zorder=4)

    y_max_c = max(case_c["sm_l1"].max(), case_c["sm_l2l3"].max()) + 0.14
    y_min_c = min(case_c["sm_l1"].min(), case_c["sm_l2l3"].min()) - 0.02
    visible = case_c[(case_c["day_of_season"] >= 100) & (case_c["day_of_season"] <= 150)].copy()
    gap_visible = (visible["sm_l2l3"] - visible["sm_l1"]).to_numpy()
    annot_idx = int(np.argmax(gap_visible))
    annot_day = float(visible.iloc[annot_idx]["day_of_season"])
    annot_l1 = float(visible.iloc[annot_idx]["sm_l1"])
    annot_l2l3 = float(visible.iloc[annot_idx]["sm_l2l3"])

    ax_c.annotate(
        "Root zone remains wet\n(Concealed wetness)",
        xy=(annot_day, annot_l2l3),
        xytext=(annot_day + 2, annot_l2l3 + 0.10),
        fontsize=FONT_SIZE["annotation"],
        fontweight="semibold",
        style="italic",
        color=COLORS["false_signal"],
        ha="center",
        arrowprops=dict(arrowstyle="->", color=COLORS["false_signal"], lw=0.7, shrinkB=2),
    )
    ax_c.annotate(
        "Surface dries",
        xy=(annot_day, annot_l1),
        xytext=(annot_day + 10, annot_l1 - 0.06),
        fontsize=FONT_SIZE["annotation"],
        fontweight="semibold",
        color=COLORS["annotation"],
        ha="center",
        arrowprops=dict(arrowstyle="->", color=COLORS["annotation"], lw=0.7, shrinkB=2),
    )

    ax_c.set_xlim(100, 150)
    ax_c.set_ylim(y_min_c, y_max_c)
    vpd_range_c = case_c["vpd"].max() - case_c["vpd"].min()
    ax_c_vpd.set_ylim(case_c["vpd"].min() - 0.15 * vpd_range_c, case_c["vpd"].max() + 0.15 * vpd_range_c)
    ax_c.set_xlabel("Day of growing season", fontsize=FONT_SIZE["axis_label"])
    ax_c.set_ylabel(r"Soil moisture (m$^3$/m$^3$)", fontsize=FONT_SIZE["axis_label"])
    ax_c.text(
        0.97,
        0.03,
        f"Example: {case_c_meta['country']}, {case_c_meta['year']}",
        transform=ax_c.transAxes,
        fontsize=FONT_SIZE["stats"],
        ha="right",
        va="bottom",
        color="#444444",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
    )

    colors_d = ["#FFFDE7", "#FFF9C4", "#FFF176", "#FFEE58", "#FFD54F", "#FFB300", "#FF8F00", "#F57C00", "#E65100", "#D84315", "#BF360C", "#8D1900", "#5D0000"]
    cmap_d = LinearSegmentedColormap.from_list("ineffective", colors_d, N=256)
    gdf_d = gdf[gdf["ineffective_freq"].notna()]
    gdf_d.plot(ax=ax_d, column="ineffective_freq", cmap=cmap_d, vmin=0.0, vmax=0.15, edgecolor="#888888", linewidth=0.05, legend=False)
    coast.plot(ax=ax_d, color="none", edgecolor="#404040", linewidth=0.6, alpha=0.8, zorder=10)
    ax_d.set_xlim(-20, 55)
    ax_d.set_ylim(-40, 40)
    ax_d.set_aspect("equal")
    ax_d.set_axis_off()

    cax_d = inset_axes(ax_d, width="3%", height="60%", loc="center right", bbox_to_anchor=(0.08, 0, 1, 1), bbox_transform=ax_d.transAxes, borderpad=0)
    sm_cb_d = plt.cm.ScalarMappable(cmap=cmap_d, norm=plt.Normalize(vmin=0.0, vmax=0.15))
    cbar_d = plt.colorbar(sm_cb_d, cax=cax_d, orientation="vertical", extend="max")
    cbar_d.set_label("Frequency of ineffective\nsurface wetness", fontsize=FONT_SIZE["axis_label"], labelpad=5)
    cbar_d.ax.tick_params(labelsize=FONT_SIZE["tick_label"], length=2, width=0.5, pad=1)
    cbar_d.set_ticks([0.0, 0.05, 0.10, 0.15])
    cbar_d.set_ticklabels(["0%", "5%", "10%", "15%"])
    cbar_d.outline.set_linewidth(0.5)

    colors_e = ["#E3F2FD", "#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5", "#2196F3", "#1E88E5", "#1976D2", "#1565C0", "#0D47A1", "#0A3D91", "#072B6F", "#051D4D"]
    cmap_e = LinearSegmentedColormap.from_list("concealed", colors_e, N=256)
    gdf_e = gdf[gdf["opposite_freq"].notna()]
    gdf_e.plot(ax=ax_e, column="opposite_freq", cmap=cmap_e, vmin=0.0, vmax=0.15, edgecolor="#888888", linewidth=0.05, legend=False)
    coast.plot(ax=ax_e, color="none", edgecolor="#404040", linewidth=0.6, alpha=0.8, zorder=10)
    ax_e.set_xlim(-20, 55)
    ax_e.set_ylim(-40, 40)
    ax_e.set_aspect("equal")
    ax_e.set_axis_off()

    cax_e = inset_axes(ax_e, width="3%", height="60%", loc="center right", bbox_to_anchor=(0.08, 0, 1, 1), bbox_transform=ax_e.transAxes, borderpad=0)
    sm_cb_e = plt.cm.ScalarMappable(cmap=cmap_e, norm=plt.Normalize(vmin=0.0, vmax=0.15))
    cbar_e = plt.colorbar(sm_cb_e, cax=cax_e, orientation="vertical", extend="max")
    cbar_e.set_label("Frequency of concealed\nroot wetness", fontsize=FONT_SIZE["axis_label"], labelpad=5)
    cbar_e.ax.tick_params(labelsize=FONT_SIZE["tick_label"], length=2, width=0.5, pad=1)
    cbar_e.set_ticks([0.0, 0.05, 0.10, 0.15])
    cbar_e.set_ticklabels(["0%", "5%", "10%", "15%"])
    cbar_e.outline.set_linewidth(0.5)

    for axis in fig.get_axes():
        pos = axis.get_position()
        if pos.y0 > 0.55 and abs(pos.height - top_h) > 1e-5:
            axis.set_position([pos.x0, top_y0, pos.width, top_h])

    shared_handles = [
        Line2D([0], [0], color=COLORS["L1"], linestyle="--", linewidth=1.5, marker="o", markersize=6, markerfacecolor=COLORS["L1"], label="Surface (L1)"),
        Line2D([0], [0], color=COLORS["L2L3"], linestyle="-", linewidth=2.0, marker="s", markersize=6, markerfacecolor=COLORS["L2L3"], label="Root zone (L2L3)"),
        Line2D([0], [0], color=COLORS["VPD"], linestyle=":", linewidth=1.5, marker="^", markersize=6, markerfacecolor=COLORS["VPD"], label="VPD"),
    ]
    fig.legend(
        handles=shared_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.99),
        frameon=False,
        fontsize=FONT_SIZE["legend"],
        ncol=3,
        handlelength=2.5,
        columnspacing=2.0,
    )

    label_props = dict(fontsize=FONT_SIZE["panel_label"], fontweight="bold", va="top", ha="left")
    fig.text(ax_a.get_position().x0 - 0.02, ax_a.get_position().y1 + 0.04, "a", **label_props)
    fig.text(ax_b.get_position().x0 - 0.02, ax_b.get_position().y1 + 0.04, "b", **label_props)
    fig.text(ax_c.get_position().x0 - 0.02, ax_c.get_position().y1 + 0.04, "c", **label_props)
    fig.text(ax_d.get_position().x0 - 0.01, ax_d.get_position().y1 + 0.02, "d", **label_props)
    fig.text(ax_e.get_position().x0 - 0.01, ax_e.get_position().y1 + 0.02, "e", **label_props)

    fig.savefig(OUT_FILE, dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()

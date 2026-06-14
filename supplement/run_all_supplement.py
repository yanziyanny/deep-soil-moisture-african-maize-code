#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
import numpy as np
import pandas as pd


sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "supplement" / "data"
COAST_FILE = REPO_ROOT / "common" / "map_layers" / "africa_coastline.geojson"
OUTPUT_DIR = REPO_ROOT / "supplement" / "outputs"

KOPPEN_SHORT = {1: "Tropical", 2: "Savanna", 3: "Desert", 4: "Semi-arid", 5: "Temperate"}
GROUP_ORDER = ["RootWater", "Energy", "SurfaceWater"]
GROUP_LABELS = {"RootWater": "Root", "Energy": "Energy", "SurfaceWater": "Surface"}
COLOR_MAP = {"RootWater": "#1B7837", "Energy": "#D95F02", "SurfaceWater": "#92C5DE"}
FEATURE_GROUP = {
    "VPDa_8mean": "Energy",
    "SWa_8mean": "Energy",
    "Tmaxa_8mean": "Energy",
    "SMa_L1_8mean": "SurfaceWater",
    "PPTa_8sum": "SurfaceWater",
    "SMa_L2_8mean": "RootWater",
    "SMa_L3_8mean": "RootWater",
}
VAR_LABELS = {
    "VPDa_8mean": "VPD",
    "SWa_8mean": "SW",
    "Tmaxa_8mean": "Tmax",
    "SMa_L1_8mean": "SM L1",
    "PPTa_8sum": "PPT",
    "SMa_L2_8mean": "SM L2",
    "SMa_L3_8mean": "SM L3",
}


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def validate_s9_source() -> None:
    source = DATA_DIR / "shapley_r2" / "s9_shapley_group_decomposition.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing S9 source data: {source}")
    df = pd.read_csv(source)
    sums = df.groupby("koppen_id", as_index=False).agg(
        full_model_r2=("full_model_r2", "first"),
        shapley_sum=("shapley_r2", "sum"),
    )
    gap = (sums["full_model_r2"] - sums["shapley_sum"]).abs().max()
    if gap > 1e-9:
        raise ValueError(f"S9 Shapley R2 values do not sum to full-model R2; max gap={gap}")


def validate_s10_source() -> None:
    source = DATA_DIR / "hard_energy_filtering" / "s10_hard_energy_filtering_sensitivity.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing S10 source data: {source}")
    df = pd.read_csv(source)
    if set(df["sw_threshold"]) != {21.6}:
        raise ValueError("S10 source data must use SW_8mean_raw >= 21.6 MJ m-2 day-1")
    if set(df["tmax_threshold"]) != {18.0}:
        raise ValueError("S10 source data must use Tmax_8mean_raw >= 18 C")
    vpd_thresholds = [column for column in df.columns if "vpd" in column.lower() and "threshold" in column.lower()]
    if vpd_thresholds:
        raise ValueError(f"S10 source data should not contain a VPD threshold: {vpd_thresholds}")


def sample_values(arr: np.ndarray, rng: np.random.Generator, n_sample: int = 30000) -> np.ndarray:
    values = arr[np.isfinite(arr)].ravel()
    if len(values) <= n_sample:
        return values
    return rng.choice(values, size=n_sample, replace=False)


def write_correlation_maps(
    output_dir: Path,
    arrays: list[tuple[Path, str]],
    output_stem: str,
    figure_width: float,
    flip_vertical: bool = False,
) -> list[Path]:
    loaded_arrays = []
    for path, _ in arrays:
        if not path.exists():
            raise FileNotFoundError(f"Missing coupling source array: {path}")
        arr = np.load(path).astype(float)
        arr[~np.isfinite(arr)] = np.nan
        np.clip(arr, -1, 1, out=arr)
        if flip_vertical:
            arr = np.flipud(arr)
        loaded_arrays.append(arr)

    coast = gpd.read_file(COAST_FILE)
    rng = np.random.default_rng(42)
    violin_data = [sample_values(arr, rng) for arr in loaded_arrays]
    medians = [float(np.nanmedian(arr)) for arr in loaded_arrays]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.6,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    n_panels = len(arrays)
    fig = plt.figure(figsize=(figure_width / 25.4, 86 / 25.4), facecolor="white")
    gs = fig.add_gridspec(
        1,
        n_panels + 1,
        width_ratios=[1] * n_panels + [0.045],
        left=0.06,
        right=0.96,
        top=0.88,
        bottom=0.14,
        wspace=0.10,
    )
    cmap = plt.cm.Blues_r
    vmin, vmax = -0.8, 0.0
    labels = "abc"
    mesh = None

    for idx, ((_, title), arr, vals, med) in enumerate(zip(arrays, loaded_arrays, violin_data, medians)):
        ax = fig.add_subplot(gs[0, idx])
        n_lat, n_lon = arr.shape
        lon = np.linspace(-20, 55, n_lon)
        lat = np.linspace(40, -40, n_lat)
        mesh = ax.pcolormesh(lon, lat, arr, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
        coast.plot(ax=ax, color="none", edgecolor="#333333", linewidth=0.35, zorder=3)
        ax.set_xlim(-20, 55)
        ax.set_ylim(-40, 40)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=11, pad=5, style="italic")
        ax.text(-0.08, 1.10, labels[idx], transform=ax.transAxes, fontsize=13, fontweight="bold")
        ax.set_xticks([0, 30])
        ax.set_xticklabels([r"0$^\circ$", r"30$^\circ$E"])
        ax.set_yticks([-30, 0, 30])
        if idx == 0:
            ax.set_yticklabels([r"30$^\circ$S", r"0$^\circ$", r"30$^\circ$N"])
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=2.5, width=0.5, colors="#444444")

        inset = ax.inset_axes([0.16, 0.08, 0.16, 0.42])
        inset.patch.set_facecolor("white")
        inset.patch.set_alpha(0.90)
        vp = inset.violinplot([vals], positions=[0], widths=0.8, showextrema=False, showmedians=False)
        for body in vp["bodies"]:
            body.set_facecolor("#3182bd")
            body.set_edgecolor("white")
            body.set_alpha(0.9)
            body.set_linewidth(1.0)
        inset.hlines(med, -0.30, 0.30, colors="white", linewidth=2.4, zorder=4)
        inset.hlines(med, -0.30, 0.30, colors="#222222", linewidth=1.4, zorder=5)
        inset.axhline(0, color="#666666", linestyle="--", linewidth=0.6, zorder=2)
        inset.text(0.45, med, f"{med:.2f}", fontsize=8, fontweight="bold", va="center", ha="left", color="#222222")
        inset.set_xlim(-0.6, 0.9)
        inset.set_ylim(-0.95, 0.15)
        inset.set_yticks([-0.8, -0.4, 0.0])
        inset.set_xticks([])
        inset.tick_params(axis="y", labelsize=7, length=2, width=0.5, pad=2)
        inset.spines["top"].set_visible(False)
        inset.spines["right"].set_visible(False)
        inset.spines["bottom"].set_visible(False)
        inset.spines["left"].set_linewidth(0.5)

    cax = fig.add_subplot(gs[0, n_panels])
    cbar = fig.colorbar(mesh, cax=cax, orientation="vertical", extend="min")
    cbar.set_ticks([0.0, -0.2, -0.4, -0.6, -0.8])
    cbar.set_ticklabels(["0", "-0.2", "-0.4", "-0.6", "-0.8"])
    cbar.set_label("Correlation (r)", fontsize=10, labelpad=5)
    cbar.ax.tick_params(labelsize=8, width=0.6, length=3)
    cbar.outline.set_linewidth(0.6)

    paths = [output_dir / f"{output_stem}.png", output_dir / f"{output_stem}.pdf"]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)
    return paths


def write_s3(output_dir: Path) -> list[Path]:
    return write_correlation_maps(
        output_dir,
        [
            (DATA_DIR / "coupling" / "gleam_surface_sm_vpd_r.npy", "Surface"),
            (DATA_DIR / "coupling" / "gleam_rootzone_sm_vpd_r.npy", "Root zone"),
        ],
        "supplementary_figure_s3_gleam_sm_vpd_correlation",
        180,
    )


def write_s4(output_dir: Path) -> list[Path]:
    return write_correlation_maps(
        output_dir,
        [
            (DATA_DIR / "coupling" / "gldas_l1_sm_vpd_r.npy", "L1: 0-10 cm"),
            (DATA_DIR / "coupling" / "gldas_l2_sm_vpd_r.npy", "L2: 10-40 cm"),
            (DATA_DIR / "coupling" / "gldas_l3_sm_vpd_r.npy", "L3: 40-100 cm"),
        ],
        "supplementary_figure_s4_gldas_sm_vpd_correlation",
        210,
        flip_vertical=True,
    )


def write_s5(output_dir: Path) -> list[Path]:
    source = DATA_DIR / "gldas_yield" / "s5_gldas_exposure_coefficients.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing S5 coefficient source data: {source}")
    df = pd.read_csv(source)
    required = {
        "decile",
        "L1_pct",
        "L1_pct_se",
        "L2L3_pct",
        "L2L3_pct_se",
        "VPD_pct",
        "VPD_pct_se",
        "L1_coef_logpoint",
        "L1_se_logpoint",
        "L2L3_coef_logpoint",
        "L2L3_se_logpoint",
        "VPD_coef_logpoint",
        "VPD_se_logpoint",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"S5 coefficient source missing columns: {sorted(missing)}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.linewidth": 0.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {"L1": "#4A90D9", "L2L3": "#2E7D32", "L2L3_light": "#A5D6A7", "VPD": "#E65100", "ref": "#BDBDBD"}
    fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor="white")
    x = df["decile"].to_numpy()
    ax.axhline(y=0, color=colors["ref"], linestyle="-", linewidth=0.5, zorder=0)

    series = [
        ("L1", "Surface (L1)", "L1_pct", "L1_pct_se", "L1_coef_logpoint", "L1_se_logpoint", "--", "o", 36, 1.5, 1),
        ("L2L3", "Root zone (L2L3)", "L2L3_pct", "L2L3_pct_se", "L2L3_coef_logpoint", "L2L3_se_logpoint", "-", "s", 49, 2.0, 5),
        ("VPD", "VPD", "VPD_pct", "VPD_pct_se", "VPD_coef_logpoint", "VPD_se_logpoint", ":", "^", 36, 1.5, 2),
    ]
    for key, label, value_col, se_col, coef_col, coef_se_col, linestyle, marker, size, linewidth, zorder in series:
        values = df[value_col].to_numpy()
        errors = df[se_col].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            sig = np.abs(df[coef_col].to_numpy() / df[coef_se_col].to_numpy()) > 1.96
        sig[0] = False
        fill_color = colors["L2L3_light"] if key == "L2L3" else colors[key]
        fill_alpha = 0.20 if key == "L2L3" else 0.06
        ax.fill_between(x, values - 1.96 * errors, values + 1.96 * errors, color=fill_color, alpha=fill_alpha, zorder=max(1, zorder - 1), edgecolor="none")
        ax.plot(x, values, linestyle, color=colors[key], linewidth=linewidth, label=label, zorder=zorder)
        ax.scatter(x[sig], values[sig], s=size, marker=marker, facecolor=colors[key], edgecolor=colors[key], linewidth=0, zorder=7)
        ax.scatter(x[~sig], values[~sig], s=size, marker=marker, facecolor="white", edgecolor=colors[key], linewidth=1.2, zorder=7)

    ax.set_xlabel("Exposure decile", fontsize=11)
    ax.set_ylabel("Yield response (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i}" for i in x], fontsize=8)
    ax.set_xlim(0.3, 10.7)
    upper = np.concatenate([df["L1_pct"] + 1.96 * df["L1_pct_se"], df["L2L3_pct"] + 1.96 * df["L2L3_pct_se"], df["VPD_pct"] + 1.96 * df["VPD_pct_se"]])
    lower = np.concatenate([df["L1_pct"] - 1.96 * df["L1_pct_se"], df["L2L3_pct"] - 1.96 * df["L2L3_pct_se"], df["VPD_pct"] - 1.96 * df["VPD_pct_se"]])
    ax.set_ylim(float(np.nanmin(lower) - 0.3), float(np.nanmax(upper) + 0.3))
    ax.legend(loc="upper center", fontsize=9, ncol=3, bbox_to_anchor=(0.5, 1.15), handlelength=1.5, columnspacing=0.6)

    paths = [
        output_dir / "supplementary_figure_s5_gldas_yield_response.png",
        output_dir / "supplementary_figure_s5_gldas_yield_response.pdf",
    ]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    return paths


def add_colored_xtick(ax: plt.Axes, x: float, parts: tuple[tuple[str, str, bool], ...]) -> None:
    children = [
        TextArea(
            text,
            textprops={
                "color": color,
                "fontsize": 10,
                "fontweight": "bold" if bold else "normal",
                "fontfamily": "Arial",
            },
        )
        for text, color, bold in parts
    ]
    packed = HPacker(children=children, pad=0, sep=0, align="baseline")
    box = AnnotationBbox(
        packed,
        (x, 0),
        xycoords=("data", "axes fraction"),
        box_alignment=(0.5, 1.0),
        frameon=False,
        xybox=(0, -3),
        boxcoords="offset points",
    )
    ax.add_artist(box)


def write_s6(output_dir: Path) -> list[Path]:
    source = DATA_DIR / "gldas_yield" / "s6_gldas_yield_sensitivity_coefficients.csv"
    r2_source = DATA_DIR / "gldas_yield" / "s6_gldas_r2_results.json"
    if not source.exists():
        raise FileNotFoundError(f"Missing S6 coefficient source data: {source}")
    if not r2_source.exists():
        raise FileNotFoundError(f"Missing S6 R2 source data: {r2_source}")
    df = pd.read_csv(source)
    for column in ["panel", "model", "term", "stage", "variable", "estimate", "se", "stars", "significant"]:
        if column not in df.columns:
            raise ValueError(f"S6 coefficient source missing column: {column}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.linewidth": 0.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {"vpd": "#2166AC", "L1": "#5DADE2", "L2L3": "#006400", "gray": "#4A4A4A", "light_gray": "#BFBFBF", "annotation": "#333333"}
    fig = plt.figure(figsize=(7.0, 5.8), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], width_ratios=[1, 1], wspace=0.22, hspace=0.35, left=0.08, right=0.97, top=0.93, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_a = df[df["panel"] == "a"].reset_index(drop=True)
    x_pos = np.arange(len(panel_a))
    ax_a.bar(x_pos, panel_a["estimate"], width=0.45, color=colors["vpd"], alpha=0.85, edgecolor="white", linewidth=0.5)
    ax_a.errorbar(x_pos, panel_a["estimate"], yerr=panel_a["se"], fmt="none", ecolor=colors["gray"], elinewidth=0.8, capsize=2, capthick=0.8)
    for i, row in panel_a.iterrows():
        if isinstance(row["stars"], str) and row["stars"]:
            ax_a.text(i, row["estimate"] + row["se"] - 1.5, row["stars"], ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels([])
    for xp, parts in zip(
        x_pos,
        [
            (("VPD", "#B22222", True), ("+L1", "black", False)),
            (("VPD", "#B22222", True), ("+L2L3", "black", False)),
            (("VPD", "#B22222", True), ("+L1+L2L3", "black", False)),
        ],
    ):
        add_colored_xtick(ax_a, xp, parts)
    ax_a.set_xlabel("Model specification", fontsize=11, labelpad=10)
    ax_a.set_ylabel("Yield Sensitivity\n(% per kPa VPD)", fontsize=11)
    y_max_a = float((panel_a["estimate"] + panel_a["se"]).max() + 5)
    ax_a.set_ylim(0, y_max_a)
    ax_a.set_xlim(-0.5, 2.5)
    ax_a.axhline(y=0, color=colors["light_gray"], linestyle="-", linewidth=0.5, zorder=0)
    ax_a.axvline(x=1.5, color="black", linestyle="--", linewidth=0.8)
    ax_a.annotate("", xy=(2, panel_a.loc[2, "estimate"] + panel_a.loc[2, "se"] + 0.8), xytext=(0, panel_a.loc[0, "estimate"] + panel_a.loc[0, "se"] + 0.8), arrowprops=dict(arrowstyle="->", color=colors["annotation"], lw=1.0, connectionstyle="arc3,rad=-0.15"))
    ax_a.text(0.7, panel_a.loc[0, "estimate"] + panel_a.loc[0, "se"] + 1.5, f"{panel_a.loc[0, 'estimate']:.1f}% -> {panel_a.loc[2, 'estimate']:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=colors["annotation"], bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.9))
    ax_a.text(0.02, 1.02, "a", transform=ax_a.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

    ax_b = fig.add_subplot(gs[0, 1])
    panel_b = df[df["panel"] == "b"].reset_index(drop=True)
    x_pos_b = [0, 1, 2 - 0.25, 2 + 0.25]
    for idx, row in panel_b.iterrows():
        color = colors[row["variable"]]
        alpha = 0.85 if bool(row["significant"]) else 0.25
        hatch = None if bool(row["significant"]) else "///"
        ax_b.bar(x_pos_b[idx], row["estimate"], width=0.45, color=color, alpha=alpha, edgecolor="white" if bool(row["significant"]) else color, linewidth=0.5 if bool(row["significant"]) else 1.0, hatch=hatch)
    ax_b.errorbar(x_pos_b, panel_b["estimate"], yerr=panel_b["se"], fmt="none", ecolor=colors["gray"], elinewidth=0.8, capsize=2, capthick=0.8)
    for idx, row in panel_b.iterrows():
        if isinstance(row["stars"], str) and row["stars"]:
            y = row["estimate"] + row["se"] - 1 if row["estimate"] >= 0 else row["estimate"] - row["se"] + 1
            va = "bottom" if row["estimate"] >= 0 else "top"
            ax_b.text(x_pos_b[idx], y, row["stars"], ha="center", va=va, fontsize=9, fontweight="bold")
    for idx, label in [(2, "L1"), (3, "L2L3")]:
        row = panel_b.loc[idx]
        ax_b.text(x_pos_b[idx], max(row["estimate"] + row["se"], 0) + 4.0, label, ha="center", va="bottom", fontsize=10, color=colors[row["variable"]], fontweight="bold")
    ax_b.set_xticks([0, 1, 2])
    ax_b.set_xticklabels([])
    for xp, parts in zip(
        [0, 1, 2],
        [
            (("VPD+", "black", False), ("L1", "#B22222", True)),
            (("VPD+", "black", False), ("L2L3", "#B22222", True)),
            (("VPD+", "black", False), ("L1+L2L3", "#B22222", True)),
        ],
    ):
        add_colored_xtick(ax_b, xp, parts)
    ax_b.set_xlabel("Model specification", fontsize=11, labelpad=10)
    ax_b.set_ylabel("Yield Sensitivity\n(% per 0.1 m$^3$/m$^3$ SM)", fontsize=11)
    ax_b.set_ylim(float((panel_b["estimate"] - panel_b["se"]).min() - 3), float((panel_b["estimate"] + panel_b["se"]).max() + 8))
    ax_b.set_xlim(-0.5, 2.7)
    ax_b.axhline(y=0, color=colors["light_gray"], linestyle="-", linewidth=0.5, zorder=0)
    ax_b.axvline(x=1.5, color="black", linestyle="--", linewidth=0.8)
    sig_legend = [
        mpatches.Patch(facecolor=colors["gray"], alpha=0.85, edgecolor="none", label="P < 0.05"),
        mpatches.Patch(facecolor=colors["gray"], alpha=0.25, hatch="///", edgecolor=colors["gray"], label="n.s."),
    ]
    ax_b.legend(handles=sig_legend, loc="lower left", fontsize=8, handlelength=1.0, frameon=False, handletextpad=0.4)
    ax_b.text(0.02, 1.02, "b", transform=ax_b.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

    ax_c = fig.add_subplot(gs[1, :])
    stages = ["Vegetative", "Reproductive", "Maturation"]
    x_c = np.arange(len(stages))
    w_c = 0.28
    panel_c = df[df["panel"] == "c"]
    for i, stage in enumerate(stages):
        for variable, offset in [("L1", -w_c / 2), ("L2L3", w_c / 2)]:
            row = panel_c[(panel_c["stage"] == stage) & (panel_c["variable"] == variable)].iloc[0]
            alpha = 0.85 if bool(row["significant"]) else 0.25
            hatch = None if bool(row["significant"]) else "///"
            ax_c.bar(i + offset, row["estimate"], w_c, color=colors[variable], alpha=alpha, edgecolor="white" if bool(row["significant"]) else colors[variable], linewidth=0.5 if bool(row["significant"]) else 1.0, hatch=hatch)
            ax_c.errorbar(i + offset, row["estimate"], yerr=1.96 * row["se"], fmt="none", ecolor=colors["gray"], elinewidth=0.8, capsize=2, capthick=0.8, zorder=5)
            if isinstance(row["stars"], str) and row["stars"]:
                if row["estimate"] > 0:
                    ax_c.text(i + offset, row["estimate"] + 1.96 * row["se"] - 2.5, row["stars"], ha="center", va="bottom", fontsize=9, fontweight="bold")
                else:
                    ax_c.text(i + offset, row["estimate"] - 1.96 * row["se"] - 4, row["stars"], ha="center", va="top", fontsize=9, fontweight="bold")
    ax_c.axhline(y=0, color=colors["light_gray"], linewidth=0.5, zorder=1)
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(stages, fontsize=10)
    ax_c.set_xlim(-0.6, 2.6)
    lows = panel_c["estimate"] - 1.96 * panel_c["se"]
    highs = panel_c["estimate"] + 1.96 * panel_c["se"]
    ax_c.set_ylim(float(lows.min() - 10), float(highs.max() + 15))
    ax_c.set_xlabel("Growth stage", fontsize=11)
    ax_c.set_ylabel("Yield Sensitivity\n(% per 0.1 m$^3$/m$^3$ SM)", fontsize=11)
    layer_legend = ax_c.legend(
        handles=[
            mpatches.Patch(facecolor=colors["L1"], alpha=0.85, edgecolor="none", label="Surface (L1)"),
            mpatches.Patch(facecolor=colors["L2L3"], alpha=0.85, edgecolor="none", label="Root zone (L2L3)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        fontsize=10,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    ax_c.add_artist(layer_legend)
    ax_c.legend(handles=sig_legend, loc="lower left", fontsize=8, handlelength=1.0, frameon=False, handletextpad=0.4)
    ax_c.text(0.01, 1.02, "c", transform=ax_c.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

    paths = [
        output_dir / "supplementary_figure_s6_gldas_yield_sensitivity.png",
        output_dir / "supplementary_figure_s6_gldas_yield_sensitivity.pdf",
    ]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    return paths


def load_s7_results() -> dict[int, dict]:
    results = {}
    for zone_id in range(1, 6):
        path = DATA_DIR / "gldas_sif" / f"koppen{zone_id}" / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing S7 GLDAS SIF result: {path}")
        with path.open("r") as handle:
            results[zone_id] = json.load(handle)
    return results


def write_s7(output_dir: Path) -> list[Path]:
    results = load_s7_results()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(180 / 25.4, 130 / 25.4), facecolor="white")
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 0.48], hspace=0.50, wspace=0.38, left=0.10, right=0.97, top=0.84, bottom=0.10)
    panel_labels = list("abcdefghij")
    fig.text(0.012, 0.58, "Individual variables", fontsize=9, fontweight="bold", ha="center", va="center", rotation=90, color="#333333")

    for idx, zone_id in enumerate(range(1, 6)):
        ax = fig.add_subplot(gs[0, idx])
        boot = results[zone_id].get("drop_column_bootstrap")
        if boot:
            items = sorted(boot.items(), key=lambda item: item[1]["mean"], reverse=True)
            labels = [VAR_LABELS[var] for var, _ in items]
            values = [stats["mean"] for _, stats in items]
            errors = np.array([[stats["mean"] - stats["ci_low"] for _, stats in items], [stats["ci_high"] - stats["mean"] for _, stats in items]])
            colors = [COLOR_MAP[FEATURE_GROUP[var]] for var, _ in items]
        else:
            items = sorted(results[zone_id]["drop_column_delta"].items(), key=lambda item: item[1], reverse=True)
            labels = [VAR_LABELS[var] for var, _ in items]
            values = [value for _, value in items]
            errors = None
            colors = [COLOR_MAP[FEATURE_GROUP[var]] for var, _ in items]
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, color=colors, height=0.68, edgecolor="none")
        if errors is not None:
            ax.errorbar(values, y_pos, xerr=errors, fmt="none", ecolor="#333333", elinewidth=0.7, capsize=2, capthick=0.6, clip_on=True)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linestyle="-", linewidth=0.3, alpha=0.5, color="#CCCCCC")
        ax.set_title(f"{KOPPEN_SHORT[zone_id]}\n($R^2$={results[zone_id]['metrics']['test_r2']:.2f})", fontsize=10, pad=5)
        ax.text(-0.18, 1.14, panel_labels[idx], transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.set_xlim(0, 0.04)
        ax.set_xticks([0, 0.02, 0.04])
        ax.set_xticklabels(["0", "0.02", "0.04"], fontsize=8)

    fig.text(0.012, 0.20, "Variable groups", fontsize=9, fontweight="bold", ha="center", va="center", rotation=90, color="#333333")
    x_max = 0.12
    for idx, zone_id in enumerate(range(1, 6)):
        ax = fig.add_subplot(gs[1, idx])
        boot = results[zone_id].get("drop_group_bootstrap")
        if boot:
            values = [boot[group]["mean"] for group in GROUP_ORDER]
            errors = np.array([[boot[group]["mean"] - boot[group]["ci_low"] for group in GROUP_ORDER], [boot[group]["ci_high"] - boot[group]["mean"] for group in GROUP_ORDER]])
        else:
            values = [results[zone_id]["drop_group_delta"].get(group, 0) for group in GROUP_ORDER]
            errors = None
        y_pos = np.arange(len(GROUP_ORDER))
        ax.barh(y_pos, values, color=[COLOR_MAP[group] for group in GROUP_ORDER], height=0.65, edgecolor="none")
        if errors is not None:
            ax.errorbar(values, y_pos, xerr=errors, fmt="none", ecolor="#333333", elinewidth=0.7, capsize=2.5, capthick=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(["Root", "Energy", "Surface"] if idx == 0 else [], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, x_max)
        ax.set_xticks([0, 0.04, 0.08, 0.12])
        ax.set_xticklabels(["0", "0.04", "0.08", "0.12"], fontsize=8)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linestyle="-", linewidth=0.3, alpha=0.5, color="#CCCCCC")
        ax.text(-0.18, 1.28, panel_labels[idx + 5], transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    fig.text(0.53, 0.015, r"$\Delta R^2$", fontsize=10, ha="center", va="bottom")
    fig.legend(
        handles=[
            mpatches.Patch(facecolor=COLOR_MAP["RootWater"], edgecolor="none", label="Root Water (SM L2, L3)"),
            mpatches.Patch(facecolor=COLOR_MAP["Energy"], edgecolor="none", label="Energy (VPD, SW, Tmax)"),
            mpatches.Patch(facecolor=COLOR_MAP["SurfaceWater"], edgecolor="none", label="Surface Water (SM L1, PPT)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        frameon=False,
        fontsize=9,
        ncol=3,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.5,
    )
    paths = [
        output_dir / "supplementary_figure_s7_gldas_sif_attribution.png",
        output_dir / "supplementary_figure_s7_gldas_sif_attribution.pdf",
    ]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)
    return paths


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = weights.sum()
    y_mean = np.sum(weights * y_true) / (weight_sum + 1e-12)
    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_true - y_mean) ** 2)
    if ss_tot <= 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def write_s8(output_dir: Path) -> list[Path]:
    source = DATA_DIR / "sif_predictions" / "s8_sif_pred_vs_obs.csv.gz"
    if not source.exists():
        raise FileNotFoundError(f"Missing S8 prediction data: {source}")
    df = pd.read_csv(source)
    required = {"koppen_id", "observed_sif_anom", "predicted_sif_anom", "sample_weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"S8 prediction data missing columns: {sorted(missing)}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(180 / 25.4, 130 / 25.4), facecolor="white")
    gs = fig.add_gridspec(2, 6, hspace=0.05, wspace=0.30, left=0.09, right=0.97, top=0.97, bottom=0.08)
    panel_slices = [(0, 0, 2), (0, 2, 4), (0, 4, 6), (1, 1, 3), (1, 3, 5)]
    panel_labels = "abcde"
    limit = np.ceil(max(df["observed_sif_anom"].abs().max(), df["predicted_sif_anom"].abs().max()) * 100) / 100

    for idx, zone_id in enumerate(range(1, 6)):
        zone = df[df["koppen_id"] == zone_id]
        r, c0, c1 = panel_slices[idx]
        ax = fig.add_subplot(gs[r, c0:c1])
        observed = zone["observed_sif_anom"].to_numpy()
        predicted = zone["predicted_sif_anom"].to_numpy()
        weights = zone["sample_weight"].to_numpy()
        r2 = weighted_r2(observed, predicted, weights)

        ax.hexbin(observed, predicted, gridsize=40, cmap="magma", mincnt=1, linewidths=0, edgecolors="none", norm=mcolors.LogNorm())
        ax.plot([-limit, limit], [-limit, limit], color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=5)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.text(0.88, 0.05, f"$R^2$={r2:.2f}\n$N$={len(zone):,}", transform=ax.transAxes, fontsize=9, va="bottom", ha="right", color="#333333")
        ax.text(0.05, 0.95, panel_labels[idx], transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")
        ax.text(0.14, 0.95, KOPPEN_SHORT[zone_id], transform=ax.transAxes, fontsize=10, va="top", ha="left")
        ax.set_ylabel("Predicted SIF anomaly" if idx in {0, 3} else "", fontsize=9)
        ax.set_xlabel("Observed SIF anomaly" if r == 1 else "", fontsize=9)

    paths = [
        output_dir / "supplementary_figure_s8_sif_pred_vs_obs.png",
        output_dir / "supplementary_figure_s8_sif_pred_vs_obs.pdf",
    ]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)
    return paths


def write_s9(output_dir: Path) -> list[Path]:
    validate_s9_source()
    df = pd.read_csv(DATA_DIR / "shapley_r2" / "s9_shapley_group_decomposition.csv")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(180 / 25.4, 90 / 25.4), facecolor="white")
    fig.subplots_adjust(left=0.14, right=0.88, top=0.86, bottom=0.16)
    y_pos = np.arange(5)
    left = np.zeros(5)
    zones = sorted(df["koppen_id"].unique())
    for group in GROUP_ORDER:
        values = np.array([df[(df["koppen_id"] == zone) & (df["feature_group"] == group)]["shapley_r2"].iloc[0] for zone in zones])
        ax.barh(y_pos, values, left=left, height=0.52, color=COLOR_MAP[group], edgecolor="white", linewidth=0.4)
        for idx, zone in enumerate(zones):
            total = df[df["koppen_id"] == zone]["shapley_r2"].sum()
            if values[idx] > 0.012:
                ax.text(left[idx] + values[idx] / 2, y_pos[idx], f"{values[idx] / total * 100:.0f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        left += values

    for idx, zone in enumerate(zones):
        total = df[df["koppen_id"] == zone]["shapley_r2"].sum()
        r2 = df[df["koppen_id"] == zone]["full_model_r2"].iloc[0]
        ax.text(total + 0.012, y_pos[idx], f"$R^2$={r2:.2f}", va="center", ha="left", fontsize=10, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([KOPPEN_SHORT[int(zone)] for zone in zones], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel(r"Shapley $\varphi$ (contribution to $R^2$)", fontsize=11)
    ax.set_xlim(0, df["full_model_r2"].max() * 1.32)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.3, alpha=0.5, color="#CCCCCC")
    fig.legend(
        handles=[
            mpatches.Patch(facecolor=COLOR_MAP["RootWater"], edgecolor="none", label="Root Water (SM L2, L3)"),
            mpatches.Patch(facecolor=COLOR_MAP["Energy"], edgecolor="none", label="Energy (VPD, SW, Tmax)"),
            mpatches.Patch(facecolor=COLOR_MAP["SurfaceWater"], edgecolor="none", label="Surface Water (SM L1, PPT)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.46, 0.99),
        frameon=False,
        fontsize=10,
        ncol=3,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    paths = [
        output_dir / "supplementary_figure_s9_shapley_r2_decomposition.png",
        output_dir / "supplementary_figure_s9_shapley_r2_decomposition.pdf",
    ]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)
    return paths


def load_s10_results() -> dict[int, dict]:
    validate_s10_source()
    results = {}
    for zone_id in range(1, 6):
        path = DATA_DIR / "hard_energy_filtering" / f"koppen{zone_id}" / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing S10 hard-filter result: {path}")
        with path.open("r") as handle:
            results[zone_id] = json.load(handle)
    return results


def write_s10(output_dir: Path) -> list[Path]:
    results = load_s10_results()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(180 / 25.4, 130 / 25.4), facecolor="white")
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 0.48], hspace=0.50, wspace=0.38, left=0.10, right=0.97, top=0.84, bottom=0.10)
    panel_labels = list("abcdefghij")
    fig.text(0.012, 0.58, "Individual variables", fontsize=9, fontweight="bold", ha="center", va="center", rotation=90, color="#333333")

    for idx, zone_id in enumerate(range(1, 6)):
        ax = fig.add_subplot(gs[0, idx])
        boot = results[zone_id]["drop_column_bootstrap"]
        items = sorted(boot.items(), key=lambda item: item[1]["mean"], reverse=True)
        labels = [VAR_LABELS[var] for var, _ in items]
        values = [stats["mean"] for _, stats in items]
        errors = np.array([[stats["mean"] - stats["ci_low"] for _, stats in items], [stats["ci_high"] - stats["mean"] for _, stats in items]])
        colors = [COLOR_MAP[FEATURE_GROUP[var]] for var, _ in items]
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, color=colors, height=0.68, edgecolor="none")
        ax.errorbar(values, y_pos, xerr=errors, fmt="none", ecolor="#333333", elinewidth=0.7, capsize=2, capthick=0.6, clip_on=True)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linestyle="-", linewidth=0.3, alpha=0.5, color="#CCCCCC")
        ax.set_title(f"{KOPPEN_SHORT[zone_id]}\n($R^2$={results[zone_id]['metrics']['test_r2']:.2f})", fontsize=10, pad=5)
        ax.text(-0.18, 1.14, panel_labels[idx], transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.set_xlim(0, 0.095)
        ax.set_xticks([0, 0.04, 0.08])
        ax.set_xticklabels(["0", "0.04", "0.08"], fontsize=8)

    fig.text(0.012, 0.20, "Variable groups", fontsize=9, fontweight="bold", ha="center", va="center", rotation=90, color="#333333")
    x_max = max(results[zone_id]["drop_group_bootstrap"][group]["mean"] for zone_id in range(1, 6) for group in GROUP_ORDER) * 1.15
    for idx, zone_id in enumerate(range(1, 6)):
        ax = fig.add_subplot(gs[1, idx])
        boot = results[zone_id]["drop_group_bootstrap"]
        values = [boot[group]["mean"] for group in GROUP_ORDER]
        errors = np.array([[boot[group]["mean"] - boot[group]["ci_low"] for group in GROUP_ORDER], [boot[group]["ci_high"] - boot[group]["mean"] for group in GROUP_ORDER]])
        y_pos = np.arange(len(GROUP_ORDER))
        ax.barh(y_pos, values, color=[COLOR_MAP[group] for group in GROUP_ORDER], height=0.65, edgecolor="none")
        ax.errorbar(values, y_pos, xerr=errors, fmt="none", ecolor="#333333", elinewidth=0.7, capsize=2.5, capthick=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(["Root", "Energy", "Surface"] if idx == 0 else [], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, x_max)
        ax.set_xticks([0, 0.05, 0.10, 0.15])
        ax.set_xticklabels(["0", "0.05", "0.10", "0.15"], fontsize=8)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linestyle="-", linewidth=0.3, alpha=0.5, color="#CCCCCC")
        ax.text(-0.18, 1.28, panel_labels[idx + 5], transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    fig.text(0.53, 0.015, r"$\Delta R^2$", fontsize=10, ha="center", va="bottom")
    fig.legend(
        handles=[
            mpatches.Patch(facecolor=COLOR_MAP["RootWater"], edgecolor="none", label="Root Water (SM L2, L3)"),
            mpatches.Patch(facecolor=COLOR_MAP["Energy"], edgecolor="none", label="Energy (VPD, SW, Tmax)"),
            mpatches.Patch(facecolor=COLOR_MAP["SurfaceWater"], edgecolor="none", label="Surface Water (SM L1, PPT)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        frameon=False,
        fontsize=9,
        ncol=3,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.5,
    )
    paths = [
        output_dir / "supplementary_figure_s10_hard_energy_filtering.png",
        output_dir / "supplementary_figure_s10_hard_energy_filtering.pdf",
    ]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce supplement figures from packaged inputs.")
    parser.add_argument("--quick", action="store_true", help="Accepted for command compatibility; all panels are lightweight.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    figure_specs = [
        ("S3 GLEAM SM-VPD coupling", write_s3),
        ("S4 GLDAS Noah SM-VPD coupling", write_s4),
        ("S5 GLDAS Noah nonlinear yield response", write_s5),
        ("S6 GLDAS Noah yield sensitivity", write_s6),
        ("S7 GLDAS Noah SIF attribution", write_s7),
        ("S8 SIF predicted vs observed", write_s8),
        ("S9 Shapley R2 decomposition", write_s9),
        ("S10 hard energy filtering", write_s10),
    ]
    for label, writer in figure_specs:
        try:
            paths = writer(OUTPUT_DIR)
            report.append({"item": label, "status": "written", "paths": [rel(path) for path in paths]})
        except Exception as exc:
            report.append({"item": label, "status": "skipped", "reason": str(exc)})

    report_path = OUTPUT_DIR / "supplement_run_report.json"
    with report_path.open("w") as handle:
        json.dump({"quick": bool(args.quick), "items": report}, handle, indent=2)
    for item in report:
        label = item["item"]
        status = item["status"]
        detail = ", ".join(item.get("paths", [])) or item.get("reason", "")
        print(f"[supplement] {label}: {status} {detail}")
    print(f"[supplement] Wrote {rel(report_path)}")


if __name__ == "__main__":
    main()

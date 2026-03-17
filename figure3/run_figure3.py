#!/usr/bin/env python3

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
PANEL_FILE = HERE / "data" / "admin2_year_panel_v3.csv"
PANEL_STAGES_FILE = HERE / "data" / "admin2_year_panel_stages_v3.csv"
OUT_FILE = HERE / "outputs" / "figure3_panel_effect_estimates.png"


COLORS = {
    "vpd": "#2166AC",
    "L1": "#5DADE2",
    "L2L3": "#006400",
    "gray": "#4A4A4A",
    "light_gray": "#BFBFBF",
    "annotation": "#333333",
}

plt.rcParams.update(
    {
        "font.family": "Arial",
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
    }
)


def calculate_overall_rsquared(model_result, y):
    resid = model_result.resids
    y_mean = y.mean()
    ss_tot = ((y - y_mean) ** 2).sum()
    ss_res = (resid ** 2).sum()
    return 1 - (ss_res / ss_tot)


def run_panel_fe(df, y_var, exog_vars, return_detailed=False):
    df_reg = df.dropna(subset=[y_var] + exog_vars).copy()
    df_reg["admin2_idx"] = df_reg["admin2_idx"].astype(int)
    df_reg["year"] = df_reg["year"].astype(int)
    df_panel = df_reg.set_index(["admin2_idx", "year"])

    y = df_panel[y_var]
    X = df_panel[exog_vars]

    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    results = model.fit(cov_type="clustered", cluster_entity=True)

    r2_within = results.rsquared
    r2_overall = calculate_overall_rsquared(results, y)

    if return_detailed:
        return results, r2_within, r2_overall, len(df_reg)
    return results, r2_within, r2_overall


def coef_to_pct_change(beta, se):
    pct = (np.exp(beta) - 1) * 100
    se_pct = np.exp(beta) * se * 100
    return pct, se_pct


def get_significance(pval):
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return ""


def main():
    HERE.joinpath("outputs").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PANEL_FILE)
    control_vars = ["ppt_gs_sum", "tmax_gs_mean"]
    if not df["rad_gs_mean"].isna().all():
        control_vars.append("rad_gs_mean")

    model_specs = {
        "M1": ["vpd_gs_mean", "sm_L1_gs_mean"] + control_vars,
        "M2": ["vpd_gs_mean", "sm_L2L3_gs_mean"] + control_vars,
        "M3": ["vpd_gs_mean", "sm_L1_gs_mean", "sm_L2L3_gs_mean"] + control_vars,
    }

    models = {}
    r2_results = {}
    for name, exog in model_specs.items():
        result, r2_within, r2_overall, n_obs = run_panel_fe(df, "log_yield", exog, return_detailed=True)
        models[name] = result
        r2_results[name] = {"within_r2": r2_within, "overall_r2": r2_overall, "n_obs": n_obs}

    df_stages = pd.read_csv(PANEL_STAGES_FILE)
    stage_vars = ["sm_L1_veg", "sm_L1_rep", "sm_L1_mat", "sm_L2L3_veg", "sm_L2L3_rep", "sm_L2L3_mat"]
    control_stage_vars = ["vpd_veg", "vpd_rep", "vpd_mat", "ppt_veg", "ppt_rep", "ppt_mat", "tmax_veg", "tmax_rep", "tmax_mat", "rad_veg", "rad_rep", "rad_mat"]
    df_stages = df_stages.dropna(subset=stage_vars + control_stage_vars + ["log_yield"])

    stage_results = []
    stages = ["Vegetative", "Reproductive", "Maturation"]
    stage_abbrev = {"Vegetative": "veg", "Reproductive": "rep", "Maturation": "mat"}

    for stage in stages:
        abbr = stage_abbrev[stage]
        exog = [f"vpd_{abbr}", f"sm_L1_{abbr}", f"sm_L2L3_{abbr}", f"ppt_{abbr}", f"tmax_{abbr}", f"rad_{abbr}"]
        model, _, _, _ = run_panel_fe(df_stages, "log_yield", exog, return_detailed=True)
        for var_name, label in [(f"sm_L1_{abbr}", "L1"), (f"sm_L2L3_{abbr}", "L2L3")]:
            beta = model.params[var_name]
            se = model.std_errors[var_name]
            pval = model.pvalues[var_name]
            pct = (np.exp(beta * 0.1) - 1) * 100
            pct_se = np.exp(beta * 0.1) * se * 0.1 * 100
            stage_results.append({"stage": stage, "var": label, "coef": pct, "se": pct_se, "pval": pval})

    df_stage = pd.DataFrame(stage_results)

    fig = plt.figure(figsize=(7.0, 5.8), facecolor="white")
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 0.85],
        width_ratios=[1, 1],
        wspace=0.22,
        hspace=0.35,
        left=0.14,
        right=0.97,
        top=0.93,
        bottom=0.08,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    vpd_models = [("M1", "VPD", "+L1"), ("M2", "VPD", "+L2L3"), ("M3", "VPD", "+L1+L2L3")]
    x_pos = np.arange(len(vpd_models))
    vpd_pcts, vpd_ses, vpd_pvals = [], [], []
    for model_name, _, _ in vpd_models:
        result = models[model_name]
        beta = result.params["vpd_gs_mean"]
        se = result.std_errors["vpd_gs_mean"]
        pval = result.pvalues["vpd_gs_mean"]
        pct, se_pct = coef_to_pct_change(beta, se)
        vpd_pcts.append(pct)
        vpd_ses.append(se_pct)
        vpd_pvals.append(pval)

    bar_width = 0.45
    ax_a.bar(x_pos, vpd_pcts, width=bar_width, color=COLORS["vpd"], alpha=0.85, edgecolor="white", linewidth=0.5)
    ax_a.errorbar(x_pos, vpd_pcts, yerr=vpd_ses, fmt="none", ecolor=COLORS["gray"], elinewidth=0.8, capsize=2, capthick=0.8)
    for i, (pct, se, pval) in enumerate(zip(vpd_pcts, vpd_ses, vpd_pvals)):
        sig = get_significance(pval)
        if sig:
            ax_a.text(i, pct + se - 1.5, sig, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels([])
    for xp, (_, red_part, black_part) in zip(x_pos, [("M1", "VPD", "+L1"), ("M2", "VPD", "+L2L3"), ("M3", "VPD", "+L1+L2L3")]):
        ta1 = TextArea(red_part, textprops=dict(color="#B22222", fontsize=10, fontweight="bold", fontfamily="Arial"))
        ta2 = TextArea(black_part, textprops=dict(color="black", fontsize=10, fontfamily="Arial"))
        packed = HPacker(children=[ta1, ta2], pad=0, sep=0, align="baseline")
        annotation = AnnotationBbox(packed, (xp, 0), xycoords=("data", "axes fraction"), box_alignment=(0.5, 1.0), frameon=False, xybox=(0, -3), boxcoords="offset points")
        ax_a.add_artist(annotation)
    ax_a.set_xlabel("Model specification", fontsize=11, labelpad=10)
    ax_a.set_ylabel("Yield Sensitivity\n(% per kPa VPD)", fontsize=11)
    ax_a.set_ylim([0, max(p + s for p, s in zip(vpd_pcts, vpd_ses)) + 5])
    ax_a.set_xlim(-0.5, 2.5)
    ax_a.axhline(y=0, color=COLORS["light_gray"], linewidth=0.5)
    ax_a.axvline(x=1.5, color="black", linestyle="--", linewidth=0.8)
    ax_a.annotate("", xy=(2, vpd_pcts[2] + vpd_ses[2] + 0.8), xytext=(0, vpd_pcts[0] + vpd_ses[0] + 0.8), arrowprops=dict(arrowstyle="->", color=COLORS["annotation"], lw=1.0, connectionstyle="arc3,rad=-0.15"))
    ax_a.text(0.7, max(vpd_pcts[0], vpd_pcts[1]) + max(vpd_ses[0], vpd_ses[1]) - 7, f"{vpd_pcts[0]:.1f}% -> {vpd_pcts[2]:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=COLORS["annotation"], bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.9))
    ax_a.text(0.02, 1.02, "a", transform=ax_a.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

    ax_b = fig.add_subplot(gs[0, 1])
    sm_data = [("M1", "sm_L1_gs_mean", "L1", COLORS["L1"]), ("M2", "sm_L2L3_gs_mean", "L2L3", COLORS["L2L3"]), ("M3", "sm_L1_gs_mean", "L1", COLORS["L1"]), ("M3", "sm_L2L3_gs_mean", "L2L3", COLORS["L2L3"])]
    x_pos_b = [0, 1, 1.75, 2.25]
    sm_pcts, sm_ses, sm_pvals, sm_colors = [], [], [], []
    for model_name, var_name, _, color in sm_data:
        result = models[model_name]
        beta = result.params[var_name]
        se = result.std_errors[var_name]
        pval = result.pvalues[var_name]
        pct, se_pct = coef_to_pct_change(beta * 0.1, se * 0.1)
        sm_pcts.append(pct)
        sm_ses.append(se_pct)
        sm_pvals.append(pval)
        sm_colors.append(color)

    for xp, pct, color, pval in zip(x_pos_b, sm_pcts, sm_colors, sm_pvals):
        if pval < 0.05:
            ax_b.bar(xp, pct, width=bar_width, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        else:
            ax_b.bar(xp, pct, width=bar_width, color=color, alpha=0.25, edgecolor=color, linewidth=1.0, hatch="///")
    ax_b.errorbar(x_pos_b, sm_pcts, yerr=sm_ses, fmt="none", ecolor=COLORS["gray"], elinewidth=0.8, capsize=2, capthick=0.8)

    for xp, pct, se, pval in zip(x_pos_b, sm_pcts, sm_ses, sm_pvals):
        sig = get_significance(pval)
        if sig:
            y_loc = pct + se - 1 if pct >= 0 else pct - se + 1
            va = "bottom" if pct >= 0 else "top"
            ax_b.text(xp, y_loc, sig, ha="center", va=va, fontsize=9, fontweight="bold")

    for xp, label, color, pct, se in [(1.75, "L1", COLORS["L1"], sm_pcts[2], sm_ses[2]), (2.25, "L2L3", COLORS["L2L3"], sm_pcts[3], sm_ses[3])]:
        ax_b.text(xp, max(pct + se, 0) + 1.5, label, ha="center", va="bottom", fontsize=10, color=color, fontweight="bold")

    ax_b.set_xticks([0, 1, 2])
    ax_b.set_xticklabels([])
    for xp, (black_part, red_part) in zip([0, 1, 2], [("VPD+", "L1"), ("VPD+", "L2L3"), ("VPD+", "L1+L2L3")]):
        ta1 = TextArea(black_part, textprops=dict(color="black", fontsize=10, fontfamily="Arial"))
        ta2 = TextArea(red_part, textprops=dict(color="#B22222", fontsize=10, fontweight="bold", fontfamily="Arial"))
        packed = HPacker(children=[ta1, ta2], pad=0, sep=0, align="baseline")
        annotation = AnnotationBbox(packed, (xp, 0), xycoords=("data", "axes fraction"), box_alignment=(0.5, 1.0), frameon=False, xybox=(0, -3), boxcoords="offset points")
        ax_b.add_artist(annotation)
    ax_b.set_xlabel("Model specification", fontsize=11, labelpad=10)
    ax_b.set_ylabel("Yield Sensitivity\n(% per 0.1 m^3/m^3 SM)", fontsize=11)
    ax_b.set_ylim([min(p - s for p, s in zip(sm_pcts, sm_ses)) - 3, max(p + s for p, s in zip(sm_pcts, sm_ses)) + 8])
    ax_b.set_xlim(-0.5, 2.7)
    ax_b.axhline(y=0, color=COLORS["light_gray"], linewidth=0.5)
    ax_b.axvline(x=1.5, color="black", linestyle="--", linewidth=0.8)
    ax_b.legend(handles=[Patch(facecolor=COLORS["gray"], alpha=0.85, edgecolor="none", label="P < 0.05"), Patch(facecolor=COLORS["gray"], alpha=0.25, hatch="///", edgecolor=COLORS["gray"], label="n.s.")], loc="lower left", fontsize=8, handlelength=1.0, frameon=False, handletextpad=0.4)
    ax_b.text(0.02, 1.02, "b", transform=ax_b.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

    ax_c = fig.add_subplot(gs[1, :])
    x_c = np.arange(len(stages))
    width_c = 0.28
    for i, stage in enumerate(stages):
        subset = df_stage[df_stage["stage"] == stage]
        for var_name, color, offset in [("L1", COLORS["L1"], -width_c / 2), ("L2L3", COLORS["L2L3"], width_c / 2)]:
            row = subset[subset["var"] == var_name].iloc[0]
            if row["pval"] < 0.05:
                ax_c.bar(i + offset, row["coef"], width_c, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
            else:
                ax_c.bar(i + offset, row["coef"], width_c, color=color, alpha=0.25, edgecolor=color, linewidth=1.0, hatch="///")
            ax_c.errorbar(i + offset, row["coef"], yerr=1.96 * row["se"], fmt="none", ecolor=COLORS["gray"], elinewidth=0.8, capsize=2, capthick=0.8, zorder=5)
            sig = get_significance(row["pval"])
            if sig:
                y_loc = row["coef"] + 1.96 * row["se"] - 2.5 if row["coef"] > 0 else row["coef"] - 1.96 * row["se"] - 1
                va = "bottom" if row["coef"] > 0 else "top"
                ax_c.text(i + offset, y_loc, sig, ha="center", va=va, fontsize=9, fontweight="bold")

    ax_c.axhline(y=0, color=COLORS["light_gray"], linewidth=0.5, zorder=1)
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(stages, fontsize=10)
    ax_c.set_xlim(-0.6, 2.6)
    all_bounds = []
    for _, row in df_stage.iterrows():
        all_bounds.extend([row["coef"] - 1.96 * row["se"], row["coef"] + 1.96 * row["se"]])
    ax_c.set_ylim(min(all_bounds) - 10, max(all_bounds) + 15)
    ax_c.set_xlabel("Growth stage", fontsize=11)
    ax_c.set_ylabel("Yield Sensitivity\n(% per 0.1 m^3/m^3 SM)", fontsize=11)
    legend_layers = ax_c.legend(handles=[Patch(facecolor=COLORS["L1"], alpha=0.85, edgecolor="none", label="Surface (L1)"), Patch(facecolor=COLORS["L2L3"], alpha=0.85, edgecolor="none", label="Root zone (L2L3)")], loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=10, frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.0)
    ax_c.add_artist(legend_layers)
    ax_c.legend(handles=[Patch(facecolor=COLORS["gray"], alpha=0.85, edgecolor="none", label="P < 0.05"), Patch(facecolor=COLORS["gray"], alpha=0.25, hatch="///", edgecolor=COLORS["gray"], label="n.s.")], loc="lower left", fontsize=8, handlelength=1.0, frameon=False, handletextpad=0.4)
    ax_c.spines["left"].set_color(COLORS["light_gray"])
    ax_c.spines["bottom"].set_color(COLORS["light_gray"])
    ax_c.text(0.01, 1.02, "c", transform=ax_c.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

    fig.savefig(OUT_FILE, dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()

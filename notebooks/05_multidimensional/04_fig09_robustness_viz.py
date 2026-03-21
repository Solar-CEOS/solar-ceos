#!/usr/bin/env python3
"""
04_fig09_robustness_viz.py
==========================
Robustness figure for sec5: null planet test, bootstrap CI, tidal correlation.

Inputs (from 04_asymmetric/10_robustness_tests/):
  - null_planet_fake_results.csv
  - null_planet_real_results.csv
  - bootstrap_ci.csv
  - tidal_correlation.csv

Outputs:
  - results/05_multidimensional/Fig09_robustness_combined.eps
  - results/05_multidimensional/Fig09_robustness_combined.png
  - submission/figures_2026/Fig09_robustness_combined.eps
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "05_multidimensional")
FIG_DIR = os.path.join(PROJECT_ROOT, "submission", "figures_2026")
INPUT_DIR = os.path.join(PROJECT_ROOT, "results", "05_multidimensional", "10_robustness_tests")
FIG09_FIG_BASENAME = "Fig09_robustness_combined"

COLOR_RED = "#c0392b"
COLOR_BLUE = "#1f618d"
COLOR_GREEN = "#27ae60"
COLOR_GRAY = "#95a5a6"


def plot_null_test(ax: plt.Axes) -> None:
    fake = pd.read_csv(os.path.join(INPUT_DIR, "null_planet_fake_results.csv"))
    real = pd.read_csv(os.path.join(INPUT_DIR, "null_planet_real_results.csv"))

    fake_ratios = fake["Conj_Ratio"].values
    # Exclude near-Earth-period artefact (Conj_Ratio = 0 for ~365 d orbit)
    fake_ratios = fake_ratios[fake_ratios > 0]
    mean_fake = np.mean(fake_ratios)
    std_fake = np.std(fake_ratios)

    ax.hist(fake_ratios, bins=20, color="#d6eaf8", edgecolor=COLOR_BLUE,
            linewidth=0.8, alpha=0.9, label=f"{len(fake_ratios)} fake planets", zorder=2)

    # Real planet lines
    highlight = {"Venus": COLOR_RED, "Mars": "#d68910"}
    for _, row in real.iterrows():
        name = row["Name"]
        rc = row["Conj_Ratio"]
        if name in highlight:
            sigma = (rc - mean_fake) / std_fake
            ax.axvline(rc, color=highlight[name], linewidth=2.2, linestyle="-",
                       label=f"{name} = {rc:.1f}% ({sigma:.1f}σ)", zorder=4)

    ax.axvline(100, color="gray", linewidth=0.8, linestyle=":", zorder=1)

    # False positive annotation
    n_sig = (fake["Conj_p"] < 0.05).sum()
    pct = n_sig / len(fake) * 100
    ax.text(0.03, 0.55,
            f"FPR: {n_sig}/{len(fake)} ({pct:.0f}%)\nExpected: 5%",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_xlim(70, None)
    ax.set_xlabel("Conjunction ratio $R_C$ (%)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Null planet test")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(axis="y", color="#e5e7e9", linewidth=0.8)


def plot_bootstrap_forest(ax: plt.Axes) -> None:
    boot = pd.read_csv(os.path.join(INPUT_DIR, "bootstrap_ci.csv"))

    labels = boot["Combo"].tolist()
    ratios = boot["Conj_Ratio"].values
    ci_lo = boot["Conj_CI_lo"].values
    ci_hi = boot["Conj_CI_hi"].values
    n = len(labels)
    y_pos = np.arange(n)[::-1]

    for i in range(n):
        if ci_lo[i] > 100:
            color = COLOR_GREEN
        else:
            color = COLOR_GRAY

        ax.errorbar(ratios[i], y_pos[i],
                     xerr=[[ratios[i] - ci_lo[i]], [ci_hi[i] - ratios[i]]],
                     fmt="o", markersize=7, color=color,
                     ecolor=color, elinewidth=2, capsize=4, capthick=1.5,
                     zorder=3)

        ci_text = f"{ratios[i]:.1f}% [{ci_lo[i]:.1f}, {ci_hi[i]:.1f}]"
        ax.text(ci_hi[i] + 1.5, y_pos[i], ci_text, va="center", fontsize=7.5,
                color="#333333")

    ax.axvline(100, color="gray", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Conjunction ratio $R_C$ (%)")
    ax.set_title("(b) Bootstrap 95% CI")
    ax.set_xlim(min(ci_lo) - 5, max(ci_hi) + 25)
    ax.grid(axis="x", color="#e5e7e9", linewidth=0.8)


def plot_tidal_scatter(ax: plt.Axes) -> None:
    tidal = pd.read_csv(os.path.join(INPUT_DIR, "tidal_correlation.csv"))

    x = tidal["Log_Tidal"].values
    y_rc = tidal["Conj_Ratio"].values
    y_asym = tidal["Asym"].values

    planet_colors = {
        "Venus": "#c0392b", "Mars": "#d68910", "Jupiter": "#2980b9",
        "Saturn": "#8e44ad", "Mercury": "#16a085", "Uranus": "#27ae60",
        "Neptune": "#2c3e50",
    }

    from scipy.stats import spearmanr
    sp_rc, sp_rc_p = spearmanr(x, y_rc)
    sp_asym, sp_asym_p = spearmanr(x, y_asym)

    # Per-planet offsets to avoid label overlap
    label_offsets = {
        "Venus": (8, 4), "Mercury": (-5, 8), "Jupiter": (8, -12),
        "Saturn": (8, 4), "Mars": (8, 4),
        "Uranus": (8, 4), "Neptune": (8, -10),
    }

    # Left axis: R_C (circles)
    for i, (_, row) in enumerate(tidal.iterrows()):
        c = planet_colors.get(row["Planet"], "gray")
        sig = row["Conj_p"] < 0.05
        size = 70 if sig else 45
        ax.scatter(row["Log_Tidal"], row["Conj_Ratio"], s=size, c=c,
                   marker="o", edgecolors="black" if sig else "#666666",
                   linewidth=1.2, zorder=4)
        ofs = label_offsets.get(row["Planet"], (6, 2))
        ax.annotate(row["Planet"],
                    xy=(row["Log_Tidal"], row["Conj_Ratio"]),
                    xytext=ofs, textcoords="offset points",
                    fontsize=7.5, color=c, fontweight="bold")

    # R_C linear fit
    slope, intercept = np.polyfit(x, y_rc, 1)
    x_fit = np.linspace(x.min() - 0.3, x.max() + 0.3, 50)
    ax.plot(x_fit, slope * x_fit + intercept, "--", color=COLOR_RED,
            alpha=0.5, linewidth=1.2)

    ax.axhline(100, color="gray", linewidth=0.8, linestyle=":", zorder=1)

    # Right axis: Asym (triangles)
    ax2 = ax.twinx()
    for _, row in tidal.iterrows():
        c = planet_colors.get(row["Planet"], "gray")
        ax2.scatter(row["Log_Tidal"], row["Asym"], s=40, c=c,
                    marker="^", edgecolors=c, linewidth=0.8,
                    alpha=0.6, zorder=3)

    # Asym linear fit
    slope_a, intercept_a = np.polyfit(x, y_asym, 1)
    ax2.plot(x_fit, slope_a * x_fit + intercept_a, "-.", color=COLOR_BLUE,
             alpha=0.4, linewidth=1.2)

    ax2.axhline(0, color=COLOR_BLUE, linewidth=0.6, linestyle=":", alpha=0.4, zorder=1)
    ax2.set_ylabel("Asym (%)", color=COLOR_BLUE)
    ax2.tick_params(axis="y", labelcolor=COLOR_BLUE)

    # Legend with full info (no separate annotation needed)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markeredgecolor="black", markersize=7,
               label=f"$R_C$  ($r_s$ = {sp_rc:.2f}, $p$ = {sp_rc_p:.3f})"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR_BLUE,
               markeredgecolor=COLOR_BLUE, markersize=7, alpha=0.7,
               label=f"Asym ($r_s$ = {sp_asym:.2f}, $p$ = {sp_asym_p:.3f})"),
        Line2D([0], [0], linestyle="--", color=COLOR_RED, alpha=0.5,
               label="$R_C$ fit"),
        Line2D([0], [0], linestyle="-.", color=COLOR_BLUE, alpha=0.4,
               label="Asym fit"),
    ]
    ax.legend(handles=legend_elements, frameon=True, loc="upper left",
              fontsize=7.5, framealpha=0.9, edgecolor="#cccccc")

    ax.set_xlabel("$\\log_{10}(M/r^3)$  [kg m$^{-3}$]")
    ax.set_ylabel("Conjunction ratio $R_C$ (%)")
    ax.set_title("(c) Tidal force vs $R_C$")
    ax.grid(color="#e5e7e9", linewidth=0.8)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5),
                             gridspec_kw={"width_ratios": [1.1, 1.2, 1.0],
                                          "wspace": 0.35})

    plot_null_test(axes[0])
    plot_bootstrap_forest(axes[1])
    plot_tidal_scatter(axes[2])

    for ext in ["eps", "png"]:
        out = os.path.join(OUTPUT_DIR, f"{FIG09_FIG_BASENAME}.{ext}")
        plt.savefig(out, format=ext, dpi=300, bbox_inches="tight")

    fig_out = os.path.join(FIG_DIR, f"{FIG09_FIG_BASENAME}.eps")
    plt.savefig(fig_out, format="eps", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Robustness combined figure generated.")


if __name__ == "__main__":
    main()

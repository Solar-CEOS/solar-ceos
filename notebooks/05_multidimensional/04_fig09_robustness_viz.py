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
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style.figstyle import apply_acta_style, axis_log10, axis_unit, figsize_double, save_dual
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "05_multidimensional")
INPUT_DIR = os.path.join(PROJECT_ROOT, "results", "05_multidimensional", "10_robustness_tests")
FIG09_FIG_BASENAME = "Fig09_robustness_combined"

COLOR_RED = "#c0392b"
COLOR_BLUE = "#1f618d"
COLOR_GREEN = "#27ae60"
COLOR_GRAY = "#95a5a6"
COLOR_ORANGE = "#d68910"
COLOR_LIGHT_BLUE = "#dbeaf4"
COLOR_LIGHT_RED = "#e8a6a1"
COLOR_LIGHT_FIT_BLUE = "#9fbed3"

PLANET_COLORS = {
    "Venus": "#c0392b",
    "Mars": "#d68910",
    "Jupiter": "#2980b9",
    "Saturn": "#8e44ad",
    "Mercury": "#16a085",
    "Uranus": "#27ae60",
    "Neptune": "#2c3e50",
}

TIDAL_PLANET_ORDER = ["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
BOOT_LABELS = {
    "Venus": "V",
    "Mars": "M",
    "Jupiter": "J",
    "Saturn": "S",
    "Ven+Mar": "V+M",
    "Ven+Mar+Jup": "V+M+J",
    "Ven+Mar+Jup+Sat": "V+M+J+S",
    "All 5": "All 5",
}
RC_AXIS_LABEL = "Conjunction ratio " + axis_unit(r"R_{\mathrm{C}}", r"\%")
ASYM_AXIS_LABEL = axis_unit(r"\mathrm{Asym}", r"\%")
TIDAL_AXIS_LABEL = axis_log10(r"M/r^3", r"kg\cdot m^{-3}")


def _style_box(
    ax: plt.Axes,
    text: str,
    x: float,
    y: float,
    *,
    ha: str = "left",
    fontsize: float = 7.2,
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=fontsize,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#bdbdbd"),
    )


def _style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#bdbdbd")
    frame.set_linewidth(0.5)
    frame.set_alpha(1.0)


def plot_null_test(ax: plt.Axes) -> None:
    fake = pd.read_csv(os.path.join(INPUT_DIR, "null_planet_fake_results.csv"))
    real = pd.read_csv(os.path.join(INPUT_DIR, "null_planet_real_results.csv"))

    fake_ratios = fake["Conj_Ratio"].values
    # Exclude near-Earth-period artefact (Conj_Ratio = 0 for ~365 d orbit)
    fake_ratios = fake_ratios[fake_ratios > 0]
    mean_fake = np.mean(fake_ratios)
    std_fake = np.std(fake_ratios)

    ax.hist(
        fake_ratios,
        bins=18,
        color=COLOR_LIGHT_BLUE,
        edgecolor=COLOR_BLUE,
        linewidth=0.75,
        label=f"Fake planets ($n={len(fake_ratios)}$)",
        zorder=2,
    )

    # Real planet lines
    handles: list[Line2D | Patch] = [
        Patch(facecolor=COLOR_LIGHT_BLUE, edgecolor=COLOR_BLUE, label=f"Fake ($n={len(fake_ratios)}$)"),
        Line2D([0], [0], color="#777777", linestyle=":", linewidth=1.0, label="100%"),
    ]
    highlight = {"Venus": COLOR_RED, "Mars": COLOR_ORANGE}
    for _, row in real.iterrows():
        name = row["Name"]
        rc = row["Conj_Ratio"]
        if name in highlight:
            sigma = (rc - mean_fake) / std_fake
            ax.axvline(rc, color=highlight[name], linewidth=2.2, linestyle="-",
                       zorder=4)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=highlight[name],
                    linewidth=2.2,
                    label=f"{name[0]}: {rc:.1f}% ({sigma:.1f}$\\sigma$)",
                )
            )

    ax.axvline(100, color="#777777", linewidth=1.0, linestyle=":", zorder=1)

    n_sig = (fake["Conj_p"] < 0.05).sum()
    pct = n_sig / len(fake) * 100
    _style_box(ax, f"FPR = {n_sig}/{len(fake)} ({pct:.0f}%)\nexpected 5%", 0.04, 0.62)

    ax.set_xlim(70, 124)
    ax.set_ylim(0, 26)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_xlabel(RC_AXIS_LABEL)
    ax.set_ylabel("Count")
    ax.set_title("(a) Null planet test", loc="left", fontweight="normal")
    legend = ax.legend(
        handles=handles,
        frameon=True,
        loc="upper left",
        fontsize=6.8,
        handlelength=1.35,
        handletextpad=0.45,
        borderpad=0.28,
        labelspacing=0.18,
    )
    _style_legend(legend)
    ax.grid(axis="y", color="#e5e7e9", linewidth=0.8)


def plot_bootstrap_forest(ax: plt.Axes) -> None:
    boot = pd.read_csv(os.path.join(INPUT_DIR, "bootstrap_ci.csv"))

    labels = [BOOT_LABELS.get(label, label) for label in boot["Combo"].tolist()]
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
                    fmt="o",
                    markersize=6.5,
                    color=color,
                    ecolor=color,
                    elinewidth=1.8,
                    capsize=3.5,
                    capthick=1.4,
                    zorder=3,
                )

    ax.axvline(100, color="#777777", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.set_xlabel(RC_AXIS_LABEL)
    ax.set_title("(b) Bootstrap 95% CI", loc="left", fontweight="normal")
    ax.set_xlim(88, 142)
    ax.grid(axis="x", color="#e5e7e9", linewidth=0.8)


def _tidal_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min() - 0.25, x.max() + 0.25, 50)
    return x_fit, slope * x_fit + intercept


def _planet_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5.2,
            markerfacecolor=PLANET_COLORS[name],
            markeredgecolor="none",
            label=name,
        )
        for name in TIDAL_PLANET_ORDER
    ]


def plot_tidal_rc(ax: plt.Axes) -> None:
    tidal = pd.read_csv(os.path.join(INPUT_DIR, "tidal_correlation.csv"))

    x = tidal["Log_Tidal"].values
    y_rc = tidal["Conj_Ratio"].values

    from scipy.stats import spearmanr
    sp_rc, sp_rc_p = spearmanr(x, y_rc)

    for _, row in tidal.iterrows():
        c = PLANET_COLORS.get(row["Planet"], "gray")
        sig = row["Conj_p"] < 0.05
        ax.scatter(
            row["Log_Tidal"],
            row["Conj_Ratio"],
            s=58 if sig else 48,
            c=c,
            marker="o",
            edgecolors="none",
            linewidth=0,
            zorder=4,
        )

    x_fit, y_fit = _tidal_fit(x, y_rc)
    ax.plot(x_fit, y_fit, "--", color=COLOR_LIGHT_RED, linewidth=1.3, zorder=2)
    ax.axhline(100, color="#777777", linewidth=0.9, linestyle=":", zorder=1)
    _style_box(ax, f"$r_s={sp_rc:.2f}$\n$p={sp_rc_p:.3f}$", 0.05, 0.96, fontsize=7.8)

    ax.set_xlabel(TIDAL_AXIS_LABEL)
    ax.set_ylabel(RC_AXIS_LABEL)
    ax.set_title("(c) Tidal force vs $R_C$", loc="left", fontweight="normal")
    ax.set_xlim(x.min() - 0.25, x.max() + 0.25)
    ax.set_ylim(93.0, 124.5)
    ax.grid(color="#e5e7e9", linewidth=0.8)


def plot_tidal_asym(ax: plt.Axes) -> None:
    tidal = pd.read_csv(os.path.join(INPUT_DIR, "tidal_correlation.csv"))

    x = tidal["Log_Tidal"].values
    y_asym = tidal["Asym"].values

    from scipy.stats import spearmanr
    sp_asym, sp_asym_p = spearmanr(x, y_asym)

    for _, row in tidal.iterrows():
        c = PLANET_COLORS.get(row["Planet"], "gray")
        ax.scatter(
            row["Log_Tidal"],
            row["Asym"],
            s=54,
            c=c,
            marker="^",
            edgecolors="none",
            linewidth=0,
            zorder=4,
        )

    x_fit, y_fit = _tidal_fit(x, y_asym)
    ax.plot(x_fit, y_fit, "-.", color=COLOR_LIGHT_FIT_BLUE, linewidth=1.3, zorder=2)
    ax.axhline(0, color="#777777", linewidth=0.9, linestyle=":", zorder=1)
    _style_box(ax, f"$r_s={sp_asym:.2f}$\n$p={sp_asym_p:.3f}$", 0.05, 0.96, fontsize=7.8)

    ax.set_xlabel(TIDAL_AXIS_LABEL)
    ax.set_ylabel(ASYM_AXIS_LABEL)
    ax.set_title("(d) Tidal force vs Asym", loc="left", fontweight="normal")
    ax.set_xlim(x.min() - 0.25, x.max() + 0.25)
    ax.set_ylim(-5.5, 30.0)
    ax.grid(color="#e5e7e9", linewidth=0.8)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    apply_acta_style("double")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize_double(aspect=0.80),
        gridspec_kw={"wspace": 0.34, "hspace": 0.43},
    )
    fig.subplots_adjust(bottom=0.16, top=0.94)

    plot_null_test(axes[0, 0])
    plot_bootstrap_forest(axes[0, 1])
    plot_tidal_rc(axes[1, 0])
    plot_tidal_asym(axes[1, 1])

    legend = fig.legend(
        handles=_planet_legend_handles(),
        frameon=True,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=len(TIDAL_PLANET_ORDER),
        fontsize=6.8,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.9,
        borderpad=0.3,
    )
    _style_legend(legend)

    save_dual(fig, os.path.join(OUTPUT_DIR, f"{FIG09_FIG_BASENAME}.eps"))
    plt.close(fig)

    print("Robustness combined figure generated.")


if __name__ == "__main__":
    main()

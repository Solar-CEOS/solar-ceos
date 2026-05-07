#!/usr/bin/env python3
"""
03_fig08_solar_cycle_viz.py
===========================
Solar-cycle stability figure for sec5.

Inputs:
  - results/05_multidimensional/Fig08_solar_cycle_subset_summary.csv

Outputs:
  - results/05_multidimensional/Fig08_solar_cycle_stability.eps
  - results/05_multidimensional/Fig08_solar_cycle_stability.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style.figstyle import apply_acta_style, axis_unit, figsize_double, save_dual

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "05_multidimensional")
FIG08_FIG_BASENAME = "Fig08_solar_cycle_stability"
FIG08_SUMMARY_CSV = "Fig08_solar_cycle_subset_summary.csv"

CSV_PATH = os.path.join(OUTPUT_DIR, FIG08_SUMMARY_CSV)
FDR_CSV = os.path.join(OUTPUT_DIR, "fdr_audit", "fig08_solar_cycle_fdr.csv")

SUBSET_ORDER = ["best_single", "best_multi", "all_7", "jup_sat"]
SUBSET_DISPLAY = {
    "best_single": "Venus",
    "best_multi": "Venus+Mars",
    "all_7": "All 7 planets",
    "jup_sat": "Jupiter+Saturn",
}
SC_LABELS = ["SC21", "SC22", "SC23", "SC24"]

DATASET_DISPLAY = {"sf": "Flare", "sg": "Sunspot groups"}
ROLE_STYLE = {
    "best_single": {
        "color": "#E65100",
        "light": "#F8DCCB",
        "marker": "o",
        "linestyle": "-",
    },
    "best_multi": {
        "color": "#C62828",
        "light": "#F1CECE",
        "marker": "s",
        "linestyle": "-",
    },
    "all_7": {
        "color": "#555555",
        "light": "#DADADA",
        "marker": "^",
        "linestyle": "-",
    },
    "jup_sat": {
        "color": "#6A1B9A",
        "light": "#E3D1EC",
        "marker": "D",
        "linestyle": "-",
    },
}


def format_p(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#bdbdbd")
    frame.set_linewidth(0.5)
    frame.set_alpha(1.0)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # ── Load FDR q-values + staleness check ──
    if not os.path.exists(FDR_CSV):
        raise FileNotFoundError(
            f"Missing {FDR_CSV}; run 07_fdr_audit.py before plotting Fig08."
        )
    fdr_df = pd.read_csv(FDR_CSV)
    fdr_lookup: dict[tuple, float] = {}
    fdr_p_lookup: dict[tuple, float] = {}
    for _, row in fdr_df.iterrows():
        key = (row["Dataset"], row["Subset_Role"], row["SC"], int(row["Window"]))
        fdr_lookup[key] = float(row["Conj_q_dataset_window"])
        fdr_p_lookup[key] = float(row["Conj_p"])
    # Staleness check: raw Conj_p must match the source summary CSV.
    max_p_diff = 0.0
    for _, row in df.iterrows():
        key = (row["Dataset"], row["Subset_Role"], row["SC"], int(row["Window"]))
        if key in fdr_p_lookup:
            max_p_diff = max(max_p_diff, abs(fdr_p_lookup[key] - float(row["Conj_p"])))
    if max_p_diff > 1e-9:
        raise ValueError(
            f"Fig08 FDR table is stale: max raw-p mismatch = {max_p_diff:.3g}. "
            "Rerun 07_fdr_audit.py, then rerun this script."
        )
    print(f"[FDR] validated: {FDR_CSV}")

    fisher_q_lookup: dict[tuple[str, str], float] = {}
    for _, row in fdr_df.iterrows():
        key = (row["Dataset"], row["Subset_Role"])
        fisher_q_lookup.setdefault(key, float(row["Fisher_q_dataset_window"]))

    apply_acta_style("double")
    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize_double(aspect=0.50),
        sharey=True,
        gridspec_kw={"wspace": 0.16},
    )

    x = np.arange(len(SC_LABELS))
    bar_width = 0.13
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(SUBSET_ORDER))
    y_min, y_max = -30.0, 68.0
    legend_anchor_left = (0.00, 0.985)
    legend_anchor_right = (1.00, 0.985)
    note_anchor = (legend_anchor_left[0] + 0.015, legend_anchor_left[1] - 0.250)

    for ax_idx, dataset in enumerate(["sf", "sg"]):
        ax = axes[ax_idx]
        ax.axhline(0, color="#7f7f7f", linestyle="--", linewidth=0.9, zorder=1)
        ax.grid(axis="y", color="#e5e7e9", linewidth=0.7, zorder=0)

        legend_handles: list[Line2D] = []
        for role_idx, role in enumerate(SUBSET_ORDER):
            role_data = (
                df[(df["Dataset"] == dataset) & (df["Subset_Role"] == role)]
                .set_index("SC")
                .loc[SC_LABELS]
                .reset_index()
            )
            style = ROLE_STYLE[role]
            delta_rc = role_data["Conj_Ratio"].values - 100.0
            bar_x = x + offsets[role_idx]
            ax.bar(
                bar_x,
                delta_rc,
                bar_width * 0.92,
                color=style["light"],
                edgecolor=style["color"],
                linewidth=0.55,
                zorder=2,
            )
            ax.plot(
                x,
                role_data["Asym_Amp"].values,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.0,
                linewidth=1.45,
                zorder=4,
            )

            for sc_idx, (_, row) in enumerate(role_data.iterrows()):
                if row["Conj_p"] >= 0.05:
                    continue
                marker = r"$\star$"
                y_star = max(float(delta_rc[sc_idx]), 0.0) + 2.2
                ax.text(
                    bar_x[sc_idx],
                    y_star,
                    marker,
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    color=style["color"],
                    zorder=5,
                )

            fisher_p = float(role_data["Fisher_p"].iloc[0])
            fisher_q = fisher_q_lookup.get((dataset, role), np.nan)
            legend_label = (
                f"{SUBSET_DISPLAY[role]} "
                f"({format_p(fisher_p)}/{format_p(fisher_q)})"
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=1.45,
                    markersize=4.0,
                    label=legend_label,
                )
            )

        ax.set_title(
            f"({chr(ord('a') + ax_idx)}) {DATASET_DISPLAY[dataset]}",
            loc="left",
            fontsize=10.5,
            fontweight="normal",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(SC_LABELS)
        ax.set_xlabel("Solar cycle")
        ax.set_xlim(-0.55, len(SC_LABELS) - 0.45)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks([-20, 0, 20, 40, 60])
        if ax_idx == 0:
            ax.set_ylabel(axis_unit(r"\Delta R_{\mathrm{C}},\ \mathrm{Asym}", r"\%"))
        else:
            ax.tick_params(axis="y", labelleft=True)

        legend = ax.legend(
            handles=legend_handles,
            title="Fisher p/q",
            frameon=True,
            loc="upper left" if ax_idx == 0 else "upper right",
            bbox_to_anchor=legend_anchor_left if ax_idx == 0 else legend_anchor_right,
            fontsize=6.7,
            title_fontsize=7.0,
            handlelength=1.7,
            handletextpad=0.45,
            labelspacing=0.25,
            borderaxespad=0.25,
        )
        style_legend(legend)

    axes[0].text(
        note_anchor[0],
        note_anchor[1],
        (
            r"bars: $\Delta R_{\mathrm{C}}$; lines: $\mathrm{Asym}$"
            "\n"
            r"$\star$: raw $p<0.05$, FDR $q>0.05$"
        ),
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#bdbdbd"),
    )

    save_dual(fig, os.path.join(OUTPUT_DIR, f"{FIG08_FIG_BASENAME}.eps"))
    plt.close(fig)

    print("Solar cycle stability figure generated.")


if __name__ == "__main__":
    main()

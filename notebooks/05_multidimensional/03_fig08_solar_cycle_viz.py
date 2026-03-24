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
  - submission/figures_2026/Fig08_solar_cycle_stability.eps
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

COLOR_SF = "#c0392b"
COLOR_SG = "#1f618d"
LIGHT_SF = "#f5b7b1"
LIGHT_SG = "#d6eaf8"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # ── Load FDR q-values ──
    fdr_lookup: dict[tuple, float] = {}
    if os.path.exists(FDR_CSV):
        fdr_df = pd.read_csv(FDR_CSV)
        for _, row in fdr_df.iterrows():
            key = (row["Dataset"], row["Subset_Role"], row["SC"], int(row["Window"]))
            fdr_lookup[key] = float(row["Conj_q_dataset_window"])

    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5),
                             gridspec_kw={"hspace": 0.38, "wspace": 0.30})

    for col_idx, role in enumerate(SUBSET_ORDER):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]
        subset_label = SUBSET_DISPLAY[role]

        # --- Top row: R_C ---
        sf_data = df[(df["Dataset"] == "sf") & (df["Subset_Role"] == role)].sort_values("SC")
        sg_data = df[(df["Dataset"] == "sg") & (df["Subset_Role"] == role)].sort_values("SC")

        x = np.arange(len(SC_LABELS))
        w = 0.32

        bars_sf = ax_top.bar(x - w/2, sf_data["Conj_Ratio"].values, w,
                             color=LIGHT_SF, edgecolor=COLOR_SF, linewidth=0.9,
                             label="Flare", zorder=3)
        bars_sg = ax_top.bar(x + w/2, sg_data["Conj_Ratio"].values, w,
                             color=LIGHT_SG, edgecolor=COLOR_SG, linewidth=0.9,
                             label="Sunspot", zorder=3)

        # Mark significant p values with FDR status
        for i, (_, row) in enumerate(sf_data.iterrows()):
            if row["Conj_p"] < 0.05:
                fdr_key = ("sf", role, row["SC"], int(row["Window"]))
                q_val = fdr_lookup.get(fdr_key, np.nan)
                marker = "★" if (not np.isnan(q_val) and q_val < 0.05) else "☆"
                ax_top.text(i - w/2, row["Conj_Ratio"] + 1.5, marker,
                           ha="center", fontsize=9, color=COLOR_SF)

        ax_top.axhline(100, color="gray", linestyle="--", linewidth=1.0, zorder=2)
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(SC_LABELS, fontsize=9)
        ax_top.set_ylabel("$R_C$ (%)" if col_idx == 0 else "")
        ax_top.grid(axis="y", color="#e5e7e9", linewidth=0.8)

        # Fisher annotation
        fisher_sf = sf_data["Fisher_p"].iloc[0]
        fisher_sg = sg_data["Fisher_p"].iloc[0]
        fisher_text = f"Fisher $p$: {fisher_sf:.3f} / {fisher_sg:.3f}"
        ax_top.text(0.5, 0.02, fisher_text, transform=ax_top.transAxes,
                   ha="center", fontsize=7.5, color="#555555",
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                            edgecolor="#cccccc", alpha=0.9))

        panel_letter = chr(ord('a') + col_idx)
        ax_top.set_title(f"({panel_letter}) {subset_label}", fontsize=10.5)
        if col_idx == 0:
            ax_top.legend(frameon=False, loc="upper left", fontsize=8)
            ax_top.text(0.98, 0.98, "★ raw $p<.05$ & $q<.05$\n☆ raw $p<.05$ only",
                       transform=ax_top.transAxes, ha="right", va="top", fontsize=6.5,
                       color="#555555", bbox=dict(boxstyle="round,pad=0.2",
                       facecolor="white", edgecolor="#cccccc", alpha=0.9))

        # --- Bottom row: Asym ---
        bars_sf_a = ax_bot.bar(x - w/2, sf_data["Asym_Amp"].values, w,
                               color=LIGHT_SF, edgecolor=COLOR_SF, linewidth=0.9,
                               label="Flare", zorder=3)
        bars_sg_a = ax_bot.bar(x + w/2, sg_data["Asym_Amp"].values, w,
                               color=LIGHT_SG, edgecolor=COLOR_SG, linewidth=0.9,
                               label="Sunspot", zorder=3)

        ax_bot.axhline(0, color="gray", linestyle="--", linewidth=1.0, zorder=2)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(SC_LABELS, fontsize=9)
        ax_bot.set_ylabel("Asym (%)" if col_idx == 0 else "")
        ax_bot.grid(axis="y", color="#e5e7e9", linewidth=0.8)

        panel_letter_bot = chr(ord('e') + col_idx)
        ax_bot.set_title(f"({panel_letter_bot}) {subset_label} — Asym", fontsize=10.5)

    # Sync y-axes within each row
    rc_vals = df["Conj_Ratio"].values
    rc_min, rc_max = min(rc_vals.min(), 85), max(rc_vals.max(), 155)
    asym_vals = df["Asym_Amp"].values
    asym_lo = min(asym_vals.min(), -25)
    asym_hi = max(asym_vals.max(), 65)
    for col_idx in range(4):
        axes[0, col_idx].set_ylim(rc_min - 3, rc_max + 5)
        axes[1, col_idx].set_ylim(asym_lo - 3, asym_hi + 5)

    for ext in ["eps", "png"]:
        out = os.path.join(OUTPUT_DIR, f"{FIG08_FIG_BASENAME}.{ext}")
        plt.savefig(out, format=ext, dpi=300, bbox_inches="tight")

    fig_out = os.path.join(FIG_DIR, f"{FIG08_FIG_BASENAME}.eps")
    plt.savefig(fig_out, format="eps", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Solar cycle stability figure generated.")


if __name__ == "__main__":
    main()

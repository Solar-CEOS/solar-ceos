#!/usr/bin/env python3
"""
Plot Fig02 from the saved source workbook.

Fast plot-only entry — reads the precomputed statistics and renders the
5x4 sunspot/flare longitude figure without recomputing anything.

Source : results/03_coord_baseline/Fig02_Spot_Flare_Longitude_Source.xlsx
         (20 sheets; produced by 02_fig02_sg_sf_lon.py)
Output : results/03_coord_baseline/Fig02_Spot_Flare_Longitude.{eps,png}

Run 02_fig02_sg_sf_lon.py only when the underlying statistics need to be
regenerated (raw data ingest + Kuiper + permutation tests + FDR).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style.figstyle import apply_acta_style, figsize_double, save_dual

apply_acta_style("double")


SOURCE_NAME = "Fig02_Spot_Flare_Longitude_Source.xlsx"
FIG_NAME = "Fig02_Spot_Flare_Longitude.eps"


def resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [Path.cwd(), here.parents[2]]:
        if (candidate / "results" / "03_coord_baseline" / SOURCE_NAME).exists():
            return candidate
    raise FileNotFoundError(f"Could not locate results/03_coord_baseline/{SOURCE_NAME}")


# ---------------------------------------------------------------------------
# Layout: 5 rows x 4 columns
#   Rows pair Spot lifecycle stage (left block cols 0-1) with Flare class
#   (right block cols 2-3) by ordinal position for visual symmetry; no rate
#   ordering is implied.
#   Cols 0,1 : Spot in Carrington / Heliocentric Ecliptic longitude
#   Cols 2,3 : Flare in Carrington / Heliocentric Ecliptic longitude
# ---------------------------------------------------------------------------
SPOT_ROWS = [
    ("onset", "Onset SG"),
    ("diss", "Diss. SG"),
    ("dur", "Dur. SG"),
    ("daily", "Daily SG"),
    ("all", "All SG"),
]
FLARE_ROWS = [
    ("x_class", "X-Class"),
    ("m_class", "M-Class"),
    ("c_class", "C-Class"),
    ("b_class", "B-Class"),
    ("all_flares", "All Flares"),
]
COL_CFGS = [
    {"kind": "spot", "coord": "L_cr", "xlabel": r"Spot $L_{\mathrm{cr}}/(^\circ)$"},
    {"kind": "spot", "coord": "Lambda", "xlabel": r"Spot $\lambda/(^\circ)$"},
    {"kind": "flare", "coord": "L_cr", "xlabel": r"Flare $L_{\mathrm{cr}}/(^\circ)$"},
    {"kind": "flare", "coord": "Lambda", "xlabel": r"Flare $\lambda/(^\circ)$"},
]

# Pre-mixed colours: mix = alpha * RGB + (1-alpha) * 255  (EPS-safe replacement
# for the original alpha kwargs in 02_fig02_sg_sf_lon.py)
COL_BAR_POS_SIG = "#D5B46D"   # darkgoldenrod x 0.6 + white x 0.4
COL_BAR_NEG_SIG = "#669999"   # teal          x 0.6 + white x 0.4
COL_BAR_NONSIG = "#F6F0BA"    # khaki         x 0.6 + white x 0.4
COL_SCAN_LINE = "#C99412"     # saturated gold-brown; legible on non-sig bars
COL_SCATTER_POS_SIG = "crimson"
COL_SCATTER_NEG_SIG = "royalblue"
COL_SCATTER_NONSIG = "#D4D4D4"  # darkgray    x 0.5 + white x 0.5
COL_REF_RIGHT = "#ECD290"       # goldenrod   x 0.5 + white x 0.5
COL_REF_LEFT = "lightgray"      # gray        x 0.5 + white x 0.5

BAR_WIDTH = 10  # 10 deg bin step matches the source statistics

# Font-size overrides (relative to figstyle PT_DOUBLE = {axlabel:9, tick:8,
# legend:7.5, title:10, annot:7}). figstyle was tuned for 2x2 / 3x3 panel
# grids; this figure is 5x4 = 20 panels in a 7.2" x 6.12" double-column
# canvas. Internal Y-tick labels are hidden (all panels share Y range [0, 2],
# so duplicating them is redundant); only outermost columns carry tick
# labels. With horizontal pressure relieved, ticks/annotations sit closer
# to figstyle defaults; legend stays slightly compressed for the 5-col row.
FS_XLABEL = 8       # figstyle 9 -> 8 (single bottom row, x-label per col)
FS_TICK_X = 7       # figstyle 8 -> 7
FS_TICK_Y = 7       # figstyle 8 -> 7 (only on outer columns)
FS_PTEXT = 5.5      # compact 2-line annotation in the inter-row gutter
FS_LEGEND = 6.1     # compact 5-col bottom legend, 9 entries
FS_AXIS_TITLE = 8.5 # vertical shared Y-axis titles
FS_ROW_LABEL = 8.5


def sheet_name(kind: str, key: str, coord: str) -> str:
    """Match the writer convention in 02_fig02_sg_sf_lon.py L599."""
    return f"{kind.capitalize()}_{key}_{coord}"[:31]


def load_panel(xl: pd.ExcelFile, kind: str, key: str, coord: str) -> pd.DataFrame:
    df = pd.read_excel(xl, sheet_name=sheet_name(kind, key, coord))
    if kind == "flare":
        rename = {
            "Norm_Mean_Intensity": "Norm_Mean_Weight",
            "Sigma_Intensity": "Sigma_Weight",
            "Is_FDR_Sig_Intensity": "Is_FDR_Sig_Weight",
            "Scan_Norm_Intensity": "Scan_Norm_Weight",
        }
    else:
        rename = {
            "Norm_Mean_Area": "Norm_Mean_Weight",
            "Sigma_Area": "Sigma_Weight",
            "Is_FDR_Sig_Area": "Is_FDR_Sig_Weight",
            "Scan_Norm_Area": "Scan_Norm_Weight",
        }
    return df.rename(columns=rename)


def smart_p(p) -> str:
    if pd.isna(p):
        return "N/A"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def render_panel(ax, df: pd.DataFrame, col_idx: int) -> None:
    ax.set_xlim(0, 360)
    ax2 = ax.twinx()
    ax.patch.set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.grid(False)
    ax2.grid(False)

    ax2.axhline(1, color=COL_REF_RIGHT, ls="--", lw=1)
    sig_cnt = df["Is_FDR_Sig_Count"].fillna(False).astype(bool)
    bar_colors = [
        COL_BAR_POS_SIG if (s and v > 0) else COL_BAR_NEG_SIG if (s and v < 0) else COL_BAR_NONSIG
        for s, v in zip(sig_cnt, df["Sigma_Count"].fillna(0.0))
    ]
    ax2.bar(df["Bin_Center"], df["Norm_Count"], width=BAR_WIDTH, color=bar_colors)
    if "Scan_Norm_Count" in df.columns and df["Scan_Norm_Count"].notna().any():
        ax2.plot(df["Bin_Center"], df["Scan_Norm_Count"], color=COL_SCAN_LINE, ls="-.", lw=1.5)

    ax.axhline(1, color=COL_REF_LEFT, ls="--", lw=1)
    ax.plot(df["Bin_Center"], df["Norm_Mean_Weight"], "o",
            color=COL_SCATTER_NONSIG, ms=3)
    sig_w = df["Is_FDR_Sig_Weight"].fillna(False).astype(bool)
    sigma_w = df["Sigma_Weight"].fillna(0.0)
    pos = sig_w & (sigma_w > 0)
    neg = sig_w & (sigma_w <= 0)
    if pos.any():
        ax.plot(df.loc[pos, "Bin_Center"], df.loc[pos, "Norm_Mean_Weight"],
                "o", color=COL_SCATTER_POS_SIG, ms=3, zorder=10)
    if neg.any():
        ax.plot(df.loc[neg, "Bin_Center"], df.loc[neg, "Norm_Mean_Weight"],
                "o", color=COL_SCATTER_NEG_SIG, ms=3, zorder=10)

    ax.set_ylim(0, 2.0)
    ax2.set_ylim(0, 2.0)
    for axis_obj in (ax, ax2):
        axis_obj.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis_obj.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
        axis_obj.tick_params(axis="y", labelsize=FS_TICK_Y, pad=1)
        axis_obj.tick_params(axis="x", labelsize=FS_TICK_X)

    # Hide internal y-tick labels: with all panels on Y in [0, 2], internal
    # labels are redundant. Keep them only on the outermost columns:
    #   col 0  -> show LEFT  (weight axis;     Norm. Area title)
    #   col 1  -> hide both
    #   col 2  -> hide both  (Flare weight scale inferred from col 0)
    #   col 3  -> show RIGHT (shared count;    Norm. Count title)
    if col_idx == 0:
        ax2.tick_params(axis="y", which="both", labelright=False)
    elif col_idx in (1, 2):
        ax.tick_params(axis="y", which="both", labelleft=False)
        ax2.tick_params(axis="y", which="both", labelright=False)
    else:  # col_idx == 3
        ax.tick_params(axis="y", which="both", labelleft=False)

    if len(df):
        k_p = df["Global_Kuiper_P"].iloc[0]
        p_cnt = df["Global_Perm_P_Cnt_Cyc"].iloc[0]
        p_wgt = df["Global_Perm_P_Wgt_Cyc"].iloc[0]
    else:
        k_p = p_cnt = p_wgt = np.nan
    is_sig = (not pd.isna(p_cnt) and p_cnt < 0.05) or (not pd.isna(p_wgt) and p_wgt < 0.05)
    info_col = "darkred" if is_sig else "black"
    ax.text(
        0.98, 1.05,
        f"Kuiper P={smart_p(k_p)}\nCyc: C={smart_p(p_cnt)} W={smart_p(p_wgt)}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=FS_PTEXT, color=info_col,
    )


def add_figure_legend(fig) -> None:
    handles = [
        mpatches.Patch(color=COL_BAR_POS_SIG, label="Count sig +"),
        mpatches.Patch(color=COL_BAR_NEG_SIG, label="Count sig -"),
        mpatches.Patch(color=COL_BAR_NONSIG, label="Count non-sig"),
        mlines.Line2D([], [], color=COL_REF_RIGHT, ls="--", lw=1, label="Count base"),
        mlines.Line2D([], [], marker="o", color=COL_SCATTER_POS_SIG, lw=0, ms=4,
                      label=r"Weight sig $+$"),
        mlines.Line2D([], [], marker="o", color=COL_SCATTER_NEG_SIG, lw=0, ms=4,
                      label=r"Weight sig $-$"),
        mlines.Line2D([], [], marker="o", color=COL_SCATTER_NONSIG, lw=0, ms=4,
                      label="Weight non-sig"),
        mlines.Line2D([], [], color=COL_REF_LEFT, ls="--", lw=1, label="Weight base"),
        mlines.Line2D([], [], color=COL_SCAN_LINE, ls="-.", lw=1.5, label="Window scan"),
    ]
    # Keep the dense 9-entry legend outside the 20 data panels, but inside the
    # figure canvas. Acta prefers framed legends; use a very light frame so the
    # legend reads as a key rather than an extra panel.
    leg = fig.legend(
        handles=handles, loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5, fontsize=FS_LEGEND, frameon=True, facecolor="white",
        handlelength=1.6, handletextpad=0.4,
        columnspacing=0.9, borderpad=0.35,
    )
    leg.get_frame().set_edgecolor("#D6D6D6")
    leg.get_frame().set_linewidth(0.35)


def main() -> None:
    root = resolve_project_root()
    src = root / "results" / "03_coord_baseline" / SOURCE_NAME
    out = root / "results" / "03_coord_baseline" / FIG_NAME
    print(f"--- Reading {src.relative_to(root)}")
    xl = pd.ExcelFile(src)

    fig, axes = plt.subplots(
        5, 4, figsize=figsize_double(aspect=0.85),
        sharex="col", sharey=False,
    )
    plt.subplots_adjust(left=0.095, right=0.895, top=0.96, bottom=0.12,
                        hspace=0.35, wspace=0.23)

    for r, ((spot_key, spot_label), (flare_key, flare_label)) in enumerate(
            zip(SPOT_ROWS, FLARE_ROWS)):
        for c, cfg in enumerate(COL_CFGS):
            ax = axes[r, c]
            key = spot_key if cfg["kind"] == "spot" else flare_key
            try:
                df = load_panel(xl, cfg["kind"], key, cfg["coord"])
            except Exception as exc:
                print(f"   [skip] {cfg['kind']}/{key}/{cfg['coord']}: {exc}")
                ax.axis("off")
                continue
            render_panel(ax, df, c)
            if r == 4:
                ax.set_xlabel(cfg["xlabel"], fontsize=FS_XLABEL)

        axes[r, 0].text(-0.22, 0.5, spot_label,
                        transform=axes[r, 0].transAxes,
                        va="center", ha="center", rotation="vertical",
                        fontsize=FS_ROW_LABEL, fontweight="bold", color="black")
        axes[r, 3].text(1.27, 0.5, flare_label,
                        transform=axes[r, 3].transAxes,
                        va="center", ha="center", rotation="vertical",
                        fontsize=FS_ROW_LABEL, fontweight="bold", color="black")
        # Y-axis titles only on row 2 (vertical centre). Three titles for
        # three labelled axes (tick labels are hidden on internal cols):
        #   col 0  L: Norm. Area      (Spot block weight axis)
        #   col 2  L: Norm. Intensity (Flare block weight axis)
        #   col 3  R: Norm. Count     (shared right-axis scale)
        if r == 2:
            axes[r, 0].text(-0.34, 0.5, "Norm. Area",
                            transform=axes[r, 0].transAxes,
                            va="center", ha="center", rotation="vertical",
                            fontsize=FS_AXIS_TITLE, color="gray")
            axes[r, 2].text(-0.10, 0.5, "Norm. Intensity",
                            transform=axes[r, 2].transAxes,
                            va="center", ha="center", rotation="vertical",
                            fontsize=FS_AXIS_TITLE, color="gray")
            axes[r, 3].text(1.38, 0.5, "Norm. Count",
                            transform=axes[r, 3].transAxes,
                            va="center", ha="center", rotation="vertical",
                            fontsize=FS_AXIS_TITLE, color="darkgoldenrod")

    add_figure_legend(fig)
    save_dual(fig, str(out))
    print(f"--- Wrote {out.relative_to(root)} (+ .png)")


if __name__ == "__main__":
    main()

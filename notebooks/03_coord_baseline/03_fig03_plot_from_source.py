#!/usr/bin/env python3
"""Plot Fig03 from the saved source workbook.

Fast style-only entry point.  The raw-data compute+plot script
03_fig03_urian_wing.py is kept only for intentional source-workbook
regeneration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style.figstyle import apply_acta_style, figsize_double, save_dual, PT_DOUBLE


SOURCE_NAME = "Fig03_Uranian_Wing_Source.xlsx"
FIG_NAME = "Fig03_Uranian_Wing.eps"
X_LIM = (-105, 105)
X_TICKS = np.arange(-100, 101, 25)


def resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (Path.cwd(), here.parents[2]):
        if (candidate / "results" / "03_coord_baseline" / SOURCE_NAME).exists():
            return candidate
    raise FileNotFoundError(f"Could not locate results/03_coord_baseline/{SOURCE_NAME}")


def satellite_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("Sat_") and c.endswith("_Mean_SSN")]


# Two-group split reflects the only natural classification of the 18 Uranian
# regular satellites: 5 gravitationally relaxed (spherical) major moons vs.
# 13 small inner irregular moons. NAIF IDs 706-715 and 725-727 are interleaved
# in semi-major axis (a≈50,000-98,000 km); the 725+ jump is a discovery-era
# artifact (HST 1999/2003 vs Voyager 1986), not a physical boundary.
#
# Colour assignment: "few-N warm, many-N cool" data-viz convention, paired
# perceptually-uniform palettes for symmetric per-line differentiation.
# - Major (5):      YlOrRd[1:6] — yellow→orange→red, fully warm, 5 distinct hues
# - Inner minor (13): seaborn `crest` — teal-green→deep navy, fully cool, 13 distinct
# Both span hue + value simultaneously so each individual line is identifiable
# while the group still reads as one warm or cool ensemble. Plain `Oranges` and
# `Blues` colormaps were too narrow in hue to differentiate 5/13 lines.
_MAJOR_PALETTE = sns.color_palette("YlOrRd", n_colors=7).as_hex()[1:6]
_INNER_MINOR_PALETTE = sns.color_palette("crest", n_colors=13).as_hex()

SATELLITE_GROUPS = [
    ("Major",       [701, 702, 703, 704, 705],                                 _MAJOR_PALETTE,           1.10),
    ("Inner minor", list(range(706, 716)) + [725, 726, 727],                   _INNER_MINOR_PALETTE,     0.85),
]


def satellite_style(sat_id: int) -> tuple[str, float]:
    """Colour Uranian satellites by physical grouping, not by arbitrary order.

    Each group declares either a matplotlib colormap name (sampled across
    levels 0.48-0.88 for safe perceptual range) or an explicit list of hex
    colours (used directly). Lists give finer control when the colormap's
    natural range is too narrow for the number of lines.
    """
    for _, ids, palette, lw in SATELLITE_GROUPS:
        if sat_id in ids:
            idx = ids.index(sat_id)
            if isinstance(palette, str):
                cmap = plt.get_cmap(palette)
                levels = np.linspace(0.48, 0.88, len(ids))
                return to_hex(cmap(levels[idx])), lw
            return palette[idx], lw
    return "#4D4D4D", 0.9


def plot_satellite_panel(ax, curves: pd.DataFrame, stats: pd.DataFrame) -> None:
    x = curves["X_Axis"].to_numpy()
    sat_cols = satellite_columns(curves)

    for col in sat_cols:
        sat_id = int(col.split("_")[1])
        colour, lw = satellite_style(sat_id)
        ax.plot(x, curves[col], color=colour, lw=lw, label=f"Sat {sat_id}")

    p_cyc = stats["Cyclic_P"].dropna()
    p_shuf = stats["Shuffle_P"].dropna()
    n_total = len(stats)
    stats_text = (
        rf"$\bf{{Cyclic:}}$ $p$=[{p_cyc.min():.3f}, {p_cyc.max():.3f}], "
        rf"Sig: {(p_cyc < 0.05).sum()}/{n_total}"
        "\n"
        rf"$\bf{{Shuffle:}}$ $p$=[{p_shuf.min():.4f}, {p_shuf.max():.4f}], "
        rf"Sig: {(p_shuf < 0.05).sum()}/{n_total}"
    )

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        fontsize=6.0,
        handlelength=1.35,
        columnspacing=0.65,
        labelspacing=0.18,
        borderpad=0.28,
    )
    leg.get_frame().set_edgecolor("#D6D6D6")
    leg.get_frame().set_linewidth(0.35)

    ax.text(
        0.5,
        0.735,
        stats_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PT_DOUBLE["annot"] - 0.5,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#D6D6D6", linewidth=0.35),
    )
    ax.set_title("(a) Uranian satellite stacking", loc="left", fontweight="normal", pad=4)
    ax.set_xlabel(r"Normalized ecliptic longitude difference")
    ax.set_ylabel("Mean daily sunspot number")
    ax.set_xlim(*X_LIM)
    ax.set_xticks(X_TICKS)
    ax.set_ylim(18, 220)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", color="#E8E8E8", linewidth=0.6)


def mirror_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.array([]), np.array([])
    x = points[:, 0]
    y = points[:, 1]
    return np.concatenate([x, -x]), np.concatenate([y, y])


def plot_geometry_panel(ax, curves: pd.DataFrame, points: pd.DataFrame) -> None:
    x = curves["X_Axis"].to_numpy()
    ax.plot(x, curves["Y_Combined"], color="#4D4D4D", lw=1.5, label="Combined effect", zorder=3)
    ax.plot(x, curves["Y_Inclination"], color="#D85A5A", ls="--", lw=1.25, label="Inclination factor", zorder=2)
    ax.plot(x, curves["Y_Distance"], color="#1A4DCC", ls=":", lw=1.55, label="Distance factor", zorder=2)

    max_pts = points.loc[points["Type"] == "Max", ["X", "Y"]].to_numpy()
    min_pts = points.loc[points["Type"] == "Min", ["X", "Y"]].to_numpy()
    max_x, max_y = mirror_points(max_pts)
    min_x, min_y = mirror_points(min_pts)

    ax.scatter(
        max_x,
        max_y,
        c="#E31A1C",
        marker="*",
        s=62,
        edgecolors="none",
        zorder=5,
        label="Solar max",
    )
    ax.scatter(
        min_x,
        min_y,
        c="#4169E1",
        marker="s",
        s=14,
        edgecolors="none",
        zorder=5,
        label="Solar min",
    )

    line_props = dict(color="#B3B3B3", linestyle="--", linewidth=1.0, zorder=1)
    for xpos in (-100, 0, 100):
        ax.axvline(xpos, **line_props)

    text_props = dict(va="bottom", fontsize=PT_DOUBLE["annot"], fontweight="bold", color="#404040")
    ax.text(-101, 1.30, r"Solstice ($\parallel$)", ha="left", **text_props)
    ax.text(0, 1.30, r"Equinox ($\perp$)", ha="center", **text_props)
    ax.text(101, 1.30, r"Solstice ($\parallel$)", ha="right", **text_props)

    formula_text = (
        r"$\bf{Geometric\ factor}$" + "\n"
        r"$\hat{r}$: Sun-Uranus unit vector" + "\n"
        r"$\hat{p}$: Uranus pole unit vector" + "\n"
        r"$d$: normalized heliocentric distance" + "\n"
        r"$F_{\mathrm{geo}}=|\hat{r}\cdot\hat{p}|/d$"
    )
    ax.text(
        0.73,
        0.91,
        formula_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PT_DOUBLE["annot"] - 1.0,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#D6D6D6", linewidth=0.35),
        zorder=6,
    )

    handles = [
        mlines.Line2D([], [], color="#4D4D4D", lw=1.5, label="Combined effect"),
        mlines.Line2D([], [], color="#D85A5A", ls="--", lw=1.25, label="Inclination factor"),
        mlines.Line2D([], [], color="#1A4DCC", ls=":", lw=1.55, label="Distance factor"),
        mlines.Line2D([], [], marker="*", color="#E31A1C", lw=0, ms=6, label="Solar max"),
        mlines.Line2D([], [], marker="s", color="#4169E1", lw=0, ms=3.5, label="Solar min"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.17, 0.97),
        fontsize=6.6,
        handlelength=1.8,
        borderpad=0.35,
        labelspacing=0.25,
    )
    leg.get_frame().set_edgecolor("#D6D6D6")
    leg.get_frame().set_linewidth(0.35)

    ax.set_title("(b) Solar-cycle extremes vs. Uranus geometry", loc="left", fontweight="normal", pad=4)
    ax.set_xlabel(r"Normalized geometric factor $F_{\mathrm{geo}}$")
    ax.set_ylabel("Normalized SSN")
    ax.set_xlim(*X_LIM)
    ax.set_xticks(X_TICKS)
    ax.set_ylim(-0.05, 1.42)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", color="#C0C0C0", linewidth=0.6)


def main() -> None:
    root = resolve_project_root()
    out_dir = root / "results" / "03_coord_baseline"
    source_path = out_dir / SOURCE_NAME
    out_path = out_dir / FIG_NAME

    print(f"--- Reading {source_path.relative_to(root)}")
    sat_curves = pd.read_excel(source_path, sheet_name="Satellites_Curves")
    sat_stats = pd.read_excel(source_path, sheet_name="Satellites_Stats")
    uranus_curves = pd.read_excel(source_path, sheet_name="Uranus_Curves")
    uranus_points = pd.read_excel(source_path, sheet_name="Uranus_Points")

    apply_acta_style("double")
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=figsize_double(aspect=0.82),
        gridspec_kw={"height_ratios": [1.03, 1.0]},
    )

    plot_satellite_panel(ax_top, sat_curves, sat_stats)
    plot_geometry_panel(ax_bottom, uranus_curves, uranus_points)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.965, bottom=0.085, hspace=0.38)

    eps, png = save_dual(fig, out_path)
    plt.close(fig)
    print(f"--- Wrote {eps.relative_to(root)} (+ {png.name})")


if __name__ == "__main__":
    main()

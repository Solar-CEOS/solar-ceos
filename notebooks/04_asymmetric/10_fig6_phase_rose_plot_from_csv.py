#!/usr/bin/env python3
"""
Plot Fig06 from the saved phase-profile CSV.

This is the fast, style-only entry point. Run 10_fig6_phase_rose.ipynb only
when Fig06_phase_rose.csv needs to be regenerated from flare and ephemeris data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style.figstyle import apply_acta_style, figsize_double, planet_color, save_dual


PANELS = [
    {"planet": "Venus", "label": "(a) Venus"},
    {"planet": "Mars", "label": "(b) Mars"},
    {"planet": "Jupiter", "label": "(c) Jupiter"},
]

COLOR_LO = "#E5E9EC"  # was #CFD8DC@0.55 → flattened against white for EPS
COLOR_LO_EDGE = "#90A4AE"
COLOR_REF = "#9E9E9E"
COLOR_CONJ_TXT = "#1B5E20"
COLOR_CONJ_BG = "#F4FAF4"  # was #E8F5E9@0.5 → flattened against white for EPS
SIG_EDGE_WIDTH = 1.8
NORMAL_EDGE_W = 0.3
SIG_THRESHOLD = 0.05
BIN_WIDTH_DEG = 10
CONJ_BAND_DEG = 15
R_LIM = (80, 120)
R_GRID_STEP = 5
R_LABEL_STEP = 10
R_LABEL_POSITION_DEG = 315
PANEL_NOTE_POS = (0.5, -0.27)
FIG_DPI = 300
PROFILE_NAME = "Fig06_phase_rose.csv"
FIG_NAME = "Fig06_phase_rose.eps"
RADIAL_LABEL = r"$R_{\mathrm{C}}/\%$"


def format_p(value: float) -> str:
    if value < 0.001:
        return r"$p_{\mathrm{raw}} < 0.001$"
    return rf"$p_{{\mathrm{{raw}}}}={value:.3f}$"


def resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [Path.cwd(), here.parents[2]]:
        if (candidate / "results" / "04_asymmetric" / PROFILE_NAME).exists():
            return candidate
    raise FileNotFoundError(f"Could not locate results/04_asymmetric/{PROFILE_NAME}")


def load_fdr_rose(root: Path, df_profile: pd.DataFrame) -> pd.DataFrame:
    fdr_rose_path = root / "results" / "05_multidimensional" / "fdr_audit" / "fig06_phase_rose_fdr.csv"
    if not fdr_rose_path.exists():
        raise FileNotFoundError("Missing fig06_phase_rose_fdr.csv; rerun 07_fdr_audit.py before plotting Fig06.")

    fdr = pd.read_csv(fdr_rose_path)
    required = {"Planet", "Angle_Center", "p_val", "q_planet"}
    missing = sorted(required - set(fdr.columns))
    if missing:
        raise ValueError(f"Fig06 FDR table missing columns: {missing}")

    left = df_profile[["Planet", "Angle_Center", "p_val"]].copy()
    right = fdr[["Planet", "Angle_Center", "p_val", "q_planet"]].copy()
    left["Angle_Key"] = left["Angle_Center"].round(2)
    right["Angle_Key"] = right["Angle_Center"].round(2)
    merged = left.merge(
        right[["Planet", "Angle_Key", "p_val", "q_planet"]],
        on=["Planet", "Angle_Key"],
        how="left",
        suffixes=("", "_fdr"),
    )
    if merged["q_planet"].isna().any():
        raise ValueError("Fig06 FDR table does not cover all current Planet/Angle bins; rerun 07_fdr_audit.py.")

    max_p_diff = float(np.max(np.abs(merged["p_val"] - merged["p_val_fdr"])))
    if max_p_diff > 1e-9:
        raise ValueError(
            f"Fig06 FDR table is stale: max raw-p mismatch = {max_p_diff:.3g}. "
            "Rerun 07_fdr_audit.py, then rerun this plot script."
        )
    print(f"[FDR] validated: {fdr_rose_path}")
    return fdr


def draw_rose_panel(ax, df_planet: pd.DataFrame, cfg: dict, q_lookup: dict[float, float], min_q: float) -> None:
    data = df_planet.sort_values("Angle_Center").copy()
    color_hi = planet_color(cfg["planet"])

    angles_deg = data["Angle_Center"].values.copy()
    angles_deg_signed = angles_deg.copy()
    angles_deg[angles_deg < 0] += 360
    theta = np.deg2rad(angles_deg)
    ratio = data["Ratio"].values
    p_vals = data["p_val"].values
    q_vals = np.array([q_lookup.get(round(a, 2), np.nan) for a in angles_deg_signed])
    if np.isnan(q_vals).any():
        raise ValueError(f"Missing q_planet values for {cfg['planet']} bins")

    width = np.deg2rad(BIN_WIDTH_DEG)

    bg_theta = np.linspace(-np.deg2rad(CONJ_BAND_DEG), np.deg2rad(CONJ_BAND_DEG), 60)
    ax.fill_between(bg_theta, *R_LIM, color=COLOR_CONJ_BG, zorder=0)

    ref_theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ref_theta, [100] * len(ref_theta), color=COLOR_REF, linewidth=1.0, linestyle="--", zorder=1)

    for t, r, _p, q in zip(theta, ratio, p_vals, q_vals):
        is_above = r >= 100
        color = color_hi if is_above else COLOR_LO
        edge_color = color_hi if is_above else COLOR_LO_EDGE
        is_sig = (not np.isnan(q)) and (q < SIG_THRESHOLD)
        lw = SIG_EDGE_WIDTH if is_sig else NORMAL_EDGE_W
        if is_sig and is_above:
            edge_color = "#000000"

        ax.bar(
            t,
            r - 100,
            width=width,
            bottom=100,
            color=color,
            edgecolor=edge_color,
            linewidth=lw,
            zorder=3 if is_sig else 2,
        )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    r_lo, r_hi = R_LIM
    ax.set_ylim(r_lo, r_hi)

    conj_mask = np.abs(angles_deg_signed) <= CONJ_BAND_DEG
    if conj_mask.sum() > 0:
        conj_data = data.loc[conj_mask]
        peak_row = conj_data.loc[conj_data["Ratio"].idxmax()]
        peak_angle = peak_row["Angle_Center"]
        peak_ratio = peak_row["Ratio"]
        peak_p = peak_row["p_val"]
        peak_q = q_lookup.get(round(peak_angle, 2), float("nan"))

        if np.isnan(peak_q):
            raise ValueError(f"Missing q_planet for {cfg['planet']} peak bin {peak_angle}")

        peak_text = (
            f"Conj. max: $R_{{\\mathrm{{C}}}}={peak_ratio:.1f}\\%$ "
            f"({peak_angle:+.0f}$^\\circ$), {format_p(peak_p)}"
        )
        ax.text(
            *PANEL_NOTE_POS,
            f"{peak_text}\n36-bin BH-FDR: $q_{{\\min}}={min_q:.2f}$",
            transform=ax.transAxes,
            fontsize=7.0,
            color="#555555",
            ha="center",
            va="center",
            linespacing=1.35,
            clip_on=False,
        )

    yticks = list(range(r_lo, r_hi + R_GRID_STEP, R_GRID_STEP))
    yticklabels = [f"{t}" if t > r_lo and t % R_LABEL_STEP == 0 else "" for t in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=6.5, color="#757575")
    ax.set_rlabel_position(R_LABEL_POSITION_DEG)
    ax.yaxis.set_tick_params(pad=1)

    angle_labels_deg = list(range(0, 360, 30))
    ax.set_xticks(np.deg2rad(angle_labels_deg))
    labels = []
    for a in angle_labels_deg:
        if a == 0:
            labels.append("")
        elif a == 180:
            labels.append(r"$180^\circ$")
        elif a <= 180:
            labels.append(fr"${a}^\circ$")
        else:
            labels.append(fr"${a - 360}^\circ$")
    ax.set_xticklabels(labels, fontsize=7, color="#546E7A")

    ax.grid(color="#E5E5E5", linewidth=0.5)
    ax.text(
        np.deg2rad(R_LABEL_POSITION_DEG),
        r_hi + 7.0,
        RADIAL_LABEL,
        fontsize=6.8,
        color="#616161",
        ha="center",
        va="bottom",
        clip_on=False,
    )
    ax.set_title(cfg["label"], loc="left", fontsize=11.5, fontweight="normal", pad=18, color="#212121")
    ax.annotate(
        "Conj (0$^\\circ$)",
        xy=(0, r_hi + 1.5),
        fontsize=8,
        ha="center",
        va="bottom",
        color=COLOR_CONJ_TXT,
        fontweight="bold",
        annotation_clip=False,
    )


def plot_from_profile(root: Path, df: pd.DataFrame) -> Path:
    fdr_rose = load_fdr_rose(root, df)
    apply_acta_style("double")
    fig = plt.figure(figsize=figsize_double(aspect=0.53), dpi=FIG_DPI)
    gs = GridSpec(1, 3, figure=fig, left=0.04, right=0.96, bottom=0.30, top=0.88, wspace=0.34)

    for i, cfg in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i], projection="polar")
        df_p = df[df["Planet"] == cfg["planet"]]
        if df_p.empty:
            raise ValueError(f"Missing {cfg['planet']} profile rows")

        planet_fdr = fdr_rose[fdr_rose["Planet"] == cfg["planet"]]
        q_lookup = dict(zip(planet_fdr["Angle_Center"].round(2), planet_fdr["q_planet"]))
        min_q = planet_fdr["q_planet"].min()
        draw_rose_panel(ax, df_p, cfg, q_lookup=q_lookup, min_q=min_q)

    out = root / "results" / "04_asymmetric" / FIG_NAME
    save_dual(fig, out)
    plt.close(fig)
    return out


def main() -> None:
    root = resolve_project_root()
    profile_path = root / "results" / "04_asymmetric" / PROFILE_NAME
    df = pd.read_csv(profile_path)
    required = {"Planet", "Angle_Center", "Ratio", "Z_score", "p_val"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Fig06 profile missing columns: {missing}")
    out = plot_from_profile(root, df)
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    main()

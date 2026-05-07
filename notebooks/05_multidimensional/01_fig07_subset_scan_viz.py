#!/usr/bin/env python3
"""
01_fig07_subset_scan_viz.py
===========================
Formal subset-scan and multi-planet summary for sec5.

Inputs:
  - results/04_asymmetric/sf/sf_subset_scan_no_earth.csv
  - results/04_asymmetric/sg/sg_subset_scan_no_earth.csv

Outputs:
  - results/05_multidimensional/Fig07_subset_scan_combined.eps
  - results/05_multidimensional/Fig07_subset_scan_combined.png
  - results/05_multidimensional/Fig07_subset_scan_distribution_summary.csv
  - results/05_multidimensional/Fig07_subset_scan_nested_path.csv
  - results/05_multidimensional/Fig07_subset_scan_window_sensitivity.csv
  - results/05_multidimensional/Fig07_subset_scan_multi_planet_summary.csv
  - results/05_multidimensional/Fig07_subset_scan_summary.csv

Usage:
  python notebooks/05_multidimensional/01_fig07_subset_scan_viz.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style.figstyle import apply_acta_style, axis_unit, figsize_double, save_dual


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "05_multidimensional")
FIG07_FIG_BASENAME = "Fig07_subset_scan_combined"
FIG07_SUMMARY_CSV = "Fig07_subset_scan_summary.csv"
FIG07_TABLE_NAMES = {
    "distribution": "Fig07_subset_scan_distribution_summary.csv",
    "nested_path": "Fig07_subset_scan_nested_path.csv",
    "window_sensitivity": "Fig07_subset_scan_window_sensitivity.csv",
    "multi_planet_summary": "Fig07_subset_scan_multi_planet_summary.csv",
    "multi_planet_detail": "Fig07_subset_scan_multi_planet_detail.csv",
}

SF_CSV = os.path.join(PROJECT_ROOT, "results", "04_asymmetric", "sf", "sf_subset_scan_no_earth.csv")
SG_CSV = os.path.join(PROJECT_ROOT, "results", "04_asymmetric", "sg", "sg_subset_scan_no_earth.csv")
FDR_CSV = os.path.join(PROJECT_ROOT, "results", "05_multidimensional", "fdr_audit", "subset_scan_fdr.csv")

REF_WINDOW = 2
RC_YLIM = (94.0, 124.0)
RC_TICKS = [95, 100, 105, 110, 115, 120]
PLANET_ORDER = ["Mer", "Ven", "Mar", "Jup", "Sat", "Ura", "Nep"]
PLANET_FULL_NAMES = {
    "Mer": "Mercury", "Ven": "Venus", "Mar": "Mars",
    "Jup": "Jupiter", "Sat": "Saturn", "Ura": "Uranus", "Nep": "Neptune",
}
DATASET_LABELS = {"sf": "Flare", "sg": "SG"}
SUBSET_ROLE_LABELS = {
    "best_single": "Best single",
    "best_multi": "Best multi",
    "all_7": "All 7 planets",
}


def canonical_label(members: Iterable[str]) -> str:
    member_set = set(members)
    return "+".join([planet for planet in PLANET_ORDER if planet in member_set])


def expand_label(label: str) -> str:
    return "+".join(PLANET_FULL_NAMES.get(p, p) for p in label.split("+"))


def add_legend(ax: plt.Axes, **kwargs):
    defaults = {
        "frameon": True,
        "borderpad": 0.24,
        "handlelength": 1.2,
        "handletextpad": 0.35,
        "labelspacing": 0.25,
        "borderaxespad": 0.28,
    }
    defaults.update(kwargs)
    legend = ax.legend(**defaults)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#bdbdbd")
    frame.set_linewidth(0.5)
    frame.set_alpha(1.0)
    return legend


def compact_axis(ax: plt.Axes, *, tick_size: float = 7.0, label_size: float = 8.0) -> None:
    ax.tick_params(axis="both", which="major", labelsize=tick_size, pad=1.0)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.title.set_size(9.0)


def annotate_nested_key_nodes(ax: plt.Axes, nested_path: pd.DataFrame) -> None:
    """Label every nested-curve point with the planet added at that step.

    Middle steps have closely-spaced R_C values, so labels alternate above/below
    the line and carry a thin white bbox to stay legible over the gray excess
    bars. The bbox is solid white (alpha=1.0) — EPS-safe.
    """
    sorted_steps = nested_path.sort_values("Step")
    prev_members: set[str] = set()
    label_bbox = {
        "boxstyle": "round,pad=0.15",
        "facecolor": "white",
        "edgecolor": "none",
    }
    for _, row in sorted_steps.iterrows():
        step = int(row["Step"])
        members = set(str(row["Subset_Label"]).split("+"))
        added = members - prev_members
        prev_members = members
        if not added:
            continue
        planet = next(iter(added))
        full = PLANET_FULL_NAMES.get(planet, planet)
        label = full if step == 1 else f"+{full}"
        ratio = float(row["Conj_Ratio"])
        # User-specified L/R pattern: 1R 2R 3L 4R 5L 6L 7L plus per-step
        # overrides for steps 3 (above-left to clear the curve) and 4
        # (below-right to clear the tall step-4 bar top). R = above-right,
        # L = below-left.
        side_pattern = {1: "R", 2: "R", 3: "L", 4: "R", 5: "L", 6: "L", 7: "L"}
        side = side_pattern.get(step, "R")
        if side == "R":
            xytext, ha, va = (3, 5), "left", "bottom"
        else:
            xytext, ha, va = (-3, -5), "right", "top"
        if step == 3:  # +Jupiter: 翻转 va,文本整体落到曲线下方
            xytext, ha, va = (-3, -3), "right", "top"
        elif step == 4:  # +Mercury: 翻转 va 落到曲线上方,y 继续收小拉远柱顶
            xytext, ha, va = (6, -1), "left", "bottom"
        elif step in (5, 6, 7):  # +Saturn/+Uranus/+Neptune: 整体右移,松开柱子左边缘
            xytext, ha, va = (0, -5), "right", "top"
        ax.annotate(
            label,
            xy=(step, ratio),
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=6.5,
            color="#555555",
            bbox=label_bbox,
            zorder=6,
        )


def load_subset_data(csv_path: str, dataset_key: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    df = df[df["Has_Earth"] == False].copy()
    df["Dataset"] = dataset_key
    df["Dataset_Label"] = DATASET_LABELS[dataset_key]
    df["Conj_Excess"] = df["Conj_k_obs"] - df["Conj_k_exp"]
    df["Opp_Excess"] = df["Opp_k_obs"] - df["Opp_k_exp"]
    df["Members"] = df["Label"].str.split("+")
    return df


def build_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, dataset_label, window, n_planets), grp in df.groupby(
        ["Dataset", "Dataset_Label", "Window", "N_Planets"]
    ):
        positive = int((grp["Conj_Ratio"] > 100).sum())
        asym_positive = int((grp["Asym_Amp"] > 0).sum())
        rows.append(
            {
                "Section": "distribution",
                "Dataset": dataset,
                "Dataset_Label": dataset_label,
                "Window": window,
                "N_Planets": n_planets,
                "N_Subsets": len(grp),
                "Conj_Ratio_Mean": round(grp["Conj_Ratio"].mean(), 2),
                "Conj_Ratio_Median": round(grp["Conj_Ratio"].median(), 2),
                "Conj_Positive_Count": positive,
                "Conj_Positive_Pct": round(positive / len(grp) * 100, 1),
                "Asym_Mean": round(grp["Asym_Amp"].mean(), 2),
                "Asym_Median": round(grp["Asym_Amp"].median(), 2),
                "Asym_Positive_Count": asym_positive,
                "Asym_Positive_Pct": round(asym_positive / len(grp) * 100, 1),
            }
        )
    return pd.DataFrame(rows).sort_values(["Dataset", "Window", "N_Planets"]).reset_index(drop=True)


def build_nested_path(df_ref: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    singles = (
        df_ref[df_ref["N_Planets"] == 1][["Label", "Conj_Ratio", "Conj_p"]]
        .sort_values(["Conj_Ratio", "Conj_p"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranking = singles["Label"].tolist()

    rows = []
    selected = []
    for step, planet in enumerate(ranking, start=1):
        selected.append(planet)
        label = canonical_label(selected)
        row = df_ref[df_ref["Label"] == label]
        if len(row) != 1:
            raise ValueError(f"Missing nested subset row for label={label}")
        record = row.iloc[0]
        rows.append(
            {
                "Section": "nested_path",
                "Step": step,
                "Subset_Label": label,
                "N_Planets": int(record["N_Planets"]),
                "Conj_k_obs": int(record["Conj_k_obs"]),
                "Conj_k_exp": float(record["Conj_k_exp"]),
                "Conj_Excess": round(float(record["Conj_Excess"]), 2),
                "Conj_Ratio": float(record["Conj_Ratio"]),
                "Conj_p": float(record["Conj_p"]),
                "Opp_Ratio": float(record["Opp_Ratio"]),
                "Opp_p": float(record["Opp_p"]),
                "Asym_Amp": float(record["Asym_Amp"]),
            }
        )

    return pd.DataFrame(rows), ranking


def build_multi_planet_summary(df_ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    single_map = (
        df_ref[df_ref["N_Planets"] == 1][["Label", "Conj_Ratio"]]
        .set_index("Label")["Conj_Ratio"]
        .to_dict()
    )

    detail_rows = []
    for _, row in df_ref[df_ref["N_Planets"] >= 2].iterrows():
        best_single = max(single_map[m] for m in row["Members"])
        detail_rows.append(
            {
                "Section": "multi_planet_detail",
                "Subset_Label": row["Label"],
                "N_Planets": int(row["N_Planets"]),
                "Conj_Ratio": float(row["Conj_Ratio"]),
                "Best_Constituent_Single_Ratio": float(best_single),
                "Delta_vs_Best_Single": round(float(row["Conj_Ratio"] - best_single), 2),
                "Conj_p": float(row["Conj_p"]),
                "Conj_Excess": round(float(row["Conj_Excess"]), 2),
            }
        )
    detail = pd.DataFrame(detail_rows).sort_values(["N_Planets", "Delta_vs_Best_Single", "Subset_Label"]).reset_index(drop=True)

    summary_rows = []
    for n_planets, grp in detail.groupby("N_Planets"):
        summary_rows.append(
            {
                "Section": "multi_planet_summary",
                "N_Planets": n_planets,
                "N_Subsets": len(grp),
                "N_Exceed_Best_Single": int((grp["Delta_vs_Best_Single"] > 0).sum()),
                "Median_Delta_vs_Best_Single": round(grp["Delta_vs_Best_Single"].median(), 2),
                "Min_Delta_vs_Best_Single": round(grp["Delta_vs_Best_Single"].min(), 2),
                "Max_Delta_vs_Best_Single": round(grp["Delta_vs_Best_Single"].max(), 2),
            }
        )

    overall = {
        "Section": "multi_planet_summary",
        "N_Planets": "All",
        "N_Subsets": len(detail),
        "N_Exceed_Best_Single": int((detail["Delta_vs_Best_Single"] > 0).sum()),
        "Median_Delta_vs_Best_Single": round(detail["Delta_vs_Best_Single"].median(), 2),
        "Min_Delta_vs_Best_Single": round(detail["Delta_vs_Best_Single"].min(), 2),
        "Max_Delta_vs_Best_Single": round(detail["Delta_vs_Best_Single"].max(), 2),
    }
    summary = pd.concat([pd.DataFrame(summary_rows), pd.DataFrame([overall])], ignore_index=True)
    return detail, summary


def build_window_sensitivity(
    df_all: pd.DataFrame,
    best_single_label: str,
    best_multi_label: str,
) -> pd.DataFrame:
    full_label = canonical_label(PLANET_ORDER)
    label_roles = {
        best_single_label: "best_single",
        best_multi_label: "best_multi",
        full_label: "all_7",
    }

    rows = []
    for label, role in label_roles.items():
        grp = df_all[df_all["Label"] == label].sort_values(["Dataset", "Window"])
        for _, record in grp.iterrows():
            rows.append(
                {
                    "Section": "window_sensitivity",
                    "Dataset": record["Dataset"],
                    "Dataset_Label": record["Dataset_Label"],
                    "Subset_Role": role,
                    "Subset_Label": label,
                    "Window": int(record["Window"]),
                    "N_Planets": int(record["N_Planets"]),
                    "Conj_Ratio": float(record["Conj_Ratio"]),
                    "Conj_p": float(record["Conj_p"]),
                    "Conj_Excess": round(float(record["Conj_Excess"]), 2),
                }
            )
    return pd.DataFrame(rows)


def plot_distribution_panel(ax: plt.Axes, df_ref: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    colors = {"sf": "#c0392b", "sg": "#1f618d"}
    light_colors = {"sf": "#f5b7b1", "sg": "#d6eaf8"}
    point_colors = {"sf": "#d98880", "sg": "#7fb3d5"}
    offsets = {"sf": -0.12, "sg": 0.12}

    for dataset in ["sf", "sg"]:
        subset = df_ref[df_ref["Dataset"] == dataset]
        x = subset["N_Planets"].to_numpy()
        y = subset["Conj_Ratio"].to_numpy()
        jitter = rng.uniform(-0.08, 0.08, size=len(subset))
        ax.scatter(
            x + offsets[dataset] + jitter,
            y,
            s=12,
            color=point_colors[dataset],
            label=DATASET_LABELS[dataset],
            zorder=3,
        )

        for n_planets in range(1, 8):
            values = subset.loc[subset["N_Planets"] == n_planets, "Conj_Ratio"].to_numpy()
            if len(values) == 0:
                continue
            bp = ax.boxplot(
                [values],
                positions=[n_planets + offsets[dataset]],
                widths=0.18,
                patch_artist=True,
                showfliers=False,
                zorder=4,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(light_colors[dataset])
                patch.set_edgecolor(colors[dataset])
                patch.set_linewidth(0.7)
            for element in ["whiskers", "caps", "medians"]:
                for line in bp[element]:
                    line.set_color(colors[dataset])
                    line.set_linewidth(0.7)

    ax.axhline(100, color="gray", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xlim(0.4, 7.6)
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels([str(i) for i in range(1, 8)])
    ax.set_xlabel("Planets in subset\n($N_P$)")
    ax.set_ylabel(axis_unit(r"R_{\mathrm{C}}", r"\%"))
    ax.set_ylim(*RC_YLIM)
    ax.set_yticks(RC_TICKS)
    ax.set_title("(a) $R_C$ distribution", fontweight="normal")
    add_legend(ax, loc="upper right", fontsize=7)
    ax.grid(axis="y", color="#e5e7e9", linewidth=0.8)
    compact_axis(ax)


def plot_asym_panel(ax: plt.Axes, df_ref: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    colors = {"sf": "#c0392b", "sg": "#1f618d"}
    light_colors = {"sf": "#f5b7b1", "sg": "#d6eaf8"}
    point_colors = {"sf": "#d98880", "sg": "#7fb3d5"}
    offsets = {"sf": -0.12, "sg": 0.12}

    for dataset in ["sf", "sg"]:
        subset = df_ref[df_ref["Dataset"] == dataset]
        x = subset["N_Planets"].to_numpy()
        y = subset["Asym_Amp"].to_numpy()
        jitter = rng.uniform(-0.08, 0.08, size=len(subset))
        ax.scatter(
            x + offsets[dataset] + jitter,
            y,
            s=12,
            color=point_colors[dataset],
            label=DATASET_LABELS[dataset],
            zorder=3,
        )

        for n_planets in range(1, 8):
            values = subset.loc[subset["N_Planets"] == n_planets, "Asym_Amp"].to_numpy()
            if len(values) == 0:
                continue
            bp = ax.boxplot(
                [values],
                positions=[n_planets + offsets[dataset]],
                widths=0.18,
                patch_artist=True,
                showfliers=False,
                zorder=4,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(light_colors[dataset])
                patch.set_edgecolor(colors[dataset])
                patch.set_linewidth(0.7)
            for element in ["whiskers", "caps", "medians"]:
                for line in bp[element]:
                    line.set_color(colors[dataset])
                    line.set_linewidth(0.7)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xlim(0.4, 7.6)
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels([str(i) for i in range(1, 8)])
    ax.set_xlabel("Planets in subset\n($N_P$)")
    ax.set_ylabel(axis_unit(r"\mathrm{Asym}", r"\%"))
    ax.set_title("(b) Asym distribution", fontweight="normal")
    add_legend(ax, loc="upper right", fontsize=7)
    ax.grid(axis="y", color="#e5e7e9", linewidth=0.8)
    compact_axis(ax)


def plot_nested_panel(ax: plt.Axes, nested_path: pd.DataFrame) -> None:
    x = nested_path["Step"].to_numpy()
    excess = nested_path["Conj_Excess"].to_numpy()
    ratio = nested_path["Conj_Ratio"].to_numpy()

    rc_min, rc_max = RC_YLIM
    count_min = 0.0
    count_max = float(np.ceil(float(excess.max()) * 1.15 / 50.0) * 50.0)

    def count_to_ratio(count):
        return rc_min + (np.asarray(count) - count_min) * (rc_max - rc_min) / (count_max - count_min)

    def ratio_to_count(ratio_value):
        return count_min + (np.asarray(ratio_value) - rc_min) * (count_max - count_min) / (rc_max - rc_min)

    bar_bottom = float(count_to_ratio(0.0))
    bar_top = count_to_ratio(excess)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e5e7e9", linewidth=0.8, zorder=0)
    ax.bar(
        x,
        bar_top - bar_bottom,
        bottom=bar_bottom,
        width=0.65,
        color="#d5dbdb",
        edgecolor="#7f8c8d",
        linewidth=0.8,
        zorder=2,
    )
    ax_count = ax.secondary_yaxis("right", functions=(ratio_to_count, count_to_ratio))
    ax_count.set_ylabel(r"$N_{\mathrm{obs}}-N_{\mathrm{CTS}}$", labelpad=1.0)
    ax_count.yaxis.label.set_size(7.0)
    ax_count.tick_params(axis="y", labelsize=7.0, pad=0.5)
    count_tick_step = 100 if count_max >= 300 else 50
    ax_count.set_yticks(np.arange(0, count_max + count_tick_step, count_tick_step))

    ratio_line, = ax.plot(
        x,
        ratio,
        color="#c0392b",
        marker="o",
        markersize=4.6,
        linewidth=2.0,
        label=axis_unit(r"R_{\mathrm{C}}", r"\%"),
        zorder=4,
    )
    ax.axhline(100, color="gray", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_ylabel(axis_unit(r"R_{\mathrm{C}}", r"\%"), color="#c0392b")
    ax.tick_params(axis="y", labelcolor="#c0392b", labelsize=7, pad=1.0)
    ax.set_ylim(*RC_YLIM)
    ax.set_yticks(RC_TICKS)

    ax.set_xlabel("Nested subset ranked by\nsingle-planet $R_C$")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_title("(c) Nested sequence", fontweight="normal")
    annotate_nested_key_nodes(ax, nested_path)

    bar_handle = Rectangle(
        (0, 0), 1, 1,
        facecolor="#d5dbdb",
        edgecolor="#7f8c8d",
        linewidth=0.8,
        label=r"$N_{\mathrm{obs}}-N_{\mathrm{CTS}}$",
    )
    ratio_handle = Line2D(
        [0], [0],
        color="#c0392b",
        marker="o",
        markersize=4.6,
        linewidth=2.0,
        label=axis_unit(r"R_{\mathrm{C}}", r"\%"),
    )
    add_legend(ax, handles=[bar_handle, ratio_handle], loc="upper right", fontsize=5.8, handlelength=1.0)
    compact_axis(ax)


def plot_window_panel(ax: plt.Axes, window_df: pd.DataFrame, role_labels: dict[str, str]) -> None:
    colors = {
        "best_single": "#c0392b",
        "best_multi": "#d68910",
        "all_7": "#566573",
    }
    for role in ["best_single", "best_multi", "all_7"]:
        flare = window_df[
            (window_df["Dataset"] == "sf") & (window_df["Subset_Role"] == role)
        ].sort_values("Window")
        ax.plot(
            flare["Window"],
            flare["Conj_Ratio"],
            linewidth=2.0,
            color=colors[role],
            linestyle="-",
            zorder=3,
        )
        sg = window_df[
            (window_df["Dataset"] == "sg") & (window_df["Subset_Role"] == role)
        ].sort_values("Window")
        ax.plot(
            sg["Window"],
            sg["Conj_Ratio"],
            linewidth=1.4,
            color=colors[role],
            linestyle=(0, (3, 1.5)),
            solid_capstyle="round",
            zorder=2,
        )

    ax.axhline(100, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"Window half-width" "\n" r"$w/(^\circ)$")
    ax.set_ylabel(axis_unit(r"R_{\mathrm{C}}", r"\%"))
    ax.set_xticks(sorted(window_df["Window"].unique()))
    ax.set_title("(d) Window scan", fontweight="normal")
    ax.set_ylim(*RC_YLIM)
    ax.set_yticks(RC_TICKS)

    handles = [
        Line2D([0], [0], color=colors["best_single"], lw=2.0, label=role_labels["best_single"]),
        Line2D([0], [0], color=colors["best_multi"], lw=2.0, label=role_labels["best_multi"]),
        Line2D([0], [0], color=colors["all_7"], lw=2.0, label=role_labels["all_7"]),
        Line2D([0], [0], color="#555555", lw=1.6, linestyle="-", label="Flare"),
        Line2D([0], [0], color="#555555", lw=1.6, linestyle=(0, (3, 1.5)), label="SG"),
    ]
    add_legend(ax, handles=handles, loc="center right", ncol=1, fontsize=5.8, handlelength=2.0)
    ax.grid(axis="y", color="#e5e7e9", linewidth=0.8)
    compact_axis(ax)


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename, table in tables.items():
        table.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

    combined = pd.concat(
        [
            tables[FIG07_TABLE_NAMES["distribution"]],
            tables[FIG07_TABLE_NAMES["nested_path"]],
            tables[FIG07_TABLE_NAMES["window_sensitivity"]],
            tables[FIG07_TABLE_NAMES["multi_planet_summary"]],
        ],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(os.path.join(OUTPUT_DIR, FIG07_SUMMARY_CSV), index=False)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_sf = load_subset_data(SF_CSV, "sf")
    df_sg = load_subset_data(SG_CSV, "sg")
    df_all = pd.concat([df_sf, df_sg], ignore_index=True)

    df_ref = df_all[df_all["Window"] == REF_WINDOW].copy()
    df_sf_ref = df_ref[df_ref["Dataset"] == "sf"].copy()

    distribution = build_distribution_summary(df_all)
    nested_path, ranking = build_nested_path(df_sf_ref)
    best_single_label = ranking[0]
    best_multi_label = (
        df_sf_ref[df_sf_ref["N_Planets"] >= 2]
        .sort_values(["Conj_Ratio", "Conj_p"], ascending=[False, True])
        .iloc[0]["Label"]
    )
    multi_detail, multi_summary = build_multi_planet_summary(df_sf_ref)
    window_sensitivity = build_window_sensitivity(df_all, best_single_label, best_multi_label)

    tables = {
        FIG07_TABLE_NAMES["distribution"]: distribution,
        FIG07_TABLE_NAMES["nested_path"]: nested_path,
        FIG07_TABLE_NAMES["window_sensitivity"]: window_sensitivity,
        FIG07_TABLE_NAMES["multi_planet_summary"]: multi_summary,
        FIG07_TABLE_NAMES["multi_planet_detail"]: multi_detail,
    }
    save_tables(tables)

    # Staleness check: raw Conj_p in FDR table must match nested_path's Conj_p
    # exactly (both come from the same sf_subset_scan_no_earth.csv). q-values
    # themselves no longer appear in the figure (now reported only in body
    # text), but the cross-table consistency check is still worth keeping.
    if not os.path.exists(FDR_CSV):
        raise FileNotFoundError(
            f"Missing {FDR_CSV}; run 07_fdr_audit.py before plotting Fig07."
        )
    fdr_df = pd.read_csv(FDR_CSV)
    fdr_sf = fdr_df[
        (fdr_df["Source_File"] == "sf_subset_scan_no_earth.csv")
        & (fdr_df["Window"] == REF_WINDOW)
    ]
    max_p_diff = 0.0
    for _, row in nested_path.iterrows():
        match = fdr_sf[fdr_sf["Label"] == row["Subset_Label"]]
        if len(match) == 1:
            diff = abs(float(match.iloc[0]["Conj_p"]) - float(row["Conj_p"]))
            max_p_diff = max(max_p_diff, diff)
    if max_p_diff > 1e-9:
        raise ValueError(
            f"Fig07 FDR table is stale: max raw-p mismatch = {max_p_diff:.3g}. "
            "Rerun 07_fdr_audit.py, then rerun this script."
        )
    print(f"[FDR] validated: {FDR_CSV}")

    apply_acta_style("double")
    fig, axes = plt.subplots(
        1, 4,
        figsize=figsize_double(aspect=0.52),
        gridspec_kw={"width_ratios": [0.92, 0.92, 1.24, 1.02]},
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.22, wspace=0.62)
    role_display_labels = {
        "best_single": expand_label(best_single_label),
        "best_multi": "Ven+Mars",
        "all_7": "All 7",
    }
    plot_distribution_panel(axes[0], df_ref)
    plot_asym_panel(axes[1], df_ref)
    plot_nested_panel(axes[2], nested_path)
    plot_window_panel(axes[3], window_sensitivity, role_display_labels)

    save_dual(fig, os.path.join(OUTPUT_DIR, f"{FIG07_FIG_BASENAME}.eps"))
    plt.close(fig)

    print("Formal subset-scan analysis completed.")
    print(f"  Reference window: w={REF_WINDOW} deg")
    print(f"  Best single subset: {best_single_label}")
    print(f"  Best multi-planet subset: {best_multi_label}")
    print(f"  Single-planet ranking: {' > '.join(ranking)}")
    print("  Outputs written to results/05_multidimensional/")


if __name__ == "__main__":
    main()

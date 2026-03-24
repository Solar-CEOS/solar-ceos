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
from typing import Iterable

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
PLANET_ORDER = ["Mer", "Ven", "Mar", "Jup", "Sat", "Ura", "Nep"]
PLANET_FULL_NAMES = {
    "Mer": "Mercury", "Ven": "Venus", "Mar": "Mars",
    "Jup": "Jupiter", "Sat": "Saturn", "Ura": "Uranus", "Nep": "Neptune",
}
DATASET_LABELS = {"sf": "Flare", "sg": "Sunspot"}
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
            s=16,
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
            for element in ["whiskers", "caps", "medians"]:
                for line in bp[element]:
                    line.set_color(colors[dataset])

    ax.axhline(100, color="gray", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xlim(0.4, 7.6)
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels([str(i) for i in range(1, 8)])
    ax.set_xlabel("Planets in subset ($N_P$)")
    ax.set_ylabel("Conjunction ratio $R_C$ (%)")
    ax.set_title("(a) All 127 subsets at $w=2^\\circ$")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color="#e5e7e9", linewidth=0.8)


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
            s=16,
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
            for element in ["whiskers", "caps", "medians"]:
                for line in bp[element]:
                    line.set_color(colors[dataset])

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xlim(0.4, 7.6)
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels([str(i) for i in range(1, 8)])
    ax.set_xlabel("Planets in subset ($N_P$)")
    ax.set_ylabel("Asymmetric amplitude Asym (%)")
    ax.set_title("(b) Asym distribution at $w=2^\\circ$")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color="#e5e7e9", linewidth=0.8)


def plot_nested_panel(ax: plt.Axes, nested_path: pd.DataFrame, fdr_q: dict[int, float] | None = None) -> None:
    x = nested_path["Step"].to_numpy()
    excess = nested_path["Conj_Excess"].to_numpy()
    ratio = nested_path["Conj_Ratio"].to_numpy()

    ax.bar(x, excess, width=0.65, color="#d5dbdb", edgecolor="#7f8c8d", linewidth=0.8, zorder=2)
    ax.set_xlabel("Nested subset ranked by single-planet $R_C$")
    ax.set_ylabel("Observed minus CTS expected count")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_title("(c) Coverage increases while relative enrichment dilutes")
    ax.grid(axis="y", color="#e5e7e9", linewidth=0.8)

    ax2 = ax.twinx()
    ax2.plot(x, ratio, color="#c0392b", marker="o", linewidth=2.0, zorder=4)
    ax2.axhline(100, color="gray", linestyle="--", linewidth=1.0, zorder=1)
    ax2.set_ylabel("$R_C$ (%)", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_ylim(min(100, ratio.min() - 2), ratio.max() + 3)

    annotations = {
        1: {"label": "Venus",        "xytext": (20, 22),  "ha": "center"},
        2: {"label": "Venus+Mars",   "xytext": (40, 5),   "ha": "center"},
        5: {"label": "Top 5 subset", "xytext": (-15, 22), "ha": "center"},
        int(x[-1]): {"label": "All 7 planets", "xytext": (-20, -28), "ha": "center"},
    }
    for step, cfg in annotations.items():
        y = float(nested_path.loc[nested_path["Step"] == step, "Conj_Ratio"].iloc[0])
        ann_text = f"{cfg['label']}\n{y:.2f}%"
        if fdr_q and step in fdr_q:
            ann_text += f"\n$q$ = {fdr_q[step]:.3f}"
        ax2.annotate(
            ann_text,
            xy=(step, y),
            xytext=cfg["xytext"],
            textcoords="offset points",
            ha=cfg["ha"],
            fontsize=8,
            color="#922b21",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none"),
            arrowprops=dict(arrowstyle="-", color="#922b21", linewidth=0.8, shrinkA=0, shrinkB=4),
        )


def plot_window_panel(ax: plt.Axes, window_df: pd.DataFrame, role_labels: dict[str, str]) -> None:
    colors = {
        "best_single": "#c0392b",
        "best_multi": "#d68910",
        "all_7": "#566573",
    }
    flare_only = window_df[window_df["Dataset"] == "sf"].copy()

    for role in ["best_single", "best_multi", "all_7"]:
        grp = flare_only[flare_only["Subset_Role"] == role].sort_values("Window")
        ax.plot(
            grp["Window"],
            grp["Conj_Ratio"],
            marker="o",
            linewidth=2.0,
            color=colors[role],
            label=role_labels[role],
        )

    ax.axhline(100, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Window width $w$ (deg)")
    ax.set_ylabel("Conjunction ratio $R_C$ (%)")
    ax.set_xticks(sorted(flare_only["Window"].unique()))
    ax.set_title("(d) Window sensitivity for selected flare subsets")
    ax.set_ylim(99.0, 128.2)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=8)
    ax.grid(axis="y", color="#e5e7e9", linewidth=0.8)


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
    os.makedirs(FIG_DIR, exist_ok=True)

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

    # ── Load FDR q-values for nested path annotations ──
    fdr_q: dict[int, float] = {}
    if os.path.exists(FDR_CSV):
        fdr_df = pd.read_csv(FDR_CSV)
        fdr_sf = fdr_df[
            (fdr_df["Source_File"] == "sf_subset_scan_no_earth.csv")
            & (fdr_df["Window"] == REF_WINDOW)
        ]
        for _, row in nested_path.iterrows():
            match = fdr_sf[fdr_sf["Label"] == row["Subset_Label"]]
            if len(match) == 1:
                fdr_q[int(row["Step"])] = float(match.iloc[0]["Conj_q_window"])

    fig, axes = plt.subplots(1, 4, figsize=(23, 6.1), gridspec_kw={"width_ratios": [1.3, 1.3, 1.1, 1.0]})
    fig.subplots_adjust(wspace=0.38)
    role_display_labels = {
        "best_single": expand_label(best_single_label),
        "best_multi": expand_label(best_multi_label),
        "all_7": "All 7 planets",
    }
    plot_distribution_panel(axes[0], df_ref)
    plot_asym_panel(axes[1], df_ref)
    plot_nested_panel(axes[2], nested_path, fdr_q=fdr_q)
    plot_window_panel(axes[3], window_sensitivity, role_display_labels)

    for ext in ["eps", "png"]:
        out_path = os.path.join(OUTPUT_DIR, f"{FIG07_FIG_BASENAME}.{ext}")
        plt.savefig(out_path, format=ext, dpi=300, bbox_inches="tight")

    fig_path = os.path.join(FIG_DIR, f"{FIG07_FIG_BASENAME}.eps")
    plt.savefig(fig_path, format="eps", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Formal subset-scan analysis completed.")
    print(f"  Reference window: w={REF_WINDOW} deg")
    print(f"  Best single subset: {best_single_label}")
    print(f"  Best multi-planet subset: {best_multi_label}")
    print(f"  Single-planet ranking: {' > '.join(ranking)}")
    print("  Outputs written to results/05_multidimensional/")


if __name__ == "__main__":
    main()

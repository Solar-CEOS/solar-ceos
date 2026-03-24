#!/usr/bin/env python3
"""
07_fdr_audit.py
================
Post-process existing result tables with Benjamini-Hochberg FDR correction.

This script does not rerun CTS or regenerate upstream results. It only reads
existing CSV outputs, applies BH-FDR to pre-defined test families, and writes
audit tables under:

  results/05_multidimensional/fdr_audit/

Covered families:
  1. Main window scans (algo1 total-pairs CSVs)
  2. Algo2 at-least-one window scans
  3. Phase-rose bin scans (36 bins per displayed planet)
  4. Subset scans (127 subsets per window, conjunction/opposition separately,
     plus a combined conjunction+opposition family)
  5. Solar-cycle key-subset summaries
  6. Existing multi-alignment outputs (summary only; those files already carry
     BH-adjusted p-values)
  7. Table 2 latitude asymmetry (6 categories x 4 test types)
  8. Decay boundary window scans

Usage:
  python notebooks/05_multidimensional/07_fdr_audit.py
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
RESULTS_03 = os.path.join(PROJECT_ROOT, "results", "03_coord_baseline")
RESULTS_04 = os.path.join(PROJECT_ROOT, "results", "04_asymmetric")
RESULTS_05 = os.path.join(PROJECT_ROOT, "results", "05_multidimensional")
OUTPUT_DIR = os.path.join(RESULTS_05, "fdr_audit")


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return BH-adjusted q-values for a 1D numeric array."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def add_bh_by_group(
    df: pd.DataFrame,
    p_col: str,
    group_cols: list[str] | tuple[str, ...],
    out_col: str,
    size_col: str,
) -> pd.DataFrame:
    """Assign BH q-values within each group defined by group_cols."""
    df = df.copy()
    df[out_col] = np.nan
    df[size_col] = np.nan

    if group_cols:
        for _, grp in df.groupby(list(group_cols), dropna=False, sort=False):
            valid = pd.to_numeric(grp[p_col], errors="coerce").notna()
            n_valid = int(valid.sum())
            df.loc[grp.index, size_col] = n_valid
            if n_valid == 0:
                continue
            q = bh_adjust(pd.to_numeric(grp.loc[valid, p_col]).to_numpy())
            df.loc[grp.loc[valid].index, out_col] = q
    else:
        valid = pd.to_numeric(df[p_col], errors="coerce").notna()
        n_valid = int(valid.sum())
        df[size_col] = n_valid
        if n_valid:
            q = bh_adjust(pd.to_numeric(df.loc[valid, p_col]).to_numpy())
            df.loc[df.loc[valid].index, out_col] = q
    return df


def write_csv(df: pd.DataFrame, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Helper: safely locate a single row (returns None if missing)
# ---------------------------------------------------------------------------
def _safe_row(df: pd.DataFrame, mask: pd.Series) -> pd.Series | None:
    sub = df.loc[mask]
    if sub.empty:
        return None
    return sub.iloc[0]


# ===== 1. Algo1 total-pairs =================================================

def audit_algo1_total_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_04, "*", "*_algo1_total_pairs.csv"))):
        df = pd.read_csv(path).copy()
        df["Source_File"] = os.path.basename(path)
        df["Domain"] = Path(path).parent.name
        df = add_bh_by_group(
            df, "p_val",
            ["Source_File", "Stage", "Group", "Type"],
            "q_group_type", "n_group_type",
        )
        df = add_bh_by_group(
            df, "p_val",
            ["Source_File", "Stage", "Group"],
            "q_group_all", "n_group_all",
        )
        df = add_bh_by_group(
            df, "p_val",
            ["Source_File"],
            "q_file", "n_file",
        )
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(
        ["Source_File", "Stage", "Group", "Type", "Window"]
    ).reset_index(drop=True)

    summary = out[
        (out["Window"] == 2) & (out["Type"] == "Conjunction")
    ][
        [
            "Source_File", "Domain", "Stage", "Group", "Window", "Ratio", "p_val",
            "q_group_type", "q_group_all", "q_file",
        ]
    ].sort_values(["Source_File", "Stage", "Group"]).reset_index(drop=True)

    write_csv(out, "algo1_total_pairs_fdr.csv")
    write_csv(summary, "algo1_total_pairs_fdr_w2_summary.csv")
    return out, summary


# ===== 2. Algo2 at-least-one ================================================

def audit_algo2_at_least_one() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_04, "*", "*_algo2_at_least_one.csv"))):
        df = pd.read_csv(path).copy()
        df["Source_File"] = os.path.basename(path)
        df["Domain"] = Path(path).parent.name
        df = add_bh_by_group(
            df, "p_val",
            ["Source_File", "Stage", "Group", "Type"],
            "q_group_type", "n_group_type",
        )
        df = add_bh_by_group(
            df, "p_val",
            ["Source_File", "Stage", "Group"],
            "q_group_all", "n_group_all",
        )
        df = add_bh_by_group(
            df, "p_val",
            ["Source_File"],
            "q_file", "n_file",
        )
        rows.append(df)

    if not rows:
        empty = pd.DataFrame()
        return empty, empty

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(
        ["Source_File", "Stage", "Group", "Type", "Window"]
    ).reset_index(drop=True)

    summary = out[
        (out["Window"] == 2) & (out["Type"] == "Conjunction")
    ][
        [
            "Source_File", "Domain", "Stage", "Group", "Window", "Ratio", "p_val",
            "q_group_type", "q_group_all", "q_file",
        ]
    ].sort_values(["Source_File", "Stage", "Group"]).reset_index(drop=True)

    write_csv(out, "algo2_at_least_one_fdr.csv")
    write_csv(summary, "algo2_at_least_one_fdr_w2_summary.csv")
    return out, summary


# ===== 3. Phase rose ========================================================

def audit_phase_rose() -> pd.DataFrame:
    path = os.path.join(RESULTS_04, "Fig06_phase_rose.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path).copy()
    df = add_bh_by_group(df, "p_val", ["Planet"], "q_planet", "n_planet")
    df = add_bh_by_group(df, "p_val", [], "q_all_bins", "n_all_bins")
    df["In_Conjunction_Band"] = df["Angle_Center"].between(-15, 15, inclusive="both")
    df = df.sort_values(["Planet", "Angle_Center"]).reset_index(drop=True)
    write_csv(df, "fig06_phase_rose_fdr.csv")
    return df


# ===== 4. Subset scans ======================================================

def assign_subset_combined_q(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Conj_q_combined"] = np.nan
    df["Opp_q_combined"] = np.nan
    df["n_combined"] = np.nan

    for _, grp in df.groupby(["Source_File", "Window"], dropna=False, sort=False):
        long = pd.DataFrame(
            {
                "Row_Index": list(grp.index) + list(grp.index),
                "Direction": ["Conjunction"] * len(grp) + ["Opposition"] * len(grp),
                "p": pd.concat([grp["Conj_p"], grp["Opp_p"]], ignore_index=True),
            }
        )
        valid = pd.to_numeric(long["p"], errors="coerce").notna()
        n_valid = int(valid.sum())
        df.loc[grp.index, "n_combined"] = n_valid
        if n_valid == 0:
            continue
        long.loc[valid, "q"] = bh_adjust(pd.to_numeric(long.loc[valid, "p"]).to_numpy())

        conj = long[long["Direction"] == "Conjunction"].set_index("Row_Index")["q"]
        opp = long[long["Direction"] == "Opposition"].set_index("Row_Index")["q"]
        df.loc[conj.index, "Conj_q_combined"] = conj.values
        df.loc[opp.index, "Opp_q_combined"] = opp.values

    return df


def audit_subset_scans() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pattern = os.path.join(RESULTS_04, "*", "*_subset_scan_no_earth.csv")
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path).copy()
        df["Source_File"] = os.path.basename(path)
        df["Domain"] = Path(path).parent.name
        df = add_bh_by_group(
            df, "Conj_p",
            ["Source_File", "Window"],
            "Conj_q_window", "Conj_n_window",
        )
        df = add_bh_by_group(
            df, "Opp_p",
            ["Source_File", "Window"],
            "Opp_q_window", "Opp_n_window",
        )

        single = df[df["N_Planets"] == 1].copy()
        single = add_bh_by_group(
            single, "Conj_p",
            ["Source_File", "Window"],
            "Conj_q_single7", "Conj_n_single7",
        )
        single = add_bh_by_group(
            single, "Opp_p",
            ["Source_File", "Window"],
            "Opp_q_single7", "Opp_n_single7",
        )
        df = df.merge(
            single[
                [
                    "Source_File", "Window", "Label",
                    "Conj_q_single7", "Conj_n_single7",
                    "Opp_q_single7", "Opp_n_single7",
                ]
            ],
            on=["Source_File", "Window", "Label"],
            how="left",
        )

        df = assign_subset_combined_q(df)
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(
        ["Source_File", "Window", "N_Planets", "Label"]
    ).reset_index(drop=True)

    summary_rows = []
    for (source_file, window), grp in out.groupby(["Source_File", "Window"], sort=False):
        summary_rows.append(
            {
                "Source_File": source_file,
                "Domain": grp["Domain"].iloc[0],
                "Window": int(window),
                "N_Subsets": len(grp),
                "N_Conj_raw_lt_0p05": int((grp["Conj_p"] < 0.05).sum()),
                "N_Conj_q_window_lt_0p05": int((grp["Conj_q_window"] < 0.05).sum()),
                "N_Opp_raw_lt_0p05": int((grp["Opp_p"] < 0.05).sum()),
                "N_Opp_q_window_lt_0p05": int((grp["Opp_q_window"] < 0.05).sum()),
                "N_Combined_q_lt_0p05": int(
                    ((grp["Conj_q_combined"] < 0.05) | (grp["Opp_q_combined"] < 0.05)).sum()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["Source_File", "Window"]
    ).reset_index(drop=True)

    write_csv(out, "subset_scan_fdr.csv")
    write_csv(summary, "subset_scan_fdr_summary.csv")
    return out, summary


# ===== 5. Solar-cycle summary ===============================================

def audit_solar_cycle_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = os.path.join(RESULTS_05, "Fig08_solar_cycle_subset_summary.csv")
    if not os.path.isfile(path):
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(path).copy()
    df = add_bh_by_group(
        df, "Conj_p",
        ["Dataset", "Window"],
        "Conj_q_dataset_window", "Conj_n_dataset_window",
    )
    df = add_bh_by_group(
        df, "Opp_p",
        ["Dataset", "Window"],
        "Opp_q_dataset_window", "Opp_n_dataset_window",
    )
    df = add_bh_by_group(
        df, "Conj_p",
        ["Dataset", "Window", "Subset_Label"],
        "Conj_q_subset_cycles", "Conj_n_subset_cycles",
    )
    df = add_bh_by_group(
        df, "Opp_p",
        ["Dataset", "Window", "Subset_Label"],
        "Opp_q_subset_cycles", "Opp_n_subset_cycles",
    )

    fisher = df[
        ["Dataset", "Dataset_Label", "Subset_Role", "Subset_Label", "Window", "Fisher_Stat", "Fisher_p"]
    ].drop_duplicates().reset_index(drop=True)
    fisher = add_bh_by_group(
        fisher, "Fisher_p",
        ["Dataset", "Window"],
        "Fisher_q_dataset_window", "Fisher_n_dataset_window",
    )

    df = df.merge(
        fisher[["Dataset", "Window", "Subset_Label", "Fisher_q_dataset_window", "Fisher_n_dataset_window"]],
        on=["Dataset", "Window", "Subset_Label"],
        how="left",
    )
    df = df.sort_values(["Dataset", "Window", "Subset_Label", "SC"]).reset_index(drop=True)
    fisher = fisher.sort_values(["Dataset", "Window", "Subset_Label"]).reset_index(drop=True)

    write_csv(df, "fig08_solar_cycle_fdr.csv")
    write_csv(fisher, "fig08_solar_cycle_fisher_fdr.csv")
    return df, fisher


# ===== 6. Multi-alignment (existing) ========================================

def audit_multi_alignment_existing() -> pd.DataFrame:
    rows = []
    for path in [
        os.path.join(RESULTS_05, "phase2_pair_alignment_ceos.csv"),
        os.path.join(RESULTS_05, "phase2_triple_alignment_ceos.csv"),
    ]:
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path).copy()
        df["Source_File"] = os.path.basename(path)
        p_col = "p_raw" if "p_raw" in df.columns else "p_val"
        q_col = "p_adj_bh" if "p_adj_bh" in df.columns else None
        sig_col = "sig_fdr" if "sig_fdr" in df.columns else None

        group_cols = ["Source_File"]
        if "Dataset" in df.columns:
            group_cols.append("Dataset")
        if "N_Planets" in df.columns:
            group_cols.append("N_Planets")

        for key, grp in df.groupby(group_cols, dropna=False, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            row = {col: val for col, val in zip(group_cols, key)}
            row["N_Tests"] = len(grp)
            row["N_Raw_lt_0p05"] = int((grp[p_col] < 0.05).sum())
            if q_col is not None:
                row["N_q_lt_0p05"] = int((grp[q_col] < 0.05).sum())
            if sig_col is not None:
                row["N_sig_fdr_true"] = int(grp[sig_col].fillna(False).astype(bool).sum())
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    write_csv(out, "existing_multi_alignment_fdr_summary.csv")
    return out


# ===== 7. Table 2 latitude asymmetry ========================================

def audit_table2_latitude() -> pd.DataFrame:
    """Apply BH-FDR across the 6 categories for each test type in Table 2."""
    xlsx_path = os.path.join(RESULTS_03, "Fig01_Latitude_Source.xlsx")
    if not os.path.isfile(xlsx_path):
        print(f"  [skip] Table 2 source not found: {xlsx_path}")
        return pd.DataFrame()

    df = pd.read_excel(xlsx_path, sheet_name="Stats").copy()

    # p-value columns to correct across 6 categories
    p_cols = {
        "p(t-test)":   ("q_ttest",   "n_ttest"),
        "Wilcoxon_P":  ("q_wilcoxon", "n_wilcoxon"),
        "KS_Test_P":   ("q_ks",      "n_ks"),
        "Levene_P":    ("q_levene",  "n_levene"),
    }
    for p_col, (q_col, n_col) in p_cols.items():
        if p_col not in df.columns:
            continue
        df = add_bh_by_group(df, p_col, [], q_col, n_col)

    write_csv(df, "table2_latitude_fdr.csv")
    return df


# ===== 8. Decay boundary ====================================================

def audit_decay_boundary() -> pd.DataFrame:
    """Apply BH-FDR across windows within each decay-boundary file."""
    rows = []
    pattern = os.path.join(RESULTS_04, "*", "*_decay_boundary.csv")
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path).copy()
        df["Source_File"] = os.path.basename(path)
        df["Domain"] = Path(path).parent.name
        df = add_bh_by_group(
            df, "Conj_p", ["Source_File"],
            "Conj_q_file", "Conj_n_file",
        )
        df = add_bh_by_group(
            df, "Opp_p", ["Source_File"],
            "Opp_q_file", "Opp_n_file",
        )
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["Source_File", "Window"]).reset_index(drop=True)
    write_csv(out, "decay_boundary_fdr.csv")
    return out


# ===== Summary text ==========================================================

def build_summary_text(
    algo1_df: pd.DataFrame,
    algo2_df: pd.DataFrame,
    rose_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    solar_df: pd.DataFrame,
    solar_fisher_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    table2_df: pd.DataFrame,
    decay_df: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("BH-FDR audit summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Family definitions")
    lines.append("- algo1/algo2: q_group_type  over windows within File x Stage x Group x Type")
    lines.append("- algo1/algo2: q_group_all   over windows x directions within File x Stage x Group")
    lines.append("- algo1/algo2: q_file        over all tests within File")
    lines.append("- phase rose:  q_planet      over 36 bins within each planet")
    lines.append("- phase rose:  q_all_bins    over all 108 bins (3 planets x 36)")
    lines.append("- subset scan: q_window      over 127 subsets at fixed File x Window (per direction)")
    lines.append("- subset scan: q_combined    over conjunction+opposition combined at fixed File x Window")
    lines.append("- subset scan: q_single7     over the 7 single-planet rows at fixed File x Window")
    lines.append("- solar-cycle: q_dataset_window over key subsets x cycles at fixed Dataset x Window")
    lines.append("- solar-cycle Fisher: q_dataset_window over combined-p rows at fixed Dataset x Window")
    lines.append("- table2 lat:  q_<test>      over 6 categories within each test type")
    lines.append("- decay bdy:   q_file        over windows within each file")
    lines.append("")

    # --- 1. Algo1 main window scan ---
    lines.append("1. Main window scan (algo1 total-pairs)")
    lines.append("-" * 50)
    if not algo1_df.empty:
        for (group_name, src_file) in [
            ("Flare Total", "sf_algo1_total_pairs.csv"),
            ("Flare C-Class", "sf_algo1_total_pairs.csv"),
        ]:
            group_val = group_name.split()[-1]
            r = _safe_row(algo1_df, (
                (algo1_df["Source_File"] == src_file)
                & (algo1_df["Group"] == group_val)
                & (algo1_df["Type"] == "Conjunction")
                & (algo1_df["Window"] == 2)
            ))
            if r is not None:
                lines.append(
                    f"- {group_name}, w=2 conj: p={r['p_val']:.6f}, "
                    f"q_group_type={r['q_group_type']:.6f}, "
                    f"q_group_all={r['q_group_all']:.6f}, q_file={r['q_file']:.6f}"
                )
    lines.append("")

    # --- 2. Algo2 at-least-one ---
    lines.append("2. Window scan (algo2 at-least-one)")
    lines.append("-" * 50)
    if not algo2_df.empty:
        for (group_name, src_file) in [
            ("Flare Total", "sf_algo2_at_least_one.csv"),
            ("Flare C-Class", "sf_algo2_at_least_one.csv"),
        ]:
            group_val = group_name.split()[-1]
            r = _safe_row(algo2_df, (
                (algo2_df["Source_File"] == src_file)
                & (algo2_df["Group"] == group_val)
                & (algo2_df["Type"] == "Conjunction")
                & (algo2_df["Window"] == 2)
            ))
            if r is not None:
                lines.append(
                    f"- {group_name}, w=2 conj: p={r['p_val']:.6f}, "
                    f"q_group_type={r['q_group_type']:.6f}, "
                    f"q_group_all={r['q_group_all']:.6f}, q_file={r['q_file']:.6f}"
                )
    else:
        lines.append("  (no algo2 files found)")
    lines.append("")

    # --- 3. Phase rose ---
    lines.append("3. Fig06 phase rose")
    lines.append("-" * 50)
    if not rose_df.empty:
        for planet, angle in [("Venus", -5.0), ("Mars", 5.0)]:
            r = _safe_row(rose_df, (rose_df["Planet"] == planet) & (rose_df["Angle_Center"] == angle))
            if r is not None:
                lines.append(
                    f"- {planet} at {angle:+.0f} deg: p={r['p_val']:.6f}, "
                    f"q_planet={r['q_planet']:.6f}, q_all_bins={r['q_all_bins']:.6f}"
                )
    lines.append("")

    # --- 4. Subset scan ---
    lines.append("4. Fig07 subset scan (Flare Total, w=2)")
    lines.append("-" * 50)
    if not subset_df.empty:
        key_labels = ["Ven", "Ven+Mar", "Mer+Ven+Mar+Jup+Sat+Ura+Nep", "Jup+Sat"]
        sf_w2 = subset_df[
            (subset_df["Source_File"] == "sf_subset_scan_no_earth.csv")
            & (subset_df["Window"] == 2)
        ]
        for label in key_labels:
            r = _safe_row(sf_w2, sf_w2["Label"] == label)
            if r is not None:
                lines.append(
                    f"- {label}: Conj p={r['Conj_p']:.4f}, "
                    f"q_window={r['Conj_q_window']:.6f}, "
                    f"q_combined={r['Conj_q_combined']:.6f}"
                )
        n_conj_q = int((sf_w2["Conj_q_window"] < 0.05).sum()) if len(sf_w2) else 0
        n_opp_q = int((sf_w2["Opp_q_window"] < 0.05).sum()) if len(sf_w2) else 0
        lines.append(
            f"- Surviving at w=2: conj {n_conj_q}/127, opp {n_opp_q}/127"
        )

        # --- 4b. C-class 7-planet ranking ---
        lines.append("")
        lines.append("4b. C-Class single-planet ranking (q_single7, w=2)")
        lines.append("-" * 50)
        c_class_w2 = subset_df[
            (subset_df["Source_File"] == "sf_C_Class_subset_scan_no_earth.csv")
            & (subset_df["Window"] == 2)
            & (subset_df["N_Planets"] == 1)
        ].sort_values("Conj_p")
        for _, r in c_class_w2.iterrows():
            sig = "***" if r["Conj_q_single7"] < 0.05 else ""
            lines.append(
                f"- {r['Label']:>3s}: Conj p={r['Conj_p']:.4f}, "
                f"q_single7={r['Conj_q_single7']:.6f} {sig}"
            )
    lines.append("")

    # --- 5. Solar cycle ---
    lines.append("5. Fig08 solar-cycle stars (Flare, w=2)")
    lines.append("-" * 50)
    if not solar_df.empty:
        sf_w2_mask = (solar_df["Dataset"] == "sf") & (solar_df["Window"] == 2)
        for subset_label, sc in [("Ven", "SC24"), ("Ven+Mar", "SC24"), ("Jup+Sat", "SC24")]:
            r = _safe_row(solar_df, sf_w2_mask & (solar_df["Subset_Label"] == subset_label) & (solar_df["SC"] == sc))
            if r is not None:
                lines.append(
                    f"- {subset_label} {sc}: Conj p={r['Conj_p']:.4f}, "
                    f"q_dataset_window={r['Conj_q_dataset_window']:.6f}"
                )
    if not solar_fisher_df.empty:
        fisher_sf = solar_fisher_df[
            (solar_fisher_df["Dataset"] == "sf") & (solar_fisher_df["Window"] == 2)
        ]
        for subset_label in ["Ven", "Ven+Mar", "Jup+Sat",
                             "Mer+Ven+Mar+Jup+Sat+Ura+Nep"]:
            r = _safe_row(fisher_sf, fisher_sf["Subset_Label"] == subset_label)
            if r is not None:
                lines.append(
                    f"- Fisher {subset_label}: p={r['Fisher_p']:.6f}, "
                    f"q={r['Fisher_q_dataset_window']:.6f}"
                )
    if not solar_df.empty:
        sf_w2_conj = (solar_df["Dataset"] == "sf") & (solar_df["Window"] == 2) & (solar_df["Conj_q_dataset_window"] < 0.05)
        lines.append(
            f"- Surviving single-cycle flare rows at q<0.05: "
            f"{int(sf_w2_conj.sum())}/16"
        )
    lines.append("")

    # --- 6. Multi-alignment ---
    lines.append("6. Existing multi-alignment outputs")
    lines.append("-" * 50)
    if not alignment_df.empty:
        for _, row in alignment_df.iterrows():
            parts = [f"{col}={row[col]}" for col in alignment_df.columns
                     if col not in {"N_Tests", "N_Raw_lt_0p05", "N_q_lt_0p05", "N_sig_fdr_true"}]
            extra = [f"N_Tests={int(row['N_Tests'])}", f"N_Raw_lt_0p05={int(row['N_Raw_lt_0p05'])}"]
            if "N_q_lt_0p05" in row and not pd.isna(row["N_q_lt_0p05"]):
                extra.append(f"N_q_lt_0p05={int(row['N_q_lt_0p05'])}")
            if "N_sig_fdr_true" in row and not pd.isna(row["N_sig_fdr_true"]):
                extra.append(f"N_sig_fdr_true={int(row['N_sig_fdr_true'])}")
            lines.append(f"  {', '.join(parts + extra)}")
    lines.append("")

    # --- 7. Table 2 latitude ---
    lines.append("7. Table 2 latitude asymmetry (6 categories)")
    lines.append("-" * 50)
    if not table2_df.empty:
        for _, r in table2_df.iterrows():
            cat = r["Category"]
            parts = [f"{cat:>10s}:"]
            for p_col, q_col in [("p(t-test)", "q_ttest"), ("Wilcoxon_P", "q_wilcoxon")]:
                if p_col in r and q_col in r:
                    sig = "***" if r[q_col] < 0.05 else ""
                    parts.append(f"{p_col}={r[p_col]:.6f} -> q={r[q_col]:.6f}{sig}")
            lines.append("  " + "  |  ".join(parts))
    lines.append("")

    # --- 8. Decay boundary ---
    lines.append("8. Decay boundary (FDR across windows per file)")
    lines.append("-" * 50)
    if not decay_df.empty:
        for src, grp in decay_df.groupby("Source_File", sort=True):
            n_total = len(grp)
            n_raw_conj = int((grp["Conj_p"] < 0.05).sum())
            n_fdr_conj = int((grp["Conj_q_file"] < 0.05).sum())
            n_raw_opp = int((grp["Opp_p"] < 0.05).sum())
            n_fdr_opp = int((grp["Opp_q_file"] < 0.05).sum())
            lines.append(
                f"- {src}: {n_total} windows  |  "
                f"Conj raw<.05={n_raw_conj}, q<.05={n_fdr_conj}  |  "
                f"Opp raw<.05={n_raw_opp}, q<.05={n_fdr_opp}"
            )
    lines.append("")

    lines.append(f"Output directory: {OUTPUT_DIR}")
    return "\n".join(lines) + "\n"


# ===== Main ==================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Running BH-FDR audit from existing result tables...\n")

    algo1_df, _ = audit_algo1_total_pairs()
    print("  [done] algo1 total-pairs")

    algo2_df, _ = audit_algo2_at_least_one()
    print("  [done] algo2 at-least-one")

    rose_df = audit_phase_rose()
    print("  [done] phase rose")

    subset_df, _ = audit_subset_scans()
    print("  [done] subset scans")

    solar_df, solar_fisher_df = audit_solar_cycle_summary()
    print("  [done] solar-cycle summary")

    alignment_df = audit_multi_alignment_existing()
    print("  [done] multi-alignment (existing)")

    table2_df = audit_table2_latitude()
    print("  [done] table2 latitude")

    decay_df = audit_decay_boundary()
    print("  [done] decay boundary")

    summary_text = build_summary_text(
        algo1_df, algo2_df, rose_df, subset_df,
        solar_df, solar_fisher_df, alignment_df,
        table2_df, decay_df,
    )
    summary_path = os.path.join(OUTPUT_DIR, "fdr_audit_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

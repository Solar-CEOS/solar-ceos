#!/usr/bin/env python3
"""
02_fig08_solar_cycle_summary.py
===============================
Build the solar-cycle summary table used by Fig08.

Outputs:
  - results/05_multidimensional/Fig08_solar_cycle_subset_summary.csv

Usage:
  python notebooks/05_multidimensional/02_fig08_solar_cycle_summary.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "05_multidimensional"
FIG08_SUMMARY_NAME = "Fig08_solar_cycle_subset_summary.csv"

SC_FLARE_FILES = {
    "SC21": "results/04_asymmetric/sf/sf_SC21_subset_scan_no_earth.csv",
    "SC22": "results/04_asymmetric/sf/sf_SC22_subset_scan_no_earth.csv",
    "SC23": "results/04_asymmetric/sf/sf_SC23_subset_scan_no_earth.csv",
    "SC24": "results/04_asymmetric/sf/sf_SC24_subset_scan_no_earth.csv",
}
SC_SUNSPOT_FILES = {
    "SC21": "results/04_asymmetric/sg/sg_SC21_subset_scan_no_earth.csv",
    "SC22": "results/04_asymmetric/sg/sg_SC22_subset_scan_no_earth.csv",
    "SC23": "results/04_asymmetric/sg/sg_SC23_subset_scan_no_earth.csv",
    "SC24": "results/04_asymmetric/sg/sg_SC24_subset_scan_no_earth.csv",
}
SC_TARGETS = [
    ("best_single", "Ven"),
    ("best_multi", "Ven+Mar"),
    ("all_7", "Mer+Ven+Mar+Jup+Sat+Ura+Nep"),
    ("jup_sat", "Jup+Sat"),
]
REF_WINDOW = 2


def fisher_pvalue_4tests(pvals: list[float]) -> tuple[float, float]:
    clipped = [max(min(float(p), 0.999999), 1e-12) for p in pvals]
    stat = -2.0 * sum(__import__("math").log(p) for p in clipped)
    x = stat / 2.0
    survival = __import__("math").exp(-x) * (1.0 + x + x**2 / 2.0 + x**3 / 6.0)
    return stat, survival


def build_solar_cycle_summary() -> pd.DataFrame:
    rows = []
    for dataset, file_map in [("sf", SC_FLARE_FILES), ("sg", SC_SUNSPOT_FILES)]:
        dataset_label = "Flare" if dataset == "sf" else "Sunspot"
        for role, label in SC_TARGETS:
            pvals = []
            section_rows = []
            for sc, rel_path in file_map.items():
                path = PROJECT_ROOT / rel_path
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                match = df[(df["Window"] == REF_WINDOW) & (df["Label"] == label)]
                if match.empty:
                    continue
                row = match.iloc[0]
                pvals.append(float(row["Conj_p"]))
                section_rows.append(
                    {
                        "Dataset": dataset,
                        "Dataset_Label": dataset_label,
                        "Subset_Role": role,
                        "Subset_Label": label,
                        "SC": sc,
                        "Window": REF_WINDOW,
                        "Conj_Ratio": float(row["Conj_Ratio"]),
                        "Conj_p": float(row["Conj_p"]),
                        "Opp_Ratio": float(row["Opp_Ratio"]),
                        "Opp_p": float(row["Opp_p"]),
                        "Asym_Amp": float(row["Asym_Amp"]),
                    }
                )
            if section_rows:
                stat, fisher_p = fisher_pvalue_4tests(pvals)
                for item in section_rows:
                    item["Fisher_Stat"] = round(stat, 3)
                    item["Fisher_p"] = round(fisher_p, 6)
                    rows.append(item)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sc_df = build_solar_cycle_summary()
    sc_path = OUTPUT_DIR / FIG08_SUMMARY_NAME
    sc_df.to_csv(sc_path, index=False)

    print("Solar-cycle summary completed.")
    print(f"  SC : {sc_path}")


if __name__ == "__main__":
    main()

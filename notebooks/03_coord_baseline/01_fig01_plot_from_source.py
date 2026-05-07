#!/usr/bin/env python3
"""
Plot Fig01 from the saved source workbook.

This is the fast, style-only entry point. Run
01_fig01_table2_lat.ipynb only when the underlying Fig01 statistics/source
data need to be regenerated.
"""

from __future__ import annotations

import os
import sys
import datetime as dt
import numbers
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style.figstyle import (
    apply_acta_style,
    figsize_double,
    save_dual,
    COORD_COLORS,
    PT_DOUBLE,
)


FIG_NAME = "Fig01_Latitude_Asymmetry_Analysis.eps"
SOURCE_NAME = "Fig01_Latitude_Source.xlsx"
TARGETS = [("All SG", 0, "Sunspots"), ("All Flare", 1, "Flares")]
COLORS = COORD_COLORS  # {"H": teal, "E": goldenrod}


def resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [Path.cwd(), here.parents[2]]:
        if (candidate / "results" / "03_coord_baseline" / SOURCE_NAME).exists():
            return candidate
    raise FileNotFoundError(f"Could not locate results/03_coord_baseline/{SOURCE_NAME}")


def fmt_p(p_val: float) -> str:
    if p_val < 0.0001:
        return r"$<\,0.0001$"
    return f"{p_val:.4f}"


def maybe_read_cycles(root: Path) -> pd.DataFrame:
    path = root / "data" / "ready" / "solar_cycle_minmax.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "start_Min" not in df.columns:
        return pd.DataFrame()
    df["start_Min"] = pd.to_datetime(df["start_Min"]).dt.tz_localize(None)
    return df


def add_cycle_lines(axes, years: pd.Series, df_cycle: pd.DataFrame) -> None:
    if df_cycle.empty:
        return
    t_min, t_max = years.min(), years.max()
    for _, row in df_cycle.iterrows():
        tm = row["start_Min"]
        if t_min <= tm <= t_max:
            for ax in axes:
                ax.axvline(tm, c="#B3B3FF", ls="--", lw=0.5)


def get_curve(curves: pd.DataFrame, key: str, suffix: str):
    col = f"{key}_{suffix}"
    if col not in curves.columns:
        return None
    return curves[col]


def parse_groupby_dates(values: pd.Series) -> pd.Series:
    def parse_one(value):
        if isinstance(value, dt.time):
            return pd.NaT
        if isinstance(value, numbers.Integral) and 1000 <= int(value) <= 3000:
            return pd.Timestamp(year=int(value), month=1, day=1)
        return pd.to_datetime(value, errors="coerce")

    parsed = values.map(parse_one)
    if parsed.isna().any():
        ordinals = parsed.map(lambda value: value.toordinal() if pd.notna(value) else pd.NA).astype("Float64")
        ordinals = ordinals.interpolate(limit_direction="both")
        parsed = pd.to_datetime([dt.date.fromordinal(int(round(value))) for value in ordinals])
    return pd.Series(parsed)


def main() -> None:
    root = resolve_project_root()
    out_dir = root / "results" / "03_coord_baseline"
    source_path = out_dir / SOURCE_NAME

    stats = pd.read_excel(source_path, sheet_name="Stats")
    curves = pd.read_excel(source_path, sheet_name="Curves")
    curves["GroupBy"] = parse_groupby_dates(curves["GroupBy"])
    df_cycle = maybe_read_cycles(root)

    apply_acta_style("double")
    fig, axes = plt.subplots(2, 2, figsize=figsize_double(aspect=0.85), sharex="col", sharey=False)

    for key, col, title_base in TARGETS:
        stats_res = stats.loc[stats["Category"] == key].iloc[0]
        years = curves["GroupBy"]

        add_cycle_lines(axes[:, col], years, df_cycle)

        ax1 = axes[0, col]
        ax1.axhline(0, c="#808080", lw=0.8)
        ah = get_curve(curves, key, "Ah")
        ae = get_curve(curves, key, "Ae")
        sig_h = get_curve(curves, key, "SigH")
        sig_e = get_curve(curves, key, "SigE")

        if sig_h is not None and sig_e is not None:
            ax1.fill_between(years, ah - sig_h, ah + sig_h, color="#D9EBE9")
            ax1.fill_between(years, ae - sig_e, ae + sig_e, color="#F4EDDA")
        else:
            print("Warning: SigH/SigE columns are missing; redraw without uncertainty bands.")

        ax1.plot(years, ah, ".-", c="#66AFA6", lw=1, ms=3, label="Heliographic")
        ax1.plot(years, ae, ".-", c="#BF9223", lw=1.5, ms=4, label="Ecliptic")
        ax1.set_title(f"{title_base}", loc="left")
        info_text = (
            f"$p_{{\\mathrm{{W}}}}$: {fmt_p(stats_res['Wilcoxon_P'])}\n"
            f"$p_{{\\mathrm{{t}}}}$: {fmt_p(stats_res['p(t-test)'])}"
        )
        ax1.text(
            0.6,
            0.95,
            info_text,
            transform=ax1.transAxes,
            va="top",
            fontsize=PT_DOUBLE["annot"],
            bbox=dict(facecolor="white", boxstyle="round,pad=0.3", edgecolor="none"),
        )
        ax1.text(
            0.95,
            0.07,
            f"$N_{{\\mathrm{{eff}}}}$: {stats_res['N_Eff']:.0f}",
            transform=ax1.transAxes,
            ha="right",
            va="top",
            fontsize=PT_DOUBLE["annot"],
        )
        ax1.set_ylim(-1.05, 1.05)
        if col == 0:
            ax1.set_ylabel("Asymmetry ($A$)")

        ax2 = axes[1, col]
        ax2.axhline(0, c="#808080", lw=0.8)
        cusum_h = get_curve(curves, key, "CUSUM_h")
        cusum_e = get_curve(curves, key, "CUSUM_e")
        ax2.fill_between(years, cusum_h, cusum_e, color="#E8E8E8", zorder=0)
        ax2.plot(years, cusum_h, "--", c="#339489", lw=1.5, label="CUSUM (Helio)")
        ax2.plot(years, cusum_e, "-", c="#BF9223", lw=2, label="CUSUM (Eclip)")

        boot_info = (
            "Mean $|A|$ (95% CI):\n"
            f"H: {stats_res['Boot_Mean_H']:.3f} "
            f"[{stats_res['Boot_CI_H_Low']:.2f}, {stats_res['Boot_CI_H_High']:.2f}]\n"
            f"E: {stats_res['Boot_Mean_E']:.3f} "
            f"[{stats_res['Boot_CI_E_Low']:.2f}, {stats_res['Boot_CI_E_High']:.2f}]"
        )
        x_pos = 0.97 if col == 0 else 0.03
        ha = "right" if col == 0 else "left"
        ax2.text(
            x_pos,
            0.03,
            boot_info,
            transform=ax2.transAxes,
            ha=ha,
            va="bottom",
            fontsize=PT_DOUBLE["annot"] - 0.5,
            bbox=dict(facecolor="white", boxstyle="round,pad=0.3", edgecolor="none"),
        )

        if col == 0:
            leg1 = ax1.legend(loc="upper left")
            leg1.get_frame().set_edgecolor("none")
            leg2 = ax2.legend(loc="upper left")
            leg2.get_frame().set_edgecolor("none")
            ax2.set_ylabel("Cumulative Sum")
        ax2.set_xlabel("Year")

    flare_mask = curves["All Flare_Ah"].notna() if "All Flare_Ah" in curves.columns else None
    if flare_mask is not None and flare_mask.any():
        flare_years_only = curves.loc[flare_mask, "GroupBy"]
        fxmin = flare_years_only.min() - pd.DateOffset(years=2)
        fxmax = flare_years_only.max() + pd.DateOffset(years=2)
        axes[0, 1].set_xlim(fxmin, fxmax)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.1, wspace=0.15, hspace=0.15)
    plot_path = out_dir / FIG_NAME.replace(".eps", "_grid.eps")
    eps, png = save_dual(fig, plot_path)
    plt.close(fig)
    print(f"Plot saved: {eps} (+ {png.name})")


def main_wide() -> None:
    """1×4 wide layout: row-major (A | A | CUSUM | CUSUM), subtitle stats below
    panel title, shared legend at bottom. The two A panels share y-axis range
    so amplitudes are directly comparable; the two CUSUM panels carry their own
    scales since sample size and time span differ.
    """
    root = resolve_project_root()
    out_dir = root / "results" / "03_coord_baseline"
    source_path = out_dir / SOURCE_NAME

    stats = pd.read_excel(source_path, sheet_name="Stats")
    curves = pd.read_excel(source_path, sheet_name="Curves")
    curves["GroupBy"] = parse_groupby_dates(curves["GroupBy"])
    df_cycle = maybe_read_cycles(root)

    apply_acta_style("double")
    fig, axes = plt.subplots(1, 4, figsize=figsize_double(aspect=0.45))
    years = curves["GroupBy"]

    sg_stats = stats.loc[stats["Category"] == "All SG"].iloc[0]
    fl_stats = stats.loc[stats["Category"] == "All Flare"].iloc[0]

    panels = [
        ("All SG",    "(a) Sunspot groups", "A",     sg_stats),
        ("All Flare", "(b) Flares",   "A",     fl_stats),
        ("All SG",    "(c) Sunspot groups", "cusum", sg_stats),
        ("All Flare", "(d) Flares",   "cusum", fl_stats),
    ]

    line_kw = dict(lw=1.0)

    for ax, (key, panel_name, kind, stats_res) in zip(axes, panels):
        if not df_cycle.empty:
            t_min, t_max = years.min(), years.max()
            for _, row in df_cycle.iterrows():
                tm = row["start_Min"]
                if t_min <= tm <= t_max:
                    ax.axvline(tm, c="#D6D6FF", ls="--", lw=0.4)

        if kind == "A":
            ax.axhline(0, c="#808080", lw=0.8)
            ah = get_curve(curves, key, "Ah")
            ae = get_curve(curves, key, "Ae")
            sig_h = get_curve(curves, key, "SigH")
            sig_e = get_curve(curves, key, "SigE")
            if sig_h is not None and sig_e is not None:
                ax.fill_between(years, ah - sig_h, ah + sig_h, color="#D9EBE9")
                ax.fill_between(years, ae - sig_e, ae + sig_e, color="#F4EDDA")
            ax.plot(years, ah, "-", c="#66AFA6", **line_kw)
            ax.plot(years, ae, "-", c="#BF9223", **line_kw)
            ax.set_ylim(-1.05, 1.05)
            n_eff = round(stats_res["N_Eff"])
            pw = stats_res["Wilcoxon_P"]
            pt = stats_res["p(t-test)"]
            if pw < 0.0001 and pt < 0.0001:
                stats_line = (
                    f"$N_{{\\mathrm{{eff}}}}{{=}}{n_eff}$,  "
                    f"$p_{{\\mathrm{{W}}}}, p_{{\\mathrm{{t}}}}{{<}}10^{{-4}}$"
                )
            else:
                stats_line = (
                    f"$N_{{\\mathrm{{eff}}}}{{=}}{n_eff}$,  "
                    f"$p_{{\\mathrm{{W}}}}{{=}}{pw:.4f}$,  "
                    f"$p_{{\\mathrm{{t}}}}{{=}}{pt:.4f}$"
                )
        else:
            ax.axhline(0, c="#808080", lw=0.8)
            cusum_h = get_curve(curves, key, "CUSUM_h")
            cusum_e = get_curve(curves, key, "CUSUM_e")
            ax.fill_between(years, cusum_h, cusum_e, color="#D9D9D9", zorder=0)
            ax.plot(years, cusum_h, "-", c="#66AFA6", **line_kw)
            ax.plot(years, cusum_e, "-", c="#BF9223", **line_kw)
            mh = stats_res["Boot_Mean_H"]
            me = stats_res["Boot_Mean_E"]
            stats_line = (
                f"$\\overline{{|A_{{\\mathrm{{H}}}}|}}={mh:.3f}$,  "
                f"$\\overline{{|A_{{\\mathrm{{E}}}}|}}={me:.3f}$"
            )

        ax.set_title(panel_name, loc="left", fontsize=PT_DOUBLE["axlabel"], pad=25)
        ax.text(
            0.5, 1.055, stats_line,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=PT_DOUBLE["annot"],
        )

        if "Flares" in panel_name:
            flare_mask = curves[f"{key}_Ah"].notna()
            if flare_mask.any():
                fy = curves.loc[flare_mask, "GroupBy"]
                ax.set_xlim(fy.min() - pd.DateOffset(years=2), fy.max() + pd.DateOffset(years=2))
            ax.xaxis.set_major_locator(mdates.YearLocator(10))
            ax.xaxis.set_minor_locator(mdates.YearLocator(2))
        else:
            ax.xaxis.set_major_locator(mdates.YearLocator(40))
            ax.xaxis.set_minor_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    legend_handles = [
        mlines.Line2D([], [], color="#BF9223", lw=1.2, label="Ecliptic"),
        mlines.Line2D([], [], color="#66AFA6", lw=1.2, label="Heliographic"),
    ]
    leg = axes[3].legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.03, 0.055),
        ncol=1,
        fontsize=6.8,
        handlelength=1.35,
        handletextpad=0.50,
        borderpad=0.30,
        labelspacing=0.24,
        borderaxespad=0.0,
    )
    leg.get_frame().set_edgecolor("#D6D6D6")
    leg.get_frame().set_linewidth(0.35)

    plt.subplots_adjust(left=0.045, right=0.995, top=0.74, bottom=0.17, wspace=0.30)

    # Group titles spanning panel pairs (above panel-level (a)/(b)/(c)/(d) labels)
    def _panel_left_x(ax):
        bbox = ax.get_position()
        return bbox.x0

    x_AB = _panel_left_x(axes[0])
    x_CD = _panel_left_x(axes[2])
    fig.text(x_AB, 0.94, "Asymmetry index $A$",
             ha="left", va="top", fontsize=PT_DOUBLE["title"] - 1)
    fig.text(x_CD, 0.94, "Cumulative Sum",
             ha="left", va="top", fontsize=PT_DOUBLE["title"] - 1)

    fig.text(0.52, 0.045, "Year", ha="center", va="bottom",
             fontsize=PT_DOUBLE["axlabel"])

    plot_path = out_dir / FIG_NAME
    eps, png = save_dual(fig, plot_path)
    plt.close(fig)
    print(f"Plot saved: {eps} (+ {png.name})")


if __name__ == "__main__":
    # Canonical layout for submission: 1×4 wide.
    # main() produces the 2×2 grid version, kept as a backup; uncomment to also
    # generate it (saves to Fig01_..._.eps, will overwrite the canonical wide).
    main_wide()
    # main()

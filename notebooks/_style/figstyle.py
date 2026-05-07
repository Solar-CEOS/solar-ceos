"""Unified figure style for CEOS paper figures.

Conforms to 《天文学报》§九图表规范 + AAS / A&A international conventions.
See ai/figstyle_migration.md for design rationale.

EPS limitation — DO NOT USE alpha=
----------------------------------
《天文学报》 only accepts EPS. EPS (PostScript Level 2) does not natively
support transparency: matplotlib's PS backend either flattens alpha back
to 1.0 or rasterizes the affected primitive. Either way the result
diverges from the PNG preview and degrades print quality.

Rule for any plotting code that targets save_dual():
    * Do NOT pass alpha=... to plot / scatter / fill_between / hist /
      bar / legend / text / etc.
    * Replace translucency with a lighter solid colour:
        - shaded CI band: use a light shade ("#FFE0E0") not red @ 0.3
        - dense scatter:  shrink markers or use a lighter colour
        - dim grid:       use grid.color="#D0D0D0", linewidth=0.5

This module's rcParams below are set to fully-opaque defaults
(legend.framealpha=1.0, no grid.alpha). Don't add alpha back here.

Usage in any plot script:

    from _style.figstyle import apply_acta_style, figsize_double, PLANET_COLORS, save_dual
    apply_acta_style(column="double")
    fig, ax = plt.subplots(figsize=figsize_double(aspect=0.5))
    ...
    save_dual(fig, "results/04_asymmetric/Fig04_flare_decay.eps")

To make this importable from notebooks/<subdir>/<script>.py:

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _style.figstyle import apply_acta_style, ...
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Page geometry (Acta double-column publication; LaTeX template is single-col).
# ---------------------------------------------------------------------------
COL_SINGLE: float = 3.5  # inches, ~89 mm
COL_DOUBLE: float = 7.2  # inches, ~180 mm

# Font sizes — single-column figs get +1pt to survive the print downscale.
PT_DOUBLE = {"axlabel": 9, "tick": 8, "legend": 7.5, "title": 10, "annot": 7}
PT_SINGLE = {"axlabel": 10, "tick": 9, "legend": 8.5, "title": 11, "annot": 8}


# ---------------------------------------------------------------------------
# Colour palettes.
# ---------------------------------------------------------------------------
# Paul Tol bright — colourblind-friendly qualitative palette.
TOL_BRIGHT = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

# Planet palette — domain-specific; tuned for visual distinguishability and
# weak associations to the planet's appearance/temperature.
PLANET_COLORS = {
    "Mercury": "#808080",  # neutral grey
    "Venus":   "#E65100",  # warm orange (hot, dense atmosphere)
    "Earth":   "#1976D2",  # blue
    "Mars":    "#C62828",  # rust red
    "Jupiter": "#6A1B9A",  # giant purple
    "Saturn":  "#B8860B",  # dark gold
    "Uranus":  "#00838F",  # cyan
    "Neptune": "#1A237E",  # deep blue
}

# Coordinate-system palette — used by Fig01 (Heliographic vs Ecliptic).
COORD_COLORS = {"H": "#00796B", "E": "#B8860B"}  # teal / goldenrod

# Direction palette — conjunction (0°) vs opposition (180°).
DIR_COLORS = {"conj": "#1B5E20", "opp": "#B71C1C", "null": "#9E9E9E"}


# ---------------------------------------------------------------------------
# Style application.
# ---------------------------------------------------------------------------
def apply_acta_style(column: str = "double") -> None:
    """Apply 《天文学报》/ AAS-compliant rcParams.

    column: "double" (7.2") or "single" (3.5"). Single uses +1pt fonts.
    """
    if column not in ("single", "double"):
        raise ValueError(f"column must be 'single' or 'double', got {column!r}")
    pt = PT_SINGLE if column == "single" else PT_DOUBLE

    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        plt.style.use("default")

    plt.rcParams.update({
        # Fonts — serif primary, with DejaVu Sans appended as Unicode-glyph fallback
        # (★/☆/etc. literals used in some annotations are not in serif fonts).
        "font.family": "serif",
        "font.serif": [
            "DejaVu Serif", "Times New Roman", "Times", "Liberation Serif",
            "DejaVu Sans",  # fallback for Unicode glyphs (★/☆/etc.)
        ],
        "mathtext.fontset": "stix",  # STIX provides \bigstar, \star, etc.
        "axes.unicode_minus": False,

        # Sizes
        "font.size": pt["axlabel"],
        "axes.labelsize": pt["axlabel"],
        "axes.titlesize": pt["title"],
        "xtick.labelsize": pt["tick"],
        "ytick.labelsize": pt["tick"],
        "legend.fontsize": pt["legend"],
        "figure.titlesize": pt["title"] + 1,

        # Axes — Acta requires 4-side frame + inward ticks
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,

        # Legend — frame inside plot, opaque (EPS compat), light grey edge
        "legend.frameon": True,
        "legend.framealpha": 1.0,  # EPS: no alpha
        "legend.facecolor": "white",
        "legend.edgecolor": "lightgray",
        "legend.fancybox": True,
        "legend.borderpad": 0.4,

        # Grid (off by default; turn on per axis if needed) — EPS-safe light grey
        "axes.grid": False,
        "grid.color": "#D0D0D0",  # was grid.alpha=0.25 → EPS incompat
        "grid.linewidth": 0.5,

        # Output
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Figure-size helpers.
# ---------------------------------------------------------------------------
def figsize_double(aspect: float = 0.5) -> tuple[float, float]:
    """Double-column figure: (7.2, 7.2*aspect)."""
    return (COL_DOUBLE, COL_DOUBLE * aspect)


def figsize_single(aspect: float = 0.75) -> tuple[float, float]:
    """Single-column figure: (3.5, 3.5*aspect)."""
    return (COL_SINGLE, COL_SINGLE * aspect)


# ---------------------------------------------------------------------------
# Axis-label helpers.
# ---------------------------------------------------------------------------
def _is_compound_unit(unit: str) -> bool:
    """Compound units need parentheses per Acta §9.4 (e.g., ``km·s⁻¹``)."""
    return r"\cdot" in unit or " " in unit


def axis_unit(label: str, unit: str) -> str:
    """Format axis label in 《天文学报》 slash convention.

    Per Acta §9.4: variable in italic math, **unit in upright math**, with
    ``/`` separator and parentheses around compound units.
        simple   : ``P/s``           ← ``axis_unit('P', 's')``
        compound : ``v/(km·s⁻¹)``    ← ``axis_unit('v', r'km\\cdot s^{-1}')``

    Pass mathmode-safe strings (no ``$...$``); the unit is auto-wrapped in
    ``\\mathrm{}`` so letters render upright.
    """
    if _is_compound_unit(unit):
        return rf"${label}/\mathrm{{({unit})}}$"
    return rf"${label}/\mathrm{{{unit}}}$"


def axis_log10(label: str, unit: str) -> str:
    """Log-base-10 axis label per Acta §9.8.

    Acta requires ``\\lg`` (not ``\\log``), normalizes the argument with ``/``,
    and uses upright unit. Spec example: ``\\lg[v/(km·s⁻¹)]`` with the unit
    in upright math and a thin space after ``\\lg``.
    """
    if _is_compound_unit(unit):
        return rf"$\lg\,\left[{label}/\mathrm{{({unit})}}\right]$"
    return rf"$\lg\,\left[{label}/\mathrm{{{unit}}}\right]$"


# ---------------------------------------------------------------------------
# EPS safety: scan figure for alpha<1 artists (transparency unsupported in EPS).
# ---------------------------------------------------------------------------
def _scan_alpha(fig) -> list[tuple[str, float]]:
    """Walk fig, collect (label, alpha) for any artist with alpha < 1.

    Catches three sources:
      1. artist.get_alpha() set explicitly via alpha= kwarg
      2. facecolor / edgecolor RGBA tuple with channel-3 < 1
      3. text bbox patches (Text.get_bbox_patch())
    """
    issues: list[tuple[str, float]] = []
    seen: set[int] = set()

    def visit(art, path: str) -> None:
        if id(art) in seen:
            return
        seen.add(id(art))

        # Only 0 < alpha < 1 is a real EPS problem; alpha=0 means "hidden"
        # (e.g., Spine.facecolor=0 to suppress fill) and renders nothing.
        a = art.get_alpha()
        if a is not None and 0.0 < a < 1.0:
            issues.append((f"{path} (alpha={a:.2f})", float(a)))

        for attr in ("get_facecolor", "get_edgecolor"):
            fn = getattr(art, attr, None)
            if fn is None:
                continue
            try:
                c = fn()
            except Exception:
                continue
            if c is None:
                continue
            try:
                arr = np.atleast_2d(np.asarray(c, dtype=float))
            except (TypeError, ValueError):
                continue
            if arr.size == 0 or arr.shape[-1] != 4:
                continue
            mask = (arr[:, 3] > 0.0) & (arr[:, 3] < 1.0)
            if mask.any():
                amin = float(arr[mask, 3].min())
                issues.append((f"{path}.{attr[4:]} (alpha={amin:.2f})", amin))

        bbox_fn = getattr(art, "get_bbox_patch", None)
        if bbox_fn is not None:
            try:
                bp = bbox_fn()
            except Exception:
                bp = None
            if bp is not None:
                visit(bp, f"{path}.bbox")

        for child in getattr(art, "get_children", lambda: [])():
            visit(child, f"{path}/{type(child).__name__}")

    visit(fig, "Figure")
    return issues


def assert_eps_safe(fig, *, strict: bool = True) -> list[tuple[str, float]]:
    """Raise (or warn) if figure has any alpha<1 artist.

    Returns the list of offenders (empty if clean).
    """
    issues = _scan_alpha(fig)
    if not issues:
        return issues
    head = "\n  ".join(p for p, _ in issues[:15])
    msg = (
        f"EPS cannot render transparency: {len(issues)} alpha<1 artist(s) found.\n"
        f"  {head}"
        + ("" if len(issues) <= 15 else f"\n  ... ({len(issues) - 15} more)")
        + "\nReplace alpha= with a lighter solid colour. See figstyle.py docstring."
    )
    if strict:
        raise RuntimeError(msg)
    warnings.warn(msg, stacklevel=3)
    return issues


# ---------------------------------------------------------------------------
# Acta content validators (post-construction figure checks).
# ---------------------------------------------------------------------------
_CJK_RANGES = (
    (0x3000, 0x303F),   # CJK Symbols and Punctuation (《》「」 etc.)
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF),   # Halfwidth and Fullwidth Forms
)


def _has_cjk(s: str) -> bool:
    for c in s:
        cp = ord(c)
        for lo, hi in _CJK_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def validate_acta_figure(fig, *, strict: bool = True) -> list[tuple[str, str]]:
    """Check fig against Acta §九 content rules; return list of (severity, msg).

    Hard checks (severity='error'):
      - Acta §9.2: figures must be English-only — flags any matplotlib Text
        artist containing CJK characters. The bilingual caption is set in
        LaTeX via \\figcaption, NOT inside the matplotlib figure.

    Soft checks (severity='warning'):
      - Acta §9.7: ``图例框需在图框内部`` — flags legends with frameon=False.
        The spec phrasing is ambiguous (some interpret it as "if framed,
        frame must be inside"); we warn rather than error.

    With strict=True (default) errors raise RuntimeError; warnings always warn.
    Pass strict=False to downgrade errors to warnings during debug.
    """
    import matplotlib.text as mtext

    issues: list[tuple[str, str]] = []

    for art in fig.findobj(mtext.Text):
        s = art.get_text()
        if s and _has_cjk(s):
            issues.append(("error", f"CJK text in figure: {s!r}"))

    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None and not leg.get_frame_on():
            label = ax.get_label() or repr(ax)
            issues.append(("warning", f"Legend on axes [{label}] has frameon=False"))
    for leg in fig.legends:
        if not leg.get_frame_on():
            issues.append(("warning", "Figure-level legend has frameon=False"))

    warns = [m for s, m in issues if s == "warning"]
    errs = [m for s, m in issues if s == "error"]

    if warns:
        warnings.warn(
            "Acta §9.7 prefers framed legends:\n  " + "\n  ".join(warns),
            stacklevel=3,
        )
    if errs:
        msg = "Acta §9.2 requires English-only figures:\n  " + "\n  ".join(errs)
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=3)

    return issues


# ---------------------------------------------------------------------------
# I/O helpers.
# ---------------------------------------------------------------------------
def save_dual(fig, eps_path, *, strict: bool = True) -> tuple[Path, Path]:
    """Save the figure as both EPS (vector, for submission) and PNG (preview).

    Acta requires EPS only; PNG is for in-repo visual review and Claude reading.
    Creates the parent directory if it does not exist.

    Before saving EPS, runs two hard checks:
      1. assert_eps_safe(fig)     — no 0<alpha<1 (EPS can't render transparency)
      2. validate_acta_figure(fig) — no CJK text + warns on frameon=False legends

    With strict=True (default) violations raise RuntimeError. Pass strict=False
    to downgrade to warnings (debug only; never use for final submission).

    The PNG is always saved first so it remains available for visual debugging
    even when the EPS checks fail.

    Returns (eps_path, png_path).
    """
    eps_path = Path(eps_path)
    if eps_path.suffix.lower() != ".eps":
        raise ValueError(f"save_dual expects an .eps target path, got {eps_path}")
    eps_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = eps_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    assert_eps_safe(fig, strict=strict)
    validate_acta_figure(fig, strict=strict)
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    return eps_path, png_path


# ---------------------------------------------------------------------------
# Convenience: planet ordering as used throughout the paper.
# ---------------------------------------------------------------------------
PLANET_ORDER: tuple[str, ...] = (
    "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
)


def planet_color(name: str) -> str:
    """Look up a planet colour by name; fall back to grey on miss."""
    return PLANET_COLORS.get(name, "#888888")


def planet_colors_for(planets: Iterable[str]) -> list[str]:
    """Return colour list aligned with the given planet sequence."""
    return [planet_color(p) for p in planets]

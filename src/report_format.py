"""Numeric and LaTeX cell formatting for the Appendix E table."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _canonical_frame(frame: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    """Validate and order a generated report before it becomes a canonical CSV."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing canonical columns: {missing}")

    canonical = frame.loc[:, columns].copy()
    if canonical.empty:
        raise ValueError(f"{name} contains no rows")
    return canonical


def _fmt_p(value: float) -> str:
    """Return a compact LaTeX p-value without implying greater precision."""
    if value < 0.001:
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"
    if value < 0.01:
        return f"${value:.5f}$"
    return f"${value:.3f}$"


def _ci_spans_zero(row: pd.Series) -> bool:
    return float(row.ci_lo) <= 0.0 <= float(row.ci_hi)


def _fmt_effect(row: pd.Series, field: str) -> str:
    """Point estimate first, with the interval set smaller and lighter.

    The dagger flags an interval that includes zero, so the one qualified cell
    is visible in the table itself rather than only in the surrounding text.
    """
    value = float(row[field])
    flag = r"\textsuperscript{\dag}" if _ci_spans_zero(row) else ""
    return (
        rf"${value:+.4f}${flag}\;"
        rf"{{\tiny\textcolor{{black!55}}{{$[{float(row.ci_lo):+.4f},\,"
        rf"{float(row.ci_hi):+.4f}]$}}}}"
    )


def _dagger_note(frames: Iterable[pd.DataFrame]) -> list[str]:
    """Return the dagger legend, empty unless some interval includes zero.

    A caption that omits the legend while a table carries daggers, or claims a
    dagger no reader can find, is a silent failure. Deriving the line from the
    frames keeps the two in step.

    Args:
        frames: The statistics frames rendered in the table.

    Returns:
        A one-line caption fragment, or an empty list when no interval spans zero.
    """
    flagged = any(frame.apply(_ci_spans_zero, axis=1).any() for frame in frames)
    return [r"$\dagger$ marks an interval that includes zero."] if flagged else []


def _fmt_w(value: float) -> str:
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"


def _table_cells(row: pd.Series, effect_field: str) -> list[str]:
    return [
        _fmt_effect(row, effect_field),
        str(int(row.n_pos)),
        _fmt_w(float(row.w_stat)),
        f"${float(row.rank_biserial):.3f}$",
        _fmt_p(float(row.p_wilcoxon_bh)),
    ]


def _lookup(
    frame: pd.DataFrame,
    *,
    dimension: str,
    modality: str | None = None,
) -> pd.Series:
    mask = frame["dimension"].eq(dimension)
    if modality is not None:
        mask &= frame["modality"].eq(modality)
    rows = frame.loc[mask]
    if len(rows) != 1:
        key = f"{dimension}/{modality}" if modality else dimension
        raise ValueError(f"Expected exactly one row for {key}, found {len(rows)}")
    return rows.iloc[0]

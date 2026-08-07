r"""Render the two Appendix F tables.

Both put the models side by side, matching Appendix E, and both are built from
the same scaffold: only the header, the row loop, and the caption differ. They
live in one module because the alternative duplicates the ``table*`` wrapper,
the two-model pairing, and the effect cell.

The matched effect table omits ``n_+``, ``W``, ``r_rb`` and ``p_BH``: every one
of its 24 comparisons is fully positive, so those columns would repeat a single
value twelve times per model. The caption states them once instead, and
``_require_degenerate_ranks`` fails the build if that ever stops being true.
The contrast table does tabulate them, because there they genuinely vary.

The matched-to-marginal ratio stays in ``matched_factorial_significance.csv``
and in the method manifest, but is not tabulated: the appendix prose argues the
two constructions use different comparison sets and different within-group
references, so a column of ratios would invite exactly the reading that prose
rules out.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT
from .report_format import (
    _canonical_frame,
    _dagger_note,
    _fmt_effect,
    _fmt_p,
    _lookup,
    _table_cells,
)
from .report_schema import (
    DIM_LABELS,
    DIM_ORDER,
    MATCHED_CONTRAST_COLUMNS,
    MATCHED_CONTRAST_TABLE_BEGIN,
    MATCHED_CONTRAST_TABLE_END,
    MATCHED_CONTRAST_TABLE_LABEL,
    MATCHED_FACTORIAL_COLUMNS,
    MATCHED_TABLE_BEGIN,
    MATCHED_TABLE_END,
    MATCHED_TABLE_LABEL,
    MODALITY_LABELS,
    MODALITY_ORDER,
    PAPER_MODELS,
)


def _paper_model_frames(filename: str) -> list[pd.DataFrame]:
    """Read one canonical CSV for each paper model, in table column order.

    Args:
        filename: Basename under ``outputs/<model>/``.

    Returns:
        One frame per model, ordered as ``PAPER_MODELS``.
    """
    return [pd.read_csv(PROJECT_ROOT / "outputs" / model / filename) for model in PAPER_MODELS]


def _wrap_table(
    header: list[str],
    rows: list[str],
    caption: list[str],
    *,
    begin: str,
    end: str,
    column_spec: str,
    tabcolsep: str,
    label: str,
    resize: bool = False,
) -> str:
    r"""Assemble a marked ``table*`` from its header, body, and caption lines.

    Args:
        header: Lines between ``\toprule`` and ``\midrule``.
        rows: Body lines between ``\midrule`` and ``\bottomrule``.
        caption: Caption lines, without the surrounding braces.
        begin: Opening marker comment.
        end: Closing marker comment.
        column_spec: The ``tabular`` column specification.
        tabcolsep: Inter-column padding.
        label: The table's LaTeX label.
        resize: Whether to wrap the tabular in ``\resizebox``.

    Returns:
        The marked LaTeX block, ready to paste into ``paper.tex``.
    """
    lines = [
        begin,
        r"\begin{table*}[tbh]",
        r"\centering",
        r"\scriptsize",
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        r"\renewcommand{\arraystretch}{1.15}",
    ]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend([rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule"])
    lines.extend(header)
    lines.append(r"\midrule")
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}%" if resize else r"\end{tabular}"])
    if resize:
        lines.append(r"}")
    lines.append(r"\caption{")
    lines.extend(caption)
    lines.extend([r"}", rf"\label{{{label}}}", r"\end{table*}", end])
    return "\n".join(lines) + "\n"


def _require_degenerate_ranks(frames: list[pd.DataFrame]) -> str:
    """Confirm the matched effect table may state its rank statistics in prose.

    The caption asserts that every image-level contrast is positive, so ``n_+``,
    ``W``, ``r_rb`` and ``p_BH`` are constant and are left out of the table. That
    claim is data-dependent; if it ever stops holding, failing here is better
    than a caption that quietly misreports the analysis.

    Args:
        frames: The models' canonical matched-factorial statistics.

    Returns:
        The single formatted BH-adjusted p-value shared by every comparison.

    Raises:
        ValueError: If any comparison is not fully positive, or the adjusted
            p-values do not all render identically.
    """
    for frame in frames:
        offending = frame[
            (frame["n_pos"] != frame["n_images"])
            | (frame["w_stat"] != 0.0)
            | (frame["rank_biserial"] != 1.0)
        ]
        if not offending.empty:
            cells = offending[["dimension", "modality"]].to_dict("records")
            raise ValueError(
                "the matched effect caption states n_+, W and r_rb as constants, "
                f"but these comparisons are no longer degenerate: {cells}"
            )
    rendered = sorted(
        {_fmt_p(float(value)) for frame in frames for value in frame["p_wilcoxon_bh"]}
    )
    if len(rendered) != 1:
        raise ValueError(
            "the matched effect caption quotes one adjusted p-value for every "
            f"comparison, but the table would show: {rendered}"
        )
    return rendered[0]


def _matched_delta_rows(qwen: pd.DataFrame, gemma: pd.DataFrame) -> list[str]:
    """Render the twelve modality-by-dimension rows with both models side by side.

    Args:
        qwen: Qwen3-VL canonical matched-factorial statistics.
        gemma: Gemma4 canonical matched-factorial statistics.

    Returns:
        The body lines of the matched effect table.

    Raises:
        ValueError: If the two models disagree on a cell's matched-pair count,
            which the single shared ``Pairs`` column assumes.
    """
    lines: list[str] = []
    for modality in MODALITY_ORDER:
        for position, dimension in enumerate(DIM_ORDER):
            qwen_row = _lookup(qwen, dimension=dimension, modality=modality)
            gemma_row = _lookup(gemma, dimension=dimension, modality=modality)
            pairs = int(qwen_row.n_matched_pairs)
            if pairs != int(gemma_row.n_matched_pairs):
                raise ValueError(
                    "the shared Pairs column requires both models to agree on "
                    f"{dimension}/{modality}"
                )
            label = (
                rf"\multirow{{{len(DIM_ORDER)}}}{{*}}{{{MODALITY_LABELS[modality]}}}"
                if position == 0
                else ""
            )
            cells = [
                label,
                DIM_LABELS[dimension],
                str(pairs),
                _fmt_effect(qwen_row, "delta"),
                _fmt_effect(gemma_row, "delta"),
            ]
            lines.append(" & ".join(cells) + r" \\")
        if modality != MODALITY_ORDER[-1]:
            lines.append(r"\addlinespace")
    return lines


def _matched_contrast_rows(qwen: pd.DataFrame, gemma: pd.DataFrame) -> list[str]:
    """Render one row per dimension carrying both models' full rank statistics.

    Args:
        qwen: Qwen3-VL canonical matched-factorial contrast statistics.
        gemma: Gemma4 canonical matched-factorial contrast statistics.

    Returns:
        The body lines of the contrast table.
    """
    return [
        " & ".join(
            [
                DIM_LABELS[dimension],
                *_table_cells(_lookup(qwen, dimension=dimension), "mean_contrast"),
                *_table_cells(_lookup(gemma, dimension=dimension), "mean_contrast"),
            ]
        )
        + r" \\"
        for dimension in DIM_ORDER
    ]


def matched_factorial_latex_table(qwen: pd.DataFrame, gemma: pd.DataFrame) -> str:
    """Build the matched effect table as a complete ``table*``.

    Args:
        qwen: Qwen3-VL matched-factorial statistics.
        gemma: Gemma4 matched-factorial statistics.

    Returns:
        The marked LaTeX block, ready to paste into ``paper.tex``.
    """
    frames = [
        _canonical_frame(qwen, MATCHED_FACTORIAL_COLUMNS, "Qwen matched factorial"),
        _canonical_frame(gemma, MATCHED_FACTORIAL_COLUMNS, "Gemma matched factorial"),
    ]
    p_bh = _require_degenerate_ranks(frames)
    n_images = int(frames[0]["n_images"].iloc[0])

    header = [
        r"& & &",
        r"\multicolumn{2}{c}{Mean $\Delta^{\mathrm{match}}$",
        r"[95\% BCa CI]} \\",
        r"\cmidrule(lr){4-5}",
        r"Modality & Dimension & Pairs & Qwen3-VL & Gemma4 \\",
    ]
    caption = [
        r"Matched-factorial contrasts for both MLLMs. Each row averages over",
        r"profile pairs that agree on the other three persona attributes and",
        r"differ only on the stated dimension. For each matched pair, the",
        r"contrast compares the average within-profile similarity of the two",
        r"profiles with their between-profile similarity. \emph{Pairs} denotes",
        rf"the number of matched profile pairs. All {n_images} image-level contrasts are",
        rf"positive in every comparison, yielding $n_{{+}}={n_images}$, $W=0$,",
        r"$r_{\mathrm{rb}}=1.000$, and",
        rf"$p_{{\mathrm{{BH}}}}={p_bh.strip('$')}$ throughout; these repeated",
        r"statistics are therefore omitted from the table. Exact sign tests also",
        r"remain significant after Benjamini--Hochberg correction. Caption and",
        r"justification values use cosine similarity, whereas perception values",
        r"use Jaccard similarity and should be interpreted separately.",
        *_dagger_note(frames),
    ]
    return _wrap_table(
        header,
        _matched_delta_rows(*frames),
        caption,
        begin=MATCHED_TABLE_BEGIN,
        end=MATCHED_TABLE_END,
        column_spec=r"@{}llrcc@{}",
        tabcolsep="6pt",
        label=MATCHED_TABLE_LABEL,
    )


def matched_contrast_latex_table(qwen: pd.DataFrame, gemma: pd.DataFrame) -> str:
    """Build the matched justification-minus-caption table as a complete ``table*``.

    Args:
        qwen: Qwen3-VL matched-factorial contrast statistics.
        gemma: Gemma4 matched-factorial contrast statistics.

    Returns:
        The marked LaTeX block, ready to paste into ``paper.tex``.
    """
    frames = [
        _canonical_frame(qwen, MATCHED_CONTRAST_COLUMNS, "Qwen matched contrast"),
        _canonical_frame(gemma, MATCHED_CONTRAST_COLUMNS, "Gemma matched contrast"),
    ]
    n_images = int(frames[0]["n_images"].iloc[0])

    model_columns = (
        r" & Mean $\Gamma$ [95\% BCa CI] & $n_{+}$ & $W$"
        r" & $r_{\mathrm{rb}}$ & $p_{\mathrm{BH}}$"
    )
    header = [
        (
            r"& \multicolumn{5}{c}{\textbf{Qwen3-VL}}"
            r" & \multicolumn{5}{c}{\textbf{Gemma4}} \\"
        ),
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
        rf"Dimension{model_columns}{model_columns} \\",
    ]
    caption = [
        r"Matched-factorial justification-minus-caption contrasts",
        r"$\Gamma_{i,d}=\Delta^{\mathrm{just}}_{i,d}-\Delta^{\mathrm{cap}}_{i,d}$,",
        rf"formed from the two cosine terms of Table~\ref{{{MATCHED_TABLE_LABEL}}}.",
        r"Positive values indicate that changing the target persona attribute",
        r"produces a larger reduction in justification similarity than in caption",
        rf"similarity. Each test uses the {n_images} image-level contrasts: a two-sided",
        r"Wilcoxon signed-rank test, a $95\%$ BCa bootstrap interval over images,",
        r"and Benjamini--Hochberg correction across the four dimensions within each",
        r"model. $n_{+}$ is the number of images with a positive contrast, $W$ the",
        r"smaller signed-rank sum, and $r_{\mathrm{rb}}$ the matched-pairs",
        r"rank-biserial correlation. Exact sign tests are the sensitivity check and",
        r"remain significant after correction. Perception is excluded because",
        r"Jaccard and cosine effects are not commensurate.",
        *_dagger_note(frames),
    ]
    return _wrap_table(
        header,
        _matched_contrast_rows(*frames),
        caption,
        begin=MATCHED_CONTRAST_TABLE_BEGIN,
        end=MATCHED_CONTRAST_TABLE_END,
        column_spec=r"@{}l rrrrr @{\hspace{10pt}} rrrrr@{}",
        tabcolsep="4pt",
        label=MATCHED_CONTRAST_TABLE_LABEL,
        resize=True,
    )


def write_matched_factorial_table(path: Path | None = None) -> Path:
    """Read both models' matched-factorial CSVs and write the effect table.

    Args:
        path: Destination, defaulting to ``outputs/matched_factorial_table.tex``.

    Returns:
        The path written.
    """
    tex = matched_factorial_latex_table(*_paper_model_frames("matched_factorial_significance.csv"))
    path = path or PROJECT_ROOT / "outputs" / "matched_factorial_table.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")
    return path


def write_matched_contrast_table(path: Path | None = None) -> Path:
    """Read both models' matched contrast CSVs and write the contrast table.

    Args:
        path: Destination, defaulting to
            ``outputs/matched_factorial_contrast_table.tex``.

    Returns:
        The path written.
    """
    tex = matched_contrast_latex_table(
        *_paper_model_frames("matched_factorial_modality_contrast.csv")
    )
    path = path or PROJECT_ROOT / "outputs" / "matched_factorial_contrast_table.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")
    return path

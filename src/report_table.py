"""Render the Appendix E table and verify it against paper.tex."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT
from .report_format import (
    _canonical_frame,
    _ci_spans_zero,
    _lookup,
    _table_cells,
)
from .report_schema import (
    COMBINED_TABLE_LABEL,
    CONTRAST_COLUMNS,
    DIM_LABELS,
    DIM_ORDER,
    MODALITY_LABELS,
    MODALITY_ORDER,
    PAPER_MODELS,
    TABLE_BEGIN,
    TABLE_END,
    WITHIN_COLUMNS,
)


def combined_latex_table(
    qwen_within: pd.DataFrame,
    qwen_contrast: pd.DataFrame,
    gemma_within: pd.DataFrame,
    gemma_contrast: pd.DataFrame,
) -> str:
    """Build the one two-model appendix table as a complete ``table*``."""
    qwen_within = _canonical_frame(qwen_within, WITHIN_COLUMNS, "Qwen within/cross")
    qwen_contrast = _canonical_frame(qwen_contrast, CONTRAST_COLUMNS, "Qwen modality contrast")
    gemma_within = _canonical_frame(gemma_within, WITHIN_COLUMNS, "Gemma within/cross")
    gemma_contrast = _canonical_frame(gemma_contrast, CONTRAST_COLUMNS, "Gemma modality contrast")

    if set(qwen_contrast["target"]) != {"justification"}:
        raise ValueError("Qwen contrast CSV must contain justification only")
    if set(gemma_contrast["target"]) != {"justification"}:
        raise ValueError("Gemma contrast CSV must contain justification only")

    lines = [
        TABLE_BEGIN,
        r"\begin{table*}[tbh]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}ll rrrrr @{\hspace{14pt}} rrrrr@{}}",
        r"\toprule",
        (
            r"& & \multicolumn{5}{c}{\textbf{Qwen3-VL}}"
            r" & \multicolumn{5}{c}{\textbf{Gemma4}} \\"
        ),
        r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}",
        (
            r"Modality & Dimension"
            r" & Mean $\Delta$ [95\% BCa CI] & $n_{+}$ & $W$ & $r_{\mathrm{rb}}$"
            r" & $p_{\mathrm{BH}}$"
            r" & Mean $\Delta$ [95\% BCa CI] & $n_{+}$ & $W$ & $r_{\mathrm{rb}}$"
            r" & $p_{\mathrm{BH}}$ \\"
        ),
        r"\midrule",
        (
            r"\multicolumn{12}{@{}l}{\textbf{Panel A.}\enspace\textit{Within-group "
            r"minus cross-group similarity}} \\"
        ),
        r"\addlinespace[2pt]",
    ]

    for modality in MODALITY_ORDER:
        for position, dimension in enumerate(DIM_ORDER):
            qwen = _lookup(qwen_within, dimension=dimension, modality=modality)
            gemma = _lookup(gemma_within, dimension=dimension, modality=modality)
            label = (
                rf"\multirow{{{len(DIM_ORDER)}}}{{*}}"
                rf"{{{MODALITY_LABELS[modality]}}}"
                if position == 0
                else ""
            )
            row = [
                label,
                DIM_LABELS[dimension],
                *_table_cells(qwen, "delta"),
                *_table_cells(gemma, "delta"),
            ]
            lines.append(" & ".join(row) + r" \\")
        if modality != MODALITY_ORDER[-1]:
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\midrule",
            (
                r"\multicolumn{12}{@{}l}{\textbf{Panel B.}\enspace\textit{"
                r"Justification minus caption within--cross difference}} \\"
            ),
            r"\addlinespace[2pt]",
        ]
    )
    for position, dimension in enumerate(DIM_ORDER):
        qwen = _lookup(qwen_contrast, dimension=dimension)
        gemma = _lookup(gemma_contrast, dimension=dimension)
        label = (
            rf"\multirow{{{len(DIM_ORDER)}}}{{*}}"
            r"{Justification $-$ caption}"
            if position == 0
            else ""
        )
        row = [
            label,
            DIM_LABELS[dimension],
            *_table_cells(qwen, "mean_contrast"),
            *_table_cells(gemma, "mean_contrast"),
        ]
        lines.append(" & ".join(row) + r" \\")

    flagged = any(
        frame.apply(_ci_spans_zero, axis=1).any()
        for frame in (qwen_within, qwen_contrast, gemma_within, gemma_contrast)
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\caption{",
            r"Paired image-level analysis for both MLLMs. Panel A reports the mean",
            r"within-minus-cross difference $\Delta$, and Panel B reports the",
            r"justification-minus-caption contrast. $n_{+}$ is the number of images",
            r"with a positive paired difference, $W$ is the smaller signed-rank sum,",
            r"and $r_{\mathrm{rb}}$ is the matched-pairs rank-biserial correlation.",
            r"The displayed $p$-values are two-sided Wilcoxon values adjusted using",
            r"Benjamini--Hochberg within each model and panel",
            r"(12 comparisons in Panel A and four in Panel B). Confidence intervals",
        ]
    )
    if flagged:
        lines.extend(
            [
                r"are $95\%$ BCa bootstrap intervals over images;",
                r"$\dagger$ marks an interval that includes zero.",
            ]
        )
    else:
        lines.append(r"are $95\%$ BCa bootstrap intervals over images.")
    lines.extend(
        [
            r"}",
            rf"\label{{{COMBINED_TABLE_LABEL}}}",
            r"\end{table*}",
            TABLE_END,
        ]
    )
    return "\n".join(lines) + "\n"


def write_combined_table(path: Path | None = None) -> Path:
    """Read both models' canonical CSVs and write the combined table."""
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for model in PAPER_MODELS:
        model_dir = PROJECT_ROOT / "outputs" / model
        frames[model] = (
            pd.read_csv(model_dir / "within_cross_significance.csv"),
            pd.read_csv(model_dir / "within_cross_modality_contrast.csv"),
        )

    qwen_within, qwen_contrast = frames["qwen-vl"]
    gemma_within, gemma_contrast = frames["gemma-4-E4B-it_t01"]
    tex = combined_latex_table(qwen_within, qwen_contrast, gemma_within, gemma_contrast)
    path = path or PROJECT_ROOT / "outputs" / "within_cross_combined_table.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")
    return path


def _extract_generated_table(
    text: str,
    source: Path,
    begin: str = TABLE_BEGIN,
    end: str = TABLE_END,
) -> str:
    """Extract one marked table block, including its marker lines.

    The markers are parameters because ``paper.tex`` carries more than one
    generated block; each pair is counted independently.
    """
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"{source} must contain exactly one {begin!r}/{end!r} block")
    start = text.index(begin)
    stop = text.index(end, start) + len(end)
    return text[start:stop]


def verify_paper_table(
    paper_path: Path | None = None,
    generated_path: Path | None = None,
    begin: str = TABLE_BEGIN,
    end: str = TABLE_END,
) -> None:
    """Raise if the marked inline paper table differs from the generated table."""
    paper_path = paper_path or PROJECT_ROOT / "paper.tex"
    generated_path = generated_path or PROJECT_ROOT / "outputs" / "within_cross_combined_table.tex"
    paper_block = _extract_generated_table(
        paper_path.read_text(encoding="utf-8"), paper_path, begin, end
    )
    generated_block = _extract_generated_table(
        generated_path.read_text(encoding="utf-8"), generated_path, begin, end
    )
    if paper_block != generated_block:
        raise ValueError(
            f"The block marked {begin!r} in {_display_path(paper_path)} does "
            f"not exactly match {_display_path(generated_path)}"
        )


def _display_path(path: Path) -> str:
    """Render a path relative to the repository when it lies inside it.

    Args:
        path: Path to render.

    Returns:
        Repo-relative path, or the path unchanged when it lies outside.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

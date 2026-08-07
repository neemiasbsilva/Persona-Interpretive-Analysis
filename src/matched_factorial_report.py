"""Generate the matched-factorial artifacts and the Appendix F table.

Run as ``PERSONA_MODEL=<model> uv run python -m src.matched_factorial_report``.
Pass ``--check-paper`` to fail when the generated table and the marked block in
``paper.tex`` disagree; that check needs both paper models to have been processed.

This is the secondary analysis. It varies one persona attribute at a time while
holding the other three fixed, and reuses the marginal analysis' image-level
inference unchanged. The marginal report remains the primary result and must
have been produced first, because the reported ratio is taken against it.

Like ``src.similarity_report``, the statistics are always computed from the
per-image CSV *as written to disk* rather than from the in-memory frame. A CSV
round trip is not bit-exact for float64, so reading it back is what makes a
fresh run and a cache hit produce byte-identical statistics, and what makes the
published numbers reproducible from the committed artifact alone.

The table symbols are re-exported here because ``src.matched_factorial_report``
is the import path the regression tests and the paper contract use.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PATH, MODEL, OUTPUTS
from .data_loading import load_annotations, parse_demographics
from .embeddings import load_or_encode_with_ids
from .report_format import _canonical_frame
from .report_manifest import _write_matched_factorial_manifest
from .report_matched_table import (
    matched_contrast_latex_table,
    matched_factorial_latex_table,
    write_matched_contrast_table,
    write_matched_factorial_table,
)
from .report_schema import (
    MATCHED_CONTRAST_COLUMNS,
    MATCHED_CONTRAST_TABLE_BEGIN,
    MATCHED_CONTRAST_TABLE_END,
    MATCHED_FACTORIAL_COLUMNS,
    MATCHED_TABLE_BEGIN,
    MATCHED_TABLE_END,
    N_BOOT,
    SEED,
    WILCOXON_RESAMPLES,
)
from .report_table import verify_paper_table
from .similarity import (
    MATCHED_FACTORIAL_CONTRAST_SEED_NAMESPACE,
    MATCHED_FACTORIAL_SEED_NAMESPACE,
    compute_matched_factorial_similarity,
    run_modality_contrast,
    run_within_cross_significance,
    validate_within_cross_cache_metadata,
)

__all__ = [
    "MATCHED_CONTRAST_TABLE_BEGIN",
    "MATCHED_CONTRAST_TABLE_END",
    "MATCHED_TABLE_BEGIN",
    "MATCHED_TABLE_END",
    "main",
    "matched_contrast_latex_table",
    "matched_factorial_latex_table",
    "write_matched_contrast_table",
    "write_matched_factorial_table",
]


def _matched_pairs_per_cell(source: pd.DataFrame) -> pd.DataFrame:
    """Reduce the per-image pair counts to one constant per dimension and modality.

    Args:
        source: The per-image matched similarity frame.

    Returns:
        One row per dimension and modality carrying ``n_matched_pairs``.

    Raises:
        ValueError: If a cell's pair count varies across images.
    """
    grouped = source.groupby(["dimension", "modality"], sort=False)["n_matched_pairs"]
    spread = grouped.agg(["min", "max"]).reset_index()
    varying = spread[spread["min"] != spread["max"]]
    if not varying.empty:
        cells = varying[["dimension", "modality"]].to_dict("records")
        raise ValueError(f"matched profile-pair counts vary across images for: {cells}")
    return spread.rename(columns={"min": "n_matched_pairs"}).drop(columns="max")


def _matched_pairs_by_dimension(pairs_per_cell: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-cell pair counts to one constant per dimension.

    The justification-minus-caption contrast differences two matched means, so
    they have to be averages over the same profile pairs. Nothing downstream
    would notice if they were not, hence the check here rather than a
    ``drop_duplicates`` that silently keeps the first row.

    Args:
        pairs_per_cell: One row per dimension and modality carrying ``n_matched_pairs``.

    Returns:
        One row per dimension carrying ``n_matched_pairs``.

    Raises:
        ValueError: If a dimension's pair count varies across modalities.
    """
    spread = pairs_per_cell.groupby("dimension", sort=False)["n_matched_pairs"].agg(["min", "max"])
    varying = spread[spread["min"] != spread["max"]]
    if not varying.empty:
        raise ValueError(
            f"matched profile-pair counts vary across modalities for: {sorted(varying.index)}"
        )
    return spread.rename(columns={"min": "n_matched_pairs"}).drop(columns="max").reset_index()


def _refresh_paper_table(
    write: Callable[[], Path],
    *,
    begin: str,
    end: str,
    description: str,
    check_paper: bool,
) -> None:
    """Regenerate one marked table and compare it against ``paper.tex``.

    Both failures are deferred by default, because a single-model run cannot
    produce a two-model table and the deferral is what lets the pipeline be run
    one model at a time.

    Args:
        write: Callable that regenerates the table and returns its path.
        begin: Opening marker comment of the block in ``paper.tex``.
        end: Closing marker comment of the block in ``paper.tex``.
        description: Human-readable table name for the deferral messages.
        check_paper: Fail rather than defer.

    Raises:
        FileNotFoundError: Under ``check_paper``, if an input CSV is absent.
        ValueError: Under ``check_paper``, if the marked block disagrees.
    """
    try:
        table_path = write()
    except (FileNotFoundError, ValueError) as error:
        if check_paper:
            raise
        print(
            f"{description} not refreshed yet; run the report for both paper "
            f"models with the canonical code ({error})."
        )
        return
    print(f"Wrote {table_path}")
    try:
        verify_paper_table(generated_path=table_path, begin=begin, end=end)
    except ValueError as error:
        if check_paper:
            raise
        print(f"Paper {description} check deferred until both model reports are final: {error}")
    else:
        print(f"Verified the marked paper.tex {description} against the CSVs")


def _attach_marginal_ratio(stats: pd.DataFrame) -> pd.DataFrame:
    """Join the marginal effect so the reported ratio stays traceable to a CSV.

    Args:
        stats: The matched-factorial statistics frame.

    Returns:
        The frame with ``marginal_delta`` and ``matched_to_marginal_ratio`` added.

    Raises:
        FileNotFoundError: If the marginal report has not been produced yet.
        ValueError: If a marginal effect is missing, non-finite, or zero.
    """
    marginal_path = OUTPUTS / "within_cross_significance.csv"
    if not marginal_path.exists():
        raise FileNotFoundError(
            f"{marginal_path} not found; run python -m src.similarity_report first"
        )
    marginal = pd.read_csv(marginal_path)[["dimension", "modality", "delta"]].rename(
        columns={"delta": "marginal_delta"}
    )
    merged = stats.merge(
        marginal,
        on=["dimension", "modality"],
        how="left",
        validate="one_to_one",
    )
    values = merged["marginal_delta"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values == 0.0).any():
        raise ValueError("every marginal effect must be finite and non-zero to form a ratio")
    merged["matched_to_marginal_ratio"] = merged["delta"] / merged["marginal_delta"]
    return merged


def main(*, check_paper: bool = False) -> None:
    """Compute, write, and report the matched-factorial statistics for one model.

    Args:
        check_paper: Fail rather than defer when the marked block in ``paper.tex``
            does not match the generated table.
    """
    annotations = parse_demographics(load_annotations(DATA_PATH))
    cap_embs, id_index = load_or_encode_with_ids(
        annotations,
        "caption",
        OUTPUTS / "caption_embeddings.npy",
        OUTPUTS / "caption_embeddings_ids.csv",
    )
    just_embs, _ = load_or_encode_with_ids(
        annotations,
        "justification",
        OUTPUTS / "justification_embeddings.npy",
        OUTPUTS / "caption_embeddings_ids.csv",
    )

    source_path = OUTPUTS / "matched_factorial_persona_sim.csv"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    compute_matched_factorial_similarity(
        annotations,
        cap_embs,
        just_embs,
        id_index,
        cache_path=source_path,
    )
    source_metadata = validate_within_cross_cache_metadata(source_path)
    source = pd.read_csv(source_path)

    stats = run_within_cross_significance(
        wc_df=source,
        n_boot=N_BOOT,
        seed=SEED,
        wilcoxon_resamples=WILCOXON_RESAMPLES,
        seed_namespace=MATCHED_FACTORIAL_SEED_NAMESPACE,
    )
    pairs_per_cell = _matched_pairs_per_cell(source)
    stats = stats.merge(pairs_per_cell, on=["dimension", "modality"], how="left")
    stats = _attach_marginal_ratio(stats)
    stats = _canonical_frame(stats, MATCHED_FACTORIAL_COLUMNS, "matched-factorial report")

    pairs_by_dimension = _matched_pairs_by_dimension(pairs_per_cell)
    contrast = run_modality_contrast(
        wc_df=source,
        targets=("justification",),
        n_boot=N_BOOT,
        seed=SEED,
        wilcoxon_resamples=WILCOXON_RESAMPLES,
        seed_namespace=MATCHED_FACTORIAL_CONTRAST_SEED_NAMESPACE,
    )
    contrast = contrast.merge(pairs_by_dimension, on="dimension", how="left")
    contrast = _canonical_frame(
        contrast, MATCHED_CONTRAST_COLUMNS, "matched-factorial contrast report"
    )

    stats_path = OUTPUTS / "matched_factorial_significance.csv"
    contrast_path = OUTPUTS / "matched_factorial_modality_contrast.csv"
    manifest_path = OUTPUTS / "matched_factorial_method.json"
    stats.to_csv(stats_path, index=False)
    contrast.to_csv(contrast_path, index=False)
    _write_matched_factorial_manifest(
        manifest_path,
        source_path,
        source_metadata=source_metadata,
        stats=stats,
        contrast=contrast,
        matched_pairs_per_dimension={
            str(dimension): int(count)
            for dimension, count in zip(
                pairs_by_dimension["dimension"],
                pairs_by_dimension["n_matched_pairs"],
                strict=True,
            )
        },
    )

    pd.set_option("display.width", 240)
    print(f"Model: {MODEL}; {int(stats.n_images.iloc[0])} images")
    print("\nMatched-factorial within minus cross (BH family: 12 tests):")
    print(stats.to_string(index=False))
    print("\nMatched-factorial justification minus caption (BH family: 4 tests):")
    print(contrast.to_string(index=False))
    print(f"\nWrote {source_path}")
    print(f"Wrote {stats_path}")
    print(f"Wrote {contrast_path}")
    print(f"Wrote {manifest_path}")

    _refresh_paper_table(
        write_matched_factorial_table,
        begin=MATCHED_TABLE_BEGIN,
        end=MATCHED_TABLE_END,
        description="matched-factorial table",
        check_paper=check_paper,
    )
    _refresh_paper_table(
        write_matched_contrast_table,
        begin=MATCHED_CONTRAST_TABLE_BEGIN,
        end=MATCHED_CONTRAST_TABLE_END,
        description="matched-factorial contrast table",
        check_paper=check_paper,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-paper",
        action="store_true",
        help=(
            "fail unless the marked matched-factorial table in paper.tex exactly "
            "matches the two models' canonical CSVs"
        ),
    )
    args = parser.parse_args()
    main(check_paper=args.check_paper)

"""Generate the canonical paired-statistics artifacts and the Appendix E table.

Run as ``PERSONA_MODEL=<model> uv run python -m src.similarity_report``. Pass
``--check-paper`` to fail when the generated table and the marked block in
``paper.tex`` disagree; that check needs both paper models to have been processed.

The table symbols are re-exported here because ``src.similarity_report`` is the
import path the regression tests and the paper contract use.
"""

from __future__ import annotations

import argparse

import pandas as pd

from .config import DATA_PATH, MODEL, OUTPUTS
from .data_loading import load_annotations
from .report_format import _canonical_frame
from .report_manifest import _write_method_manifest
from .report_schema import (
    CONTRAST_COLUMNS,
    EXPECTED_AGENTS_PER_IMAGE,
    N_BOOT,
    SEED,
    SENSITIVITY_COLUMNS,
    TABLE_BEGIN,
    TABLE_END,
    WILCOXON_RESAMPLES,
    WITHIN_COLUMNS,
)
from .report_table import combined_latex_table, verify_paper_table, write_combined_table
from .similarity import (
    run_modality_contrast,
    run_within_cross_significance,
    validate_within_cross_cache_metadata,
)

__all__ = [
    "TABLE_BEGIN",
    "TABLE_END",
    "combined_latex_table",
    "main",
    "verify_paper_table",
    "write_combined_table",
]


def main(*, check_paper: bool = False) -> None:
    """Regenerate the within/cross statistics, CSVs and LaTeX table for one model.

    Args:
        check_paper: Fail rather than defer when the marked block in ``paper.tex``
            does not match the generated table.
    """
    source_path = OUTPUTS / "within_cross_persona_sim.csv"
    source_metadata = validate_within_cross_cache_metadata(source_path)
    source = pd.read_csv(source_path)
    stats = run_within_cross_significance(
        wc_df=source,
        n_boot=N_BOOT,
        seed=SEED,
        wilcoxon_resamples=WILCOXON_RESAMPLES,
    )
    contrast = run_modality_contrast(
        wc_df=source,
        targets=("justification",),
        n_boot=N_BOOT,
        seed=SEED,
        wilcoxon_resamples=WILCOXON_RESAMPLES,
    )
    annotations = load_annotations(DATA_PATH)
    counts = annotations.groupby("image_id", sort=False).size()
    complete_images = counts[counts.eq(EXPECTED_AGENTS_PER_IMAGE)].index
    if len(complete_images) < 2:
        raise ValueError(
            "complete-image sensitivity requires at least two images with "
            f"{EXPECTED_AGENTS_PER_IMAGE} generated agents"
        )
    complete_source = source[source["image_id"].isin(complete_images)]
    complete_image_sensitivity = run_modality_contrast(
        wc_df=complete_source,
        targets=("justification",),
        n_boot=N_BOOT,
        seed=SEED,
        wilcoxon_resamples=WILCOXON_RESAMPLES,
    )
    complete_image_sensitivity.insert(0, "expected_agents_per_image", EXPECTED_AGENTS_PER_IMAGE)
    complete_image_sensitivity.insert(0, "selection", "complete_generated_agent_panel")

    stats = _canonical_frame(stats, WITHIN_COLUMNS, "within/cross report")
    contrast = _canonical_frame(contrast, CONTRAST_COLUMNS, "justification-minus-caption report")
    complete_image_sensitivity = _canonical_frame(
        complete_image_sensitivity,
        SENSITIVITY_COLUMNS,
        "complete-image sensitivity report",
    )

    stats_path = OUTPUTS / "within_cross_significance.csv"
    contrast_path = OUTPUTS / "within_cross_modality_contrast.csv"
    complete_path = OUTPUTS / "within_cross_complete_image_sensitivity.csv"
    manifest_path = OUTPUTS / "within_cross_method.json"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    stats.to_csv(stats_path, index=False)
    contrast.to_csv(contrast_path, index=False)
    complete_image_sensitivity.to_csv(complete_path, index=False)
    _write_method_manifest(
        manifest_path,
        source_path,
        source_metadata,
        stats=stats,
        contrast=contrast,
        complete_image_sensitivity=complete_image_sensitivity,
    )

    pd.set_option("display.width", 240)
    print(f"Model: {MODEL}; {int(stats.n_images.iloc[0])} images")
    print("\nWithin minus cross (BH family: 12 tests):")
    print(stats.to_string(index=False))
    print("\nJustification minus caption (BH family: 4 tests):")
    print(contrast.to_string(index=False))
    print("\nComplete-image sensitivity (BH family: 4 tests):")
    print(complete_image_sensitivity.to_string(index=False))

    print(f"\nWrote {stats_path}")
    print(f"Wrote {contrast_path}")
    print(f"Wrote {complete_path}")
    print(f"Wrote {manifest_path}")

    try:
        combined_path = write_combined_table()
    except (FileNotFoundError, ValueError) as error:
        if check_paper:
            raise
        print(
            "Combined table not refreshed yet; run the report for both paper "
            f"models with the canonical code ({error})."
        )
    else:
        print(f"Wrote {combined_path}")
        try:
            verify_paper_table(generated_path=combined_path)
        except ValueError as error:
            if check_paper:
                raise
            print(f"Paper table check deferred until both model reports are final: {error}")
        else:
            print("Verified the marked paper.tex table against the canonical CSVs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-paper",
        action="store_true",
        help=(
            "fail unless the marked inline table in paper.tex exactly matches "
            "the two models' canonical CSVs"
        ),
    )
    args = parser.parse_args()
    main(check_paper=args.check_paper)

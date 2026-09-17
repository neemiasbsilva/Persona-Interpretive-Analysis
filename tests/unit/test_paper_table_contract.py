"""Verify both generated paper tables against committed fixtures.

These run on a bare clone. The 46 MB corpora are not needed: the canonical CSVs
per model are small enough to commit under ``tests/golden``, so a change that
would alter a published table fails here rather than only in the integration
suite. ``paper.tex`` carries three independently marked blocks -- Appendix E
(marginal), Appendix F (matched-factorial effects) and Appendix F (matched
justification-minus-caption) -- and all three contracts are pinned.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.matched_factorial_report import (
    matched_contrast_latex_table,
    matched_factorial_latex_table,
)
from src.models import PAPER_MODELS
from src.report_schema import (
    COMBINED_TABLE_LABEL,
    MATCHED_CONTRAST_TABLE_BEGIN,
    MATCHED_CONTRAST_TABLE_END,
    MATCHED_CONTRAST_TABLE_LABEL,
    MATCHED_TABLE_BEGIN,
    MATCHED_TABLE_END,
    MATCHED_TABLE_LABEL,
    RETIRED_TABLE_LABELS,
)
from src.report_table import _extract_generated_table
from src.similarity import (
    benjamini_hochberg,
    run_modality_contrast,
    run_within_cross_significance,
)
from src.similarity_report import (
    TABLE_BEGIN,
    TABLE_END,
    combined_latex_table,
    verify_paper_table,
)

MARKER_PAIRS = (
    (TABLE_BEGIN, TABLE_END),
    (MATCHED_TABLE_BEGIN, MATCHED_TABLE_END),
    (MATCHED_CONTRAST_TABLE_BEGIN, MATCHED_CONTRAST_TABLE_END),
)


def _golden(golden_dir: Path, model: str, name: str) -> pd.DataFrame:
    return pd.read_csv(golden_dir / model / name)


@pytest.fixture
def canonical_table(golden_dir: Path) -> str:
    frames = [
        _golden(golden_dir, model, name)
        for model in PAPER_MODELS
        for name in ("within_cross_significance.csv", "within_cross_modality_contrast.csv")
    ]
    return combined_latex_table(*frames)


@pytest.fixture
def matched_table(golden_dir: Path) -> str:
    frames = [
        _golden(golden_dir, model, "matched_factorial_significance.csv") for model in PAPER_MODELS
    ]
    return matched_factorial_latex_table(*frames)


@pytest.fixture
def matched_contrast_table(golden_dir: Path) -> str:
    frames = [
        _golden(golden_dir, model, "matched_factorial_modality_contrast.csv")
        for model in PAPER_MODELS
    ]
    return matched_contrast_latex_table(*frames)


def test_generated_table_matches_the_committed_artifact(
    canonical_table: str, golden_dir: Path
) -> None:
    expected = (golden_dir / "within_cross_combined_table.tex").read_text(encoding="utf-8")
    assert canonical_table == expected


def test_table_is_delimited_by_exactly_one_marker_pair(canonical_table: str) -> None:
    assert canonical_table.count(TABLE_BEGIN) == 1
    assert canonical_table.count(TABLE_END) == 1
    assert canonical_table.startswith(TABLE_BEGIN)
    assert canonical_table.rstrip().endswith(TABLE_END)


def test_table_carries_one_label_and_no_retired_labels(canonical_table: str) -> None:
    assert canonical_table.count(rf"\label{{{COMBINED_TABLE_LABEL}}}") == 1
    for retired in RETIRED_TABLE_LABELS:
        assert rf"\label{{{retired}}}" not in canonical_table


@pytest.mark.parametrize("model", PAPER_MODELS)
def test_canonical_frames_have_the_agreed_shape(golden_dir: Path, model: str) -> None:
    within = _golden(golden_dir, model, "within_cross_significance.csv")
    contrast = _golden(golden_dir, model, "within_cross_modality_contrast.csv")
    complete = _golden(golden_dir, model, "within_cross_complete_image_sensitivity.csv")

    assert len(within) == 12
    assert len(contrast) == 4
    assert len(complete) == 4
    assert set(contrast["target"]) == {"justification"}
    assert set(complete["target"]) == {"justification"}


@pytest.mark.parametrize("model", PAPER_MODELS)
def test_retired_columns_never_reappear(golden_dir: Path, model: str) -> None:
    within = _golden(golden_dir, model, "within_cross_significance.csv")
    assert not {"p_perm", "p_perm_bh", "negligible", "p_bh"}.intersection(within.columns)


@pytest.mark.parametrize("model", PAPER_MODELS)
def test_source_frame_is_one_row_per_image_dimension_and_modality(
    golden_dir: Path, model: str
) -> None:
    source = pd.read_csv(golden_dir / model / "within_cross_persona_sim.csv")
    assert source.shape == (600, 5)
    assert source["image_id"].nunique() == 50
    assert not source.duplicated(["image_id", "dimension", "modality"]).any()
    assert np.isfinite(source[["within_mean", "cross_mean"]]).all(axis=None)


@pytest.mark.parametrize("model", PAPER_MODELS)
def test_deterministic_rank_quantities_reproduce_from_the_source(
    golden_dir: Path, model: str
) -> None:
    source = pd.read_csv(golden_dir / model / "within_cross_persona_sim.csv")
    observed = run_within_cross_significance(
        source, n_boot=99, wilcoxon_resamples=999, seed=42
    ).sort_values(["dimension", "modality"])
    canonical = _golden(golden_dir, model, "within_cross_significance.csv").sort_values(
        ["dimension", "modality"]
    )

    for column in ("delta", "n_pos", "w_stat", "p_wilcoxon", "rank_biserial", "p_wilcoxon_bh"):
        np.testing.assert_allclose(
            observed[column].to_numpy(),
            canonical[column].to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )


def test_bh_correction_matches_known_example() -> None:
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.002]))
    np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.04, 0.008])


def test_generated_matched_table_matches_the_committed_artifact(
    matched_table: str, golden_dir: Path
) -> None:
    expected = (golden_dir / "matched_factorial_table.tex").read_text(encoding="utf-8")
    assert matched_table == expected


def test_matched_table_is_delimited_by_exactly_one_marker_pair(matched_table: str) -> None:
    assert matched_table.count(MATCHED_TABLE_BEGIN) == 1
    assert matched_table.count(MATCHED_TABLE_END) == 1
    assert matched_table.startswith(MATCHED_TABLE_BEGIN)
    assert matched_table.rstrip().endswith(MATCHED_TABLE_END)


def test_matched_table_carries_its_own_label(matched_table: str) -> None:
    assert matched_table.count(rf"\label{{{MATCHED_TABLE_LABEL}}}") == 1
    assert rf"\label{{{COMBINED_TABLE_LABEL}}}" not in matched_table
    for retired in RETIRED_TABLE_LABELS:
        assert rf"\label{{{retired}}}" not in matched_table


def test_generated_matched_contrast_table_matches_the_committed_artifact(
    matched_contrast_table: str, golden_dir: Path
) -> None:
    expected = (golden_dir / "matched_factorial_contrast_table.tex").read_text(encoding="utf-8")
    assert matched_contrast_table == expected


def test_matched_contrast_table_is_delimited_by_exactly_one_marker_pair(
    matched_contrast_table: str,
) -> None:
    assert matched_contrast_table.count(MATCHED_CONTRAST_TABLE_BEGIN) == 1
    assert matched_contrast_table.count(MATCHED_CONTRAST_TABLE_END) == 1
    assert matched_contrast_table.startswith(MATCHED_CONTRAST_TABLE_BEGIN)
    assert matched_contrast_table.rstrip().endswith(MATCHED_CONTRAST_TABLE_END)


def test_matched_contrast_table_carries_its_own_label(matched_contrast_table: str) -> None:
    assert matched_contrast_table.count(rf"\label{{{MATCHED_CONTRAST_TABLE_LABEL}}}") == 1
    assert rf"\label{{{MATCHED_TABLE_LABEL}}}" not in matched_contrast_table
    assert rf"\ref{{{MATCHED_TABLE_LABEL}}}" in matched_contrast_table
    for retired in RETIRED_TABLE_LABELS:
        assert rf"\label{{{retired}}}" not in matched_contrast_table


def test_marker_pairs_are_disjoint(
    canonical_table: str, matched_table: str, matched_contrast_table: str, tmp_path: Path
) -> None:
    for one, other in combinations(MARKER_PAIRS, 2):
        for one_marker, other_marker in zip(one, other, strict=True):
            assert one_marker not in other_marker
            assert other_marker not in one_marker

    blocks = (canonical_table, matched_table, matched_contrast_table)
    document = tmp_path / "paper.tex"
    document.write_text(
        "intro\n" + "middle\n".join(blocks) + "rest\n",
        encoding="utf-8",
    )
    text = document.read_text(encoding="utf-8")

    for (begin, end), block in zip(MARKER_PAIRS, blocks, strict=True):
        assert _extract_generated_table(text, document, begin, end) == block.rstrip("\n")


@pytest.mark.parametrize("model", PAPER_MODELS)
def test_matched_frames_have_the_agreed_shape(golden_dir: Path, model: str) -> None:
    matched = _golden(golden_dir, model, "matched_factorial_significance.csv")
    marginal = _golden(golden_dir, model, "within_cross_significance.csv")

    assert len(matched) == 12
    assert "ci_relation_to_exploratory_reference" not in matched.columns
    expected_pairs = {
        "gender": 12,
        "economic_status": 12,
        "political_spectrum": 12,
        "personality": 24,
    }
    for dimension, pairs in expected_pairs.items():
        assert set(matched.loc[matched["dimension"] == dimension, "n_matched_pairs"]) == {pairs}

    merged = matched.merge(
        marginal[["dimension", "modality", "delta"]].rename(columns={"delta": "expected"}),
        on=["dimension", "modality"],
        validate="one_to_one",
    )
    np.testing.assert_allclose(merged["marginal_delta"], merged["expected"], rtol=1e-12)
    np.testing.assert_allclose(
        merged["matched_to_marginal_ratio"],
        merged["delta"] / merged["marginal_delta"],
        rtol=1e-12,
    )


@pytest.mark.parametrize("model", PAPER_MODELS)
def test_matched_source_frame_is_one_row_per_image_dimension_and_modality(
    golden_dir: Path, model: str
) -> None:
    source = pd.read_csv(golden_dir / model / "matched_factorial_persona_sim.csv")
    assert source.shape == (600, 6)
    assert source["image_id"].nunique() == 50
    assert not source.duplicated(["image_id", "dimension", "modality"]).any()
    assert np.isfinite(source[["within_mean", "cross_mean"]]).all(axis=None)


@pytest.mark.parametrize("model", PAPER_MODELS)
def test_matched_deterministic_rank_quantities_reproduce_from_the_source(
    golden_dir: Path, model: str
) -> None:
    source = pd.read_csv(golden_dir / model / "matched_factorial_persona_sim.csv")
    observed = run_within_cross_significance(
        source,
        n_boot=99,
        wilcoxon_resamples=999,
        seed=42,
        seed_namespace="matched_factorial",
    ).sort_values(["dimension", "modality"])
    canonical = _golden(golden_dir, model, "matched_factorial_significance.csv").sort_values(
        ["dimension", "modality"]
    )

    for column in ("delta", "n_pos", "w_stat", "p_wilcoxon", "rank_biserial", "p_wilcoxon_bh"):
        np.testing.assert_allclose(
            observed[column].to_numpy(),
            canonical[column].to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )


@pytest.mark.parametrize("model", PAPER_MODELS)
def test_matched_contrast_frames_have_the_agreed_shape(golden_dir: Path, model: str) -> None:
    contrast = _golden(golden_dir, model, "matched_factorial_modality_contrast.csv")

    assert len(contrast) == 4
    assert set(contrast["target"]) == {"justification"}
    assert set(contrast["reference"]) == {"caption"}
    assert set(contrast["n_images"]) == {50}
    expected_pairs = {
        "gender": 12,
        "economic_status": 12,
        "political_spectrum": 12,
        "personality": 24,
    }
    for dimension, pairs in expected_pairs.items():
        assert set(contrast.loc[contrast["dimension"] == dimension, "n_matched_pairs"]) == {pairs}


@pytest.mark.parametrize("model", PAPER_MODELS)
def test_matched_contrast_equals_the_difference_of_the_matched_effects(
    golden_dir: Path, model: str
) -> None:
    contrast = _golden(golden_dir, model, "matched_factorial_modality_contrast.csv")
    matched = _golden(golden_dir, model, "matched_factorial_significance.csv")

    effects = matched.set_index(["dimension", "modality"])["delta"]
    expected = [
        effects[(dimension, "justification")] - effects[(dimension, "caption")]
        for dimension in contrast["dimension"]
    ]
    np.testing.assert_allclose(contrast["mean_contrast"].to_numpy(), expected, rtol=1e-9)


def test_matched_contrast_is_not_uniformly_degenerate(golden_dir: Path) -> None:
    """Qwen gender is the one cell whose rank statistics are not saturated.

    The matched effect table omits n_+, W and r_rb because they are constant;
    the contrast table tabulates them precisely because they are not. If this
    ever saturates, that column choice needs revisiting.
    """
    contrast = _golden(golden_dir, "qwen-vl", "matched_factorial_modality_contrast.csv")
    gender = contrast.loc[contrast["dimension"] == "gender"].iloc[0]
    assert int(gender["n_pos"]) < 50
    assert float(gender["w_stat"]) > 0.0
    assert float(gender["rank_biserial"]) < 1.0


@pytest.mark.parametrize("model", PAPER_MODELS)
def test_matched_contrast_rank_quantities_reproduce_from_the_source(
    golden_dir: Path, model: str
) -> None:
    source = pd.read_csv(golden_dir / model / "matched_factorial_persona_sim.csv")
    observed = run_modality_contrast(
        source,
        n_boot=99,
        wilcoxon_resamples=999,
        seed=42,
        seed_namespace="matched_factorial_contrast",
    ).sort_values("dimension")
    canonical = _golden(golden_dir, model, "matched_factorial_modality_contrast.csv").sort_values(
        "dimension"
    )

    for column in (
        "mean_contrast",
        "n_pos",
        "w_stat",
        "p_wilcoxon",
        "rank_biserial",
        "p_wilcoxon_bh",
    ):
        np.testing.assert_allclose(
            observed[column].to_numpy(),
            canonical[column].to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )


def test_verify_paper_table_reports_a_mismatch(tmp_path: Path) -> None:
    paper = tmp_path / "paper.tex"
    generated = tmp_path / "table.tex"
    paper.write_text(f"intro\n{TABLE_BEGIN}\nROW A\n{TABLE_END}\nrest\n", encoding="utf-8")
    generated.write_text(f"{TABLE_BEGIN}\nROW B\n{TABLE_END}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"paper\.tex"):
        verify_paper_table(paper_path=paper, generated_path=generated)


def test_verify_paper_table_accepts_an_exact_match(tmp_path: Path) -> None:
    block = f"{TABLE_BEGIN}\nROW A\n{TABLE_END}\n"
    paper = tmp_path / "paper.tex"
    generated = tmp_path / "table.tex"
    paper.write_text(f"intro\n{block}rest\n", encoding="utf-8")
    generated.write_text(block, encoding="utf-8")

    verify_paper_table(paper_path=paper, generated_path=generated)

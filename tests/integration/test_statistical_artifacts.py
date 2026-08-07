import json
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
from src.similarity import (
    MATCHED_FACTORIAL_CONSTRUCTION,
    benjamini_hochberg,
    run_within_cross_significance,
    validate_within_cross_cache_metadata,
)
from src.similarity_report import (
    TABLE_BEGIN,
    TABLE_END,
    combined_latex_table,
    verify_paper_table,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
MODELS = PAPER_MODELS


@pytest.mark.parametrize("model", MODELS)
def test_real_600_row_sources_reproduce_canonical_rank_results(model):
    output = ROOT / "outputs" / model
    metadata = validate_within_cross_cache_metadata(output / "within_cross_persona_sim.csv")
    assert metadata["empty_perception_policy"] == "exclude_pairs_with_empty_perception_responses"
    source = pd.read_csv(output / "within_cross_persona_sim.csv")

    assert source.shape == (600, 5)
    assert source["image_id"].nunique() == 50
    assert not source.duplicated(["image_id", "dimension", "modality"]).any()
    assert np.isfinite(source[["within_mean", "cross_mean"]]).all(axis=None)

    observed = run_within_cross_significance(
        source,
        n_boot=99,
        wilcoxon_resamples=999,
        seed=42,
    ).sort_values(["dimension", "modality"])
    canonical = pd.read_csv(output / "within_cross_significance.csv").sort_values(
        ["dimension", "modality"]
    )

    for column in (
        "delta",
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


@pytest.mark.parametrize("model", MODELS)
def test_raw_annotation_and_embedding_cache_rows_are_exactly_aligned(model):
    data_path = ROOT / "data" / model / "annotations_baseline.jsonl"
    output = ROOT / "outputs" / model
    with data_path.open(encoding="utf-8") as stream:
        raw_ids = [json.loads(line)["annotation_id"] for line in stream]
    cached_ids = pd.read_csv(output / "caption_embeddings_ids.csv")["annotation_id"].tolist()

    assert raw_ids == cached_ids
    assert len(raw_ids) == len(set(raw_ids))
    assert np.load(output / "caption_embeddings.npy", mmap_mode="r").shape[0] == len(raw_ids)
    assert np.load(output / "justification_embeddings.npy", mmap_mode="r").shape[0] == len(raw_ids)


def test_bh_correction_matches_known_example():
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.002]))
    np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.04, 0.008])


def test_canonical_table_and_paper_are_exactly_synchronized():
    qwen_output = ROOT / "outputs" / "qwen-vl"
    gemma_output = ROOT / "outputs" / "gemma-4-E4B-it_t01"
    expected_table = combined_latex_table(
        pd.read_csv(qwen_output / "within_cross_significance.csv"),
        pd.read_csv(qwen_output / "within_cross_modality_contrast.csv"),
        pd.read_csv(gemma_output / "within_cross_significance.csv"),
        pd.read_csv(gemma_output / "within_cross_modality_contrast.csv"),
    )
    assert (ROOT / "outputs" / "within_cross_combined_table.tex").read_text(
        encoding="utf-8"
    ) == expected_table
    verify_paper_table()

    paper = (ROOT / "paper.tex").read_text(encoding="utf-8")
    assert paper.count(TABLE_BEGIN) == 1
    assert paper.count(TABLE_END) == 1
    assert paper.count(rf"\label{{{COMBINED_TABLE_LABEL}}}") == 1
    for retired in RETIRED_TABLE_LABELS:
        assert rf"\label{{{retired}}}" not in paper

    for model in MODELS:
        output = ROOT / "outputs" / model
        within = pd.read_csv(output / "within_cross_significance.csv")
        contrast = pd.read_csv(output / "within_cross_modality_contrast.csv")
        complete = pd.read_csv(output / "within_cross_complete_image_sensitivity.csv")
        assert len(within) == 12
        assert len(contrast) == 4
        assert len(complete) == 4
        assert set(contrast["target"]) == {"justification"}
        assert set(complete["target"]) == {"justification"}
        expected_complete_images = 33 if model == "qwen-vl" else 49
        assert set(complete["n_images"]) == {expected_complete_images}
        assert not {
            "p_perm",
            "p_perm_bh",
            "negligible",
            "p_bh",
        }.intersection(within.columns)


def test_matched_factorial_artifacts_and_paper_are_exactly_synchronized():
    frames = []
    contrast_frames = []
    for model in MODELS:
        output = ROOT / "outputs" / model
        source = pd.read_csv(output / "matched_factorial_persona_sim.csv")
        assert source.shape == (600, 6)
        assert source["image_id"].nunique() == 50
        assert not source.duplicated(["image_id", "dimension", "modality"]).any()
        assert np.isfinite(source[["within_mean", "cross_mean"]]).all(axis=None)

        metadata = validate_within_cross_cache_metadata(
            output / "matched_factorial_persona_sim.csv"
        )
        assert metadata["construction"] == MATCHED_FACTORIAL_CONSTRUCTION

        matched = pd.read_csv(output / "matched_factorial_significance.csv")
        assert len(matched) == 12
        assert set(matched["n_pos"]) == {50}
        assert set(matched["w_stat"]) == {0.0}
        assert set(matched["rank_biserial"]) == {1.0}
        assert set(matched["wilcoxon_method"]) == {"exact"}
        assert "ci_relation_to_exploratory_reference" not in matched.columns
        frames.append(matched)

        contrast = pd.read_csv(output / "matched_factorial_modality_contrast.csv")
        assert len(contrast) == 4
        assert set(contrast["target"]) == {"justification"}
        assert set(contrast["reference"]) == {"caption"}
        assert set(contrast["n_images"]) == {50}
        assert set(contrast["wilcoxon_method"]) == {"exact"}

        effects = matched.set_index(["dimension", "modality"])["delta"]
        expected_contrast = [
            effects[(dimension, "justification")] - effects[(dimension, "caption")]
            for dimension in contrast["dimension"]
        ]
        np.testing.assert_allclose(
            contrast["mean_contrast"].to_numpy(), expected_contrast, rtol=1e-9
        )
        contrast_frames.append(contrast)

    qwen_gender = contrast_frames[0].set_index("dimension").loc["gender"]
    assert int(qwen_gender["n_pos"]) < 50
    assert float(qwen_gender["rank_biserial"]) < 1.0

    expected_table = matched_factorial_latex_table(*frames)
    generated_path = ROOT / "outputs" / "matched_factorial_table.tex"
    assert generated_path.read_text(encoding="utf-8") == expected_table
    verify_paper_table(
        generated_path=generated_path,
        begin=MATCHED_TABLE_BEGIN,
        end=MATCHED_TABLE_END,
    )

    expected_contrast_table = matched_contrast_latex_table(*contrast_frames)
    contrast_path = ROOT / "outputs" / "matched_factorial_contrast_table.tex"
    assert contrast_path.read_text(encoding="utf-8") == expected_contrast_table
    verify_paper_table(
        generated_path=contrast_path,
        begin=MATCHED_CONTRAST_TABLE_BEGIN,
        end=MATCHED_CONTRAST_TABLE_END,
    )

    paper = (ROOT / "paper.tex").read_text(encoding="utf-8")
    assert paper.count(MATCHED_TABLE_BEGIN) == 1
    assert paper.count(MATCHED_TABLE_END) == 1
    assert paper.count(rf"\label{{{MATCHED_TABLE_LABEL}}}") == 1
    assert paper.count(MATCHED_CONTRAST_TABLE_BEGIN) == 1
    assert paper.count(MATCHED_CONTRAST_TABLE_END) == 1
    assert paper.count(rf"\label{{{MATCHED_CONTRAST_TABLE_LABEL}}}") == 1

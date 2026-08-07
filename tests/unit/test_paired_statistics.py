import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src import similarity as sim
from src.embeddings import load_or_encode_with_ids


def _wc_frame(dimensions=("gender",), modalities=("caption",), n_images=8) -> pd.DataFrame:
    rows = []
    for dim_offset, dimension in enumerate(dimensions):
        for mod_offset, modality in enumerate(modalities):
            for image in range(n_images):
                cross = 0.50 + image / 1000
                gap = 0.001 + dim_offset / 100 + mod_offset / 200 + image / 10_000
                rows.append(
                    {
                        "image_id": f"image-{image:02d}",
                        "dimension": dimension,
                        "modality": modality,
                        "within_mean": cross + gap,
                        "cross_mean": cross,
                    }
                )
    return pd.DataFrame(rows)


def test_all_positive_differences_use_exact_wilcoxon():
    diff = np.arange(1.0, 51.0)

    result = sim.paired_stats(
        diff,
        np.zeros_like(diff),
        rng=42,
        n_boot=500,
        wilcoxon_resamples=999,
    )

    assert result["n_images"] == 50
    assert result["n_pos"] == 50
    assert result["n_zero"] == 0
    assert result["n_abs_ties"] == 0
    assert result["w_stat"] == 0
    assert result["p_wilcoxon"] == pytest.approx(2.0**-49)
    assert result["p_sign"] == pytest.approx(2.0**-49)
    assert result["rank_biserial"] == 1
    assert result["wilcoxon_method"] == "exact"


def test_negative_zero_tie_and_mixed_vectors_are_reported_correctly():
    negative = -np.arange(1.0, 7.0)
    neg_result = sim.paired_stats(negative, np.zeros(6), n_boot=200)
    assert neg_result["w_stat"] == 0
    assert neg_result["n_pos"] == 0
    assert neg_result["rank_biserial"] == -1
    assert neg_result["wilcoxon_method"] == "exact"

    zeros = np.zeros(6)
    zero_result = sim.paired_stats(zeros, zeros, n_boot=20)
    assert zero_result["mean_diff"] == 0
    assert zero_result["median_diff"] == 0
    assert zero_result["ci_lo"] == zero_result["ci_hi"] == 0
    assert zero_result["n_zero"] == 6
    assert zero_result["p_wilcoxon"] == 1
    assert zero_result["p_sign"] == 1
    assert zero_result["wilcoxon_method"] == "all_zero"

    tied = np.array([1.0, 1.0, 2.0, -2.0, 3.0, -4.0, 4.0, 5.0])
    tied_a = sim.paired_stats(tied, np.zeros_like(tied), rng=7, n_boot=100, wilcoxon_resamples=199)
    tied_b = sim.paired_stats(tied, np.zeros_like(tied), rng=7, n_boot=300, wilcoxon_resamples=199)
    assert tied_a["n_abs_ties"] == 6
    assert tied_a["wilcoxon_method"] == "permutation"
    assert tied_a["p_wilcoxon"] == tied_b["p_wilcoxon"]

    mixed = np.array([0.0, 1.0, 2.0, 3.0, -4.0])
    mixed_result = sim.paired_stats(mixed, np.zeros_like(mixed), n_boot=100, wilcoxon_resamples=199)
    assert mixed_result["n_pos"] == 3
    assert mixed_result["n_zero"] == 1
    assert mixed_result["p_sign"] == pytest.approx(0.625)
    assert mixed_result["wilcoxon_method"] == "permutation"


@pytest.mark.parametrize(
    ("a", "b", "match"),
    [
        (np.array([1.0]), np.array([0.0, 1.0]), "equal lengths"),
        (np.array([1.0, np.nan]), np.zeros(2), "finite"),
        (np.array([[1.0, 2.0]]), np.zeros(2), "one-dimensional"),
    ],
)
def test_paired_vector_validation(a, b, match):
    with pytest.raises(ValueError, match=match):
        sim.paired_stats(a, b, n_boot=10)


def test_cell_results_are_invariant_to_row_and_iteration_order():
    wc = _wc_frame(dimensions=("gender", "personality"))
    shuffled = wc.sample(frac=1, random_state=18).reset_index(drop=True)

    forward = sim.run_within_cross_significance(
        wc,
        demo_cols=("gender", "personality"),
        modalities=("caption",),
        n_boot=500,
        wilcoxon_resamples=199,
        seed=12,
    )
    reverse = sim.run_within_cross_significance(
        shuffled,
        demo_cols=("personality", "gender"),
        modalities=("caption",),
        n_boot=500,
        wilcoxon_resamples=199,
        seed=12,
    )

    order = ["dimension", "modality"]
    assert_frame_equal(
        forward.sort_values(order).reset_index(drop=True),
        reverse.sort_values(order).reset_index(drop=True),
    )
    assert set(forward["ci_relation_to_exploratory_reference"]).issubset(
        {"below", "overlaps", "above"}
    )
    assert "negligible" not in forward.columns
    assert (forward["p_bh"] == forward["p_wilcoxon_bh"]).all()


def test_exploratory_reference_relation_is_caption_only():
    wc = _wc_frame(modalities=("caption", "justification"))
    result = sim.run_within_cross_significance(
        wc,
        demo_cols=("gender",),
        modalities=("caption", "justification"),
        n_boot=200,
        wilcoxon_resamples=99,
    ).set_index("modality")

    assert result.loc["caption", "ci_relation_to_exploratory_reference"] in {
        "below",
        "overlaps",
        "above",
    }
    assert pd.isna(result.loc["justification", "ci_relation_to_exploratory_reference"])


def test_within_cross_frame_rejects_duplicates_and_nonfinite_values():
    wc = _wc_frame()
    duplicate = pd.concat([wc, wc.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique image/dimension/modality"):
        sim.run_within_cross_significance(
            duplicate,
            demo_cols=("gender",),
            modalities=("caption",),
            n_boot=10,
        )

    nonfinite = wc.copy()
    nonfinite.loc[0, "within_mean"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        sim.run_within_cross_significance(
            nonfinite,
            demo_cols=("gender",),
            modalities=("caption",),
            n_boot=10,
        )


def test_modality_contrast_defaults_to_justification_and_requires_pairing():
    wc = _wc_frame(modalities=("caption", "justification"))
    result = sim.run_modality_contrast(
        wc,
        demo_cols=("gender",),
        n_boot=300,
        wilcoxon_resamples=199,
    )
    assert result["target"].tolist() == ["justification"]
    assert {"p_sign", "p_sign_bh", "p_wilcoxon_bh", "median_contrast"}.issubset(result.columns)

    missing_pair = wc[~((wc["modality"] == "justification") & (wc["image_id"] == "image-00"))]
    with pytest.raises(ValueError, match="image pairing mismatch"):
        sim.run_modality_contrast(
            missing_pair,
            demo_cols=("gender",),
            n_boot=10,
            wilcoxon_resamples=19,
        )


def test_contrast_seed_namespace_moves_only_the_resampled_quantities():
    """The namespace exists so the matched construction gets its own streams.

    Rank statistics are resample-independent and must be untouched by it; the
    BCa interval is resampled and must not be a replay of the marginal one.
    """
    rng = np.random.default_rng(7)
    wc = _wc_frame(modalities=("caption", "justification"), n_images=25)
    varied = wc["modality"].eq("justification")
    wc.loc[varied, "within_mean"] += rng.uniform(0.001, 0.02, int(varied.sum()))

    kwargs = {"demo_cols": ("gender",), "n_boot": 400, "wilcoxon_resamples": 199}

    marginal = sim.run_modality_contrast(wc, **kwargs).iloc[0]
    matched = sim.run_modality_contrast(
        wc, seed_namespace="matched_factorial_contrast", **kwargs
    ).iloc[0]

    for column in ("mean_contrast", "median_contrast", "n_pos", "w_stat", "rank_biserial"):
        assert marginal[column] == matched[column]
    assert (marginal["ci_lo"], marginal["ci_hi"]) != (matched["ci_lo"], matched["ci_hi"])


def _alignment_fixture() -> tuple:
    frame = pd.DataFrame(
        {
            "annotation_id": ["missing", "a", "b", "c", "d"],
            "image_id": ["image"] * 5,
            "gender": ["A", "A", "A", "B", "B"],
            "predicted_perceptions": [[], [], ["x"], ["x"], ["y"]],
        }
    )
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    id_index = {"a": 0, "b": 1, "c": 2, "d": 3}
    return frame, embeddings, id_index


def test_missing_embeddings_fail_by_default_and_explicit_exclusion_stays_aligned():
    frame, embeddings, id_index = _alignment_fixture()

    with pytest.raises(ValueError, match="no embedding row"):
        sim.compute_within_cross_similarity(
            frame,
            embeddings,
            embeddings,
            id_index,
            demo_cols=["gender"],
        )

    result = sim.compute_within_cross_similarity(
        frame,
        embeddings,
        embeddings,
        id_index,
        demo_cols=["gender"],
        allow_missing_embeddings=True,
    ).set_index("modality")

    assert result.loc["caption", "within_mean"] == pytest.approx(1)
    assert result.loc["caption", "cross_mean"] == pytest.approx(0)
    assert result.loc["perception", "within_mean"] == pytest.approx(0)
    assert result.loc["perception", "cross_mean"] == pytest.approx(0.5)


def test_malformed_perception_tags_raise_instead_of_becoming_empty_sets():
    frame = pd.DataFrame(
        {
            "annotation_id": ["a", "b"],
            "image_id": ["image", "image"],
            "gender": ["A", "B"],
            "predicted_perceptions": [["x"], "{not valid"],
        }
    )
    embeddings = np.eye(2)

    with pytest.raises(ValueError, match="malformed predicted_perceptions"):
        sim.compute_within_cross_similarity(
            frame,
            embeddings,
            embeddings,
            {"a": 0, "b": 1},
            demo_cols=["gender"],
        )


def test_embedding_loader_rejects_reordered_or_length_mismatched_sidecars(
    tmp_path,
):
    frame = pd.DataFrame(
        {
            "annotation_id": ["a", "b"],
            "caption": ["first", "second"],
        }
    )
    embedding_path = tmp_path / "embeddings.npy"
    id_path = tmp_path / "embedding_ids.csv"
    np.save(embedding_path, np.eye(2))

    pd.DataFrame({"annotation_id": ["b", "a"]}).to_csv(id_path, index=False)
    with pytest.raises(ValueError, match="same row order"):
        load_or_encode_with_ids(
            frame,
            "caption",
            embedding_path,
            id_path,
        )

    pd.DataFrame({"annotation_id": ["a", "b"]}).to_csv(id_path, index=False)
    np.save(embedding_path, np.ones((1, 2)))
    with pytest.raises(ValueError, match="different row counts"):
        load_or_encode_with_ids(
            frame,
            "caption",
            embedding_path,
            id_path,
        )

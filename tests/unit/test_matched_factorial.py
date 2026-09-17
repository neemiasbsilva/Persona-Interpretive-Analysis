"""Pin the matched-factorial construction against brute-force references.

The construction is only interpretable if two properties hold exactly: the
matched pairs really do differ on one attribute alone, and the reported
within/cross means average over an identical pair set so that their difference
equals the mean of the per-pair contrasts. Both are asserted here on synthetic
corpora, so a regression surfaces without the 46 MB corpora.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from src.metrics import jaccard
from src.report_schema import MATCHED_FACTORIAL_COLUMNS
from src.schema import DEMO_COLS, MODALITIES
from src.similarity import (
    compute_matched_factorial_similarity,
    matched_profile_pairs,
    persona_profiles,
    stable_seed_sequence,
)
from src.similarity.matched_factorial import (
    _matched_pair_means,
    _profile_cosine_block,
    _profile_jaccard_block,
)

LEVELS = {
    "gender": ["Female", "Male"],
    "economic_status": ["Low income", "High income"],
    "political_spectrum": ["Liberal", "Conservative"],
    "personality": ["Pragmatic", "Idealistic", "Analytical"],
}
EXPECTED_PAIRS = {
    "gender": 12,
    "economic_status": 12,
    "political_spectrum": 12,
    "personality": 24,
}


def _synthetic_corpus(
    images: tuple[str, ...] = ("img_a", "img_b"),
    agents_per_profile: int = 3,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    profiles = list(itertools.product(*[LEVELS[column] for column in DEMO_COLS]))
    rows = [
        {
            "annotation_id": f"{image_id}-{'-'.join(profile)}-{replicate}",
            "image_id": image_id,
            "predicted_perceptions": sorted(
                rng.choice(
                    ["a", "b", "c", "d", "e"],
                    size=int(rng.integers(1, 4)),
                    replace=False,
                ).tolist()
            ),
            **dict(zip(DEMO_COLS, profile, strict=True)),
        }
        for image_id in images
        for profile in profiles
        for replicate in range(agents_per_profile)
    ]
    df = pd.DataFrame(rows)
    caption = rng.normal(size=(len(df), 6)) * rng.uniform(0.5, 3.0, size=(len(df), 1))
    justification = rng.normal(size=(len(df), 6))
    id_index = {aid: row for row, aid in enumerate(df["annotation_id"])}
    return df, caption, justification, id_index


def _profile_index(df: pd.DataFrame) -> tuple[list[tuple[str, ...]], np.ndarray]:
    profiles = persona_profiles(df)
    index_of = {profile: index for index, profile in enumerate(profiles)}
    keys = df[DEMO_COLS].astype(str).itertuples(index=False, name=None)
    return profiles, np.array([index_of[key] for key in keys])


def test_matched_pairs_have_the_expected_counts() -> None:
    df, *_ = _synthetic_corpus()
    profiles = persona_profiles(df)
    assert len(profiles) == 24
    for dimension, expected in EXPECTED_PAIRS.items():
        assert len(matched_profile_pairs(profiles, dimension)) == expected


def test_matched_pairs_differ_only_on_the_target_dimension() -> None:
    df, *_ = _synthetic_corpus()
    profiles = persona_profiles(df)
    for target, dimension in enumerate(DEMO_COLS):
        for first, second in matched_profile_pairs(profiles, dimension):
            assert profiles[first][target] != profiles[second][target]
            for other in range(len(DEMO_COLS)):
                if other != target:
                    assert profiles[first][other] == profiles[second][other]


def test_matched_pairs_are_sorted_and_deterministic() -> None:
    df, *_ = _synthetic_corpus()
    profiles = persona_profiles(df)
    for dimension in DEMO_COLS:
        pairs = matched_profile_pairs(profiles, dimension)
        assert pairs == matched_profile_pairs(profiles, dimension)
        assert pairs == sorted(pairs)
        assert all(first < second for first, second in pairs)


def test_unknown_dimension_and_duplicate_profiles_are_rejected() -> None:
    df, *_ = _synthetic_corpus()
    profiles = persona_profiles(df)
    with pytest.raises(ValueError, match="unknown persona dimension"):
        matched_profile_pairs(profiles, "height")
    with pytest.raises(ValueError, match="distinct"):
        matched_profile_pairs([*profiles, profiles[0]], "gender")


def test_cosine_block_uses_actual_norms_not_a_unit_assumption() -> None:
    df, caption, _, _ = _synthetic_corpus(images=("img_a",))
    _, profile_index = _profile_index(df)
    order = np.argsort(profile_index, kind="stable")
    counts = np.bincount(profile_index[order], minlength=24)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)[:-1]))

    observed = _profile_cosine_block(caption[order], starts, counts)

    similarity = cosine_similarity(caption)
    expected = np.full((24, 24), np.nan)
    for first in range(24):
        members = np.flatnonzero(profile_index == first)
        block = similarity[np.ix_(members, members)]
        expected[first, first] = block[np.triu_indices(len(members), k=1)].mean()
        for second in range(first + 1, 24):
            others = np.flatnonzero(profile_index == second)
            value = similarity[np.ix_(members, others)].mean()
            expected[first, second] = expected[second, first] = value

    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)


def test_jaccard_block_matches_the_metrics_helper() -> None:
    df, *_ = _synthetic_corpus(images=("img_a",))
    _, profile_index = _profile_index(df)
    tag_sets = [set(tags) for tags in df["predicted_perceptions"]]

    observed = _profile_jaccard_block(cast("list[set[str] | None]", tag_sets), profile_index, 24)

    expected = np.full((24, 24), np.nan)
    for first in range(24):
        members = np.flatnonzero(profile_index == first)
        expected[first, first] = np.mean(
            [
                jaccard(tag_sets[a], tag_sets[b], on_empty_union=0.0)
                for a, b in itertools.combinations(members, 2)
            ]
        )
        for second in range(first + 1, 24):
            others = np.flatnonzero(profile_index == second)
            value = np.mean(
                [
                    jaccard(tag_sets[a], tag_sets[b], on_empty_union=0.0)
                    for a in members
                    for b in others
                ]
            )
            expected[first, second] = expected[second, first] = value

    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)


def test_empty_perception_responses_take_no_part_in_any_pair() -> None:
    df, *_ = _synthetic_corpus(images=("img_a",))
    _, profile_index = _profile_index(df)
    tag_sets: list[set[str] | None] = [set(tags) for tags in df["predicted_perceptions"]]

    kept = _profile_jaccard_block(tag_sets, profile_index, 24)
    dropped = _profile_jaccard_block([None, *tag_sets[1:]], profile_index, 24)

    owner = int(profile_index[0])
    assert kept[owner, owner] != pytest.approx(dropped[owner, owner])

    equivalent = _profile_jaccard_block(tag_sets[1:], profile_index[1:], 24)
    np.testing.assert_allclose(dropped, equivalent, rtol=0, atol=1e-12, equal_nan=True)


def test_mean_of_differences_equals_difference_of_means() -> None:
    df, caption, justification, id_index = _synthetic_corpus(images=("img_a",))
    profiles, profile_index = _profile_index(df)
    order = np.argsort(profile_index, kind="stable")
    counts = np.bincount(profile_index[order], minlength=24)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)[:-1]))
    blocks = {
        "caption": _profile_cosine_block(caption[order], starts, counts),
        "justification": _profile_cosine_block(justification[order], starts, counts),
        "perception": _profile_jaccard_block(
            [set(tags) for tags in df["predicted_perceptions"]],
            profile_index,
            24,
        ),
    }

    frame = compute_matched_factorial_similarity(df, caption, justification, id_index)
    for dimension in DEMO_COLS:
        pairs = matched_profile_pairs(profiles, dimension)
        for modality in MODALITIES:
            block = blocks[modality]
            per_pair = np.mean([(block[p, p] + block[q, q]) / 2.0 - block[p, q] for p, q in pairs])
            row = frame[(frame["dimension"] == dimension) & (frame["modality"] == modality)].iloc[0]
            assert per_pair == pytest.approx(row["within_mean"] - row["cross_mean"], abs=1e-12)


def test_unusable_pair_is_dropped_from_both_terms_jointly() -> None:
    block = np.array(
        [
            [0.9, 0.5, 0.4],
            [0.5, np.nan, 0.3],
            [0.4, 0.3, 0.7],
        ]
    )
    pairs = [(0, 1), (0, 2), (1, 2)]

    within, cross, n_pairs = _matched_pair_means(block, pairs)

    assert n_pairs == 1
    assert within == pytest.approx((0.9 + 0.7) / 2.0)
    assert cross == pytest.approx(0.4)
    assert np.isfinite([within, cross]).all()


def test_a_cell_with_no_usable_pair_is_missing_rather_than_zero() -> None:
    within, cross, n_pairs = _matched_pair_means(np.full((2, 2), np.nan), [(0, 1)])
    assert n_pairs == 0
    assert np.isnan(within)
    assert np.isnan(cross)


def test_single_agent_profile_removes_only_the_pairs_that_touch_it() -> None:
    df, caption, justification, _ = _synthetic_corpus(images=("img_a",))
    profiles, profile_index = _profile_index(df)
    lonely = int(profile_index[0])
    keep = (profile_index != lonely) | (np.arange(len(df)) == 0)
    reduced = df.loc[keep].reset_index(drop=True)
    reduced_index = {aid: row for row, aid in enumerate(reduced["annotation_id"])}

    frame = compute_matched_factorial_similarity(
        reduced,
        caption[keep],
        justification[keep],
        reduced_index,
    )
    for dimension in DEMO_COLS:
        pairs = matched_profile_pairs(profiles, dimension)
        touching = sum(1 for pair in pairs if lonely in pair)
        observed = frame.loc[frame["dimension"] == dimension, "n_matched_pairs"].unique()
        assert observed.tolist() == [len(pairs) - touching]
        assert np.isfinite(
            frame.loc[frame["dimension"] == dimension, ["within_mean", "cross_mean"]]
        ).all(axis=None)


def test_frame_is_one_row_per_image_dimension_and_modality() -> None:
    df, caption, justification, id_index = _synthetic_corpus()
    frame = compute_matched_factorial_similarity(df, caption, justification, id_index)

    assert len(frame) == 2 * len(DEMO_COLS) * len(MODALITIES)
    assert not frame.duplicated(["image_id", "dimension", "modality"]).any()
    assert np.isfinite(frame[["within_mean", "cross_mean"]]).all(axis=None)
    for dimension, expected in EXPECTED_PAIRS.items():
        counts = frame.loc[frame["dimension"] == dimension, "n_matched_pairs"].unique()
        assert counts.tolist() == [expected]


def test_cache_round_trip_returns_an_identical_frame(tmp_path: Path) -> None:
    df, caption, justification, id_index = _synthetic_corpus()
    cache = tmp_path / "matched_factorial_persona_sim.csv"

    first = compute_matched_factorial_similarity(
        df, caption, justification, id_index, cache_path=cache
    )
    second = compute_matched_factorial_similarity(
        df, caption, justification, id_index, cache_path=cache
    )

    assert cache.exists()
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_matched_columns_exclude_the_exploratory_reference() -> None:
    assert "ci_relation_to_exploratory_reference" not in MATCHED_FACTORIAL_COLUMNS
    assert "stars" not in MATCHED_FACTORIAL_COLUMNS
    assert "marginal_delta" in MATCHED_FACTORIAL_COLUMNS
    assert "matched_to_marginal_ratio" in MATCHED_FACTORIAL_COLUMNS


def test_default_seed_namespace_reproduces_the_frozen_within_cross_key() -> None:
    for dimension in DEMO_COLS:
        for modality in MODALITIES:
            explicit = stable_seed_sequence(42, "within_cross", dimension, modality)
            assert list(cast("Sequence[int]", explicit.entropy)) == list(
                cast(
                    "Sequence[int]",
                    stable_seed_sequence(42, "within_cross", dimension, modality).entropy,
                )
            )

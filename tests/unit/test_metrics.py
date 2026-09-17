"""Cover the set-overlap metrics, especially the two empty-union conventions."""

from __future__ import annotations

import pytest

from src.metrics import jaccard, lenient_tag_set


def test_identical_sets_overlap_completely() -> None:
    assert jaccard({"a", "b"}, {"a", "b"}, on_empty_union=0.0) == 1.0


def test_disjoint_sets_do_not_overlap() -> None:
    assert jaccard({"a"}, {"b"}, on_empty_union=1.0) == 0.0


def test_partial_overlap_is_intersection_over_union() -> None:
    assert jaccard({"a", "b", "c"}, {"b", "c", "d"}, on_empty_union=0.0) == pytest.approx(2 / 4)


def test_one_empty_side_is_zero_regardless_of_the_empty_union_convention() -> None:
    assert jaccard(set(), {"a"}, on_empty_union=1.0) == 0.0
    assert jaccard({"a"}, set(), on_empty_union=0.0) == 0.0


def test_statistical_paths_score_two_empty_responses_as_no_overlap() -> None:
    assert jaccard(set(), set(), on_empty_union=0.0) == 0.0


def test_label_agreement_scores_two_empty_responses_as_full_agreement() -> None:
    assert jaccard(set(), set(), on_empty_union=1.0) == 1.0


def test_on_empty_union_is_keyword_only_and_has_no_default() -> None:
    with pytest.raises(TypeError):
        jaccard({"a"}, {"b"})  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        jaccard({"a"}, {"b"}, 0.0)  # type: ignore[call-arg]


def test_duplicates_and_iterables_are_normalised_to_sets() -> None:
    assert jaccard(["a", "a", "b"], ("b", "a"), on_empty_union=0.0) == 1.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (["a", "b"], {"a", "b"}),
        ("['a', 'b']", {"a", "b"}),
        ("[]", set()),
        ("{not valid", set()),
        (None, set()),
        (42, set()),
        ("'solo'", set()),
        ("('a', 'b')", {"a", "b"}),
    ],
)
def test_lenient_tag_set_never_raises(value: object, expected: set[str]) -> None:
    assert lenient_tag_set(value) == expected

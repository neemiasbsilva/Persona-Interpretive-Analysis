"""Guard the seed derivation that every published interval and p-value depends on.

``stable_seed_sequence`` hashes a literal string built from the cell key. Renaming
a key literal, reordering the varargs, or swapping the two spawned streams changes
every BCa interval and permutation p-value in the paper without raising an error.
These tests pin the derivation so that damage surfaces in milliseconds rather than
after a full 100,000-resample regeneration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest

from src.schema import DEMO_COLS, MODALITIES
from src.similarity import stable_seed_sequence


def _cell_keys() -> list[tuple[str, ...]]:
    """Enumerate every statistical cell key in the order the report generates them.

    Returns:
        The 12 within/cross keys, the 4 modality-contrast keys, the 12
        matched-factorial keys, and the 4 matched-factorial contrast keys.
    """
    keys: list[tuple[str, ...]] = [
        ("within_cross", dim, mod) for dim in DEMO_COLS for mod in MODALITIES
    ]
    keys.extend(("modality_contrast", dim, "justification", "caption") for dim in DEMO_COLS)
    keys.extend(("matched_factorial", dim, mod) for dim in DEMO_COLS for mod in MODALITIES)
    keys.extend(
        ("matched_factorial_contrast", dim, "justification", "caption") for dim in DEMO_COLS
    )
    return keys


def _entropy(sequence: np.random.SeedSequence) -> list[int]:
    return list(cast("Sequence[int]", sequence.entropy))


def test_every_cell_key_derives_its_recorded_entropy(
    frozen_seed_entropy: dict[str, list[int]],
) -> None:
    assert len(frozen_seed_entropy) == 32

    for key in _cell_keys():
        recorded = frozen_seed_entropy["|".join(key)]
        actual = [int(value) for value in _entropy(stable_seed_sequence(42, *key))]
        assert actual == recorded, f"seed derivation changed for {key}"


def test_cell_keys_cover_the_recorded_set(frozen_seed_entropy: dict[str, list[int]]) -> None:
    assert {"|".join(key) for key in _cell_keys()} == set(frozen_seed_entropy)


def test_seed_depends_on_key_order_and_separator() -> None:
    joined = stable_seed_sequence(42, "within_cross", "gender", "caption")
    reordered = stable_seed_sequence(42, "within_cross", "caption", "gender")
    assert _entropy(joined) != _entropy(reordered)

    split = stable_seed_sequence(42, "within_cross", "gender", "caption")
    merged = stable_seed_sequence(42, "within_cross", "gendercaption")
    assert _entropy(split) != _entropy(merged)


def test_seed_namespaces_do_not_collide() -> None:
    for dim in DEMO_COLS:
        for mod in MODALITIES:
            marginal = stable_seed_sequence(42, "within_cross", dim, mod)
            matched = stable_seed_sequence(42, "matched_factorial", dim, mod)
            assert _entropy(marginal) != _entropy(matched)


def test_matched_contrast_does_not_reuse_the_marginal_contrast_streams() -> None:
    for dim in DEMO_COLS:
        marginal = stable_seed_sequence(42, "modality_contrast", dim, "justification", "caption")
        matched = stable_seed_sequence(
            42, "matched_factorial_contrast", dim, "justification", "caption"
        )
        assert _entropy(marginal) != _entropy(matched)


def test_base_seed_participates_in_the_derivation() -> None:
    assert _entropy(stable_seed_sequence(42, "within_cross", "gender", "caption")) != _entropy(
        stable_seed_sequence(43, "within_cross", "gender", "caption")
    )


def test_spawned_stream_order_is_bootstrap_then_permutation() -> None:
    parent = stable_seed_sequence(42, "within_cross", "gender", "caption")
    first, second = parent.spawn(2)
    assert _entropy(first) == _entropy(parent)
    assert first.spawn_key == (0,)
    assert second.spawn_key == (1,)
    assert np.random.default_rng(first).random() != np.random.default_rng(second).random()


@pytest.mark.parametrize("seed", [0, 42, 2**31])
def test_derivation_is_deterministic_across_calls(seed: int) -> None:
    key = ("modality_contrast", "personality", "justification", "caption")
    assert _entropy(stable_seed_sequence(seed, *key)) == _entropy(stable_seed_sequence(seed, *key))

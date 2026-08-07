"""Guard the seed derivation that every published interval and p-value depends on.

``stable_seed_sequence`` hashes a literal string built from the cell key. Renaming
a key literal, reordering the varargs, or swapping the two spawned streams changes
every BCa interval and permutation p-value in the paper without raising an error.
These tests pin the derivation so that damage surfaces in milliseconds rather than
after a full 100,000-resample regeneration.
"""

from __future__ import annotations

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


def test_every_cell_key_derives_its_recorded_entropy(frozen_seed_entropy):
    assert len(frozen_seed_entropy) == 32

    for key in _cell_keys():
        recorded = frozen_seed_entropy["|".join(key)]
        actual = [int(value) for value in stable_seed_sequence(42, *key).entropy]
        assert actual == recorded, f"seed derivation changed for {key}"


def test_cell_keys_cover_the_recorded_set(frozen_seed_entropy):
    assert {"|".join(key) for key in _cell_keys()} == set(frozen_seed_entropy)


def test_seed_depends_on_key_order_and_separator():
    joined = stable_seed_sequence(42, "within_cross", "gender", "caption")
    reordered = stable_seed_sequence(42, "within_cross", "caption", "gender")
    assert list(joined.entropy) != list(reordered.entropy)

    split = stable_seed_sequence(42, "within_cross", "gender", "caption")
    merged = stable_seed_sequence(42, "within_cross", "gendercaption")
    assert list(split.entropy) != list(merged.entropy)


def test_seed_namespaces_do_not_collide():
    for dim in DEMO_COLS:
        for mod in MODALITIES:
            marginal = stable_seed_sequence(42, "within_cross", dim, mod)
            matched = stable_seed_sequence(42, "matched_factorial", dim, mod)
            assert list(marginal.entropy) != list(matched.entropy)


def test_matched_contrast_does_not_reuse_the_marginal_contrast_streams():
    for dim in DEMO_COLS:
        marginal = stable_seed_sequence(42, "modality_contrast", dim, "justification", "caption")
        matched = stable_seed_sequence(
            42, "matched_factorial_contrast", dim, "justification", "caption"
        )
        assert list(marginal.entropy) != list(matched.entropy)


def test_base_seed_participates_in_the_derivation():
    assert list(stable_seed_sequence(42, "within_cross", "gender", "caption").entropy) != list(
        stable_seed_sequence(43, "within_cross", "gender", "caption").entropy
    )


def test_spawned_stream_order_is_bootstrap_then_permutation():
    parent = stable_seed_sequence(42, "within_cross", "gender", "caption")
    first, second = parent.spawn(2)
    assert list(first.entropy) == list(parent.entropy)
    assert first.spawn_key == (0,)
    assert second.spawn_key == (1,)
    assert np.random.default_rng(first).random() != np.random.default_rng(second).random()


@pytest.mark.parametrize("seed", [0, 42, 2**31])
def test_derivation_is_deterministic_across_calls(seed):
    key = ("modality_contrast", "personality", "justification", "caption")
    assert list(stable_seed_sequence(seed, *key).entropy) == list(
        stable_seed_sequence(seed, *key).entropy
    )

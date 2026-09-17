"""Cover the digests recorded in cache sidecars.

These digests decide whether a cached artifact is considered current, so the
tests pin both what they detect and what they deliberately ignore.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.hashing import array_digest, sequence_digest, sha256_file


def test_file_digest_matches_hashlib(tmp_path: Path) -> None:
    payload = b"persona" * 5000
    path = tmp_path / "corpus.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_file_digest_spans_chunk_boundaries(tmp_path: Path) -> None:
    payload = b"x" * (1024 * 1024 * 2 + 17)
    path = tmp_path / "big.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sequence_digest_is_order_sensitive() -> None:
    assert sequence_digest(["a", "b"]) != sequence_digest(["b", "a"])


def test_sequence_digest_resists_boundary_forgery() -> None:
    assert sequence_digest(["ab", "c"]) != sequence_digest(["a", "bc"])
    assert sequence_digest(["abc"]) != sequence_digest(["a", "b", "c"])


def test_sequence_digest_distinguishes_types() -> None:
    assert sequence_digest([1]) != sequence_digest(["1"])


def test_sequence_digest_is_stable_across_calls() -> None:
    assert sequence_digest(range(50)) == sequence_digest(range(50))


def test_array_digest_detects_content_shape_and_dtype() -> None:
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert array_digest(base) != array_digest(base + 1)
    assert array_digest(base) != array_digest(base.reshape(4, 3))
    assert array_digest(base) != array_digest(base.astype(np.float32))


def test_array_digest_ignores_memory_layout() -> None:
    base = np.arange(24, dtype=np.float64).reshape(4, 6)
    view = base[::2]
    assert not view.flags["C_CONTIGUOUS"]
    assert array_digest(view) == array_digest(np.ascontiguousarray(view))


def test_array_digest_handles_empty_arrays() -> None:
    assert array_digest(np.array([], dtype=np.float64)) == array_digest(
        np.array([], dtype=np.float64)
    )

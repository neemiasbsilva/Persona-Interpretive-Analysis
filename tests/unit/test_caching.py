"""Cover the read-through cache helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.caching import cached_array, cached_frame


class Counter:
    """Callable wrapper recording how many times the payload was computed."""

    def __init__(self, payload: object) -> None:
        """Store the payload and reset the call count."""
        self.payload = payload
        self.calls = 0

    def __call__(self) -> object:
        """Return the payload, counting the call."""
        self.calls += 1
        return self.payload


def test_frame_cache_computes_once_then_reads(tmp_path):
    compute = Counter(pd.DataFrame({"image_id": [1, 2], "score": [0.5, 0.25]}))
    path = tmp_path / "scores.csv"

    first = cached_frame(path, compute)
    assert compute.calls == 1
    assert path.exists()

    second = cached_frame(path, compute)
    assert compute.calls == 1
    pd.testing.assert_frame_equal(first, second)


def test_frame_cache_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deep" / "scores.csv"
    cached_frame(path, Counter(pd.DataFrame({"a": [1]})))
    assert path.exists()


def test_frame_cache_round_trips_the_index_when_requested(tmp_path):
    frame = pd.DataFrame({"score": [1.0, 2.0]}, index=pd.Index(["a", "b"], name="profile"))
    path = tmp_path / "indexed.csv"

    cached_frame(path, Counter(frame), index=True)
    restored = cached_frame(path, Counter(pd.DataFrame()), index=True)
    pd.testing.assert_frame_equal(frame, restored)


def test_disabled_frame_cache_always_computes_and_writes_nothing(tmp_path):
    compute = Counter(pd.DataFrame({"a": [1]}))
    cached_frame(None, compute)
    cached_frame(None, compute)
    assert compute.calls == 2
    assert list(tmp_path.iterdir()) == []


def test_array_cache_computes_once_then_reads(tmp_path):
    payload = np.arange(6, dtype=np.float64).reshape(2, 3)
    compute = Counter(payload)
    path = tmp_path / "embeddings.npy"

    first = cached_array(path, compute)
    assert compute.calls == 1

    second = cached_array(path, compute)
    assert compute.calls == 1
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(second, payload)


def test_array_cache_preserves_dtype_and_shape(tmp_path):
    payload = np.zeros((3, 7), dtype=np.float64)
    path = tmp_path / "zeros.npy"
    cached_array(path, Counter(payload))
    restored = cached_array(path, Counter(np.array([])))
    assert restored.shape == payload.shape
    assert restored.dtype == payload.dtype


def test_disabled_array_cache_always_computes(tmp_path):
    compute = Counter(np.ones(3))
    cached_array(None, compute)
    cached_array(None, compute)
    assert compute.calls == 2
    assert list(tmp_path.iterdir()) == []

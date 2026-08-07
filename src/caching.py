"""Read-through caches for the expensive stages of the pipeline.

Every cache here is content-addressed only by its path: a present file is trusted
and returned. Caches whose staleness would change a published number carry a
``.meta.json`` sidecar instead and are validated by
:mod:`src.similarity.cache_metadata`, not by these helpers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .arrays import FloatArray


def cached_frame(
    cache_path: Path | None,
    compute: Callable[[], pd.DataFrame],
    *,
    index: bool = False,
) -> pd.DataFrame:
    """Return a cached DataFrame, computing and writing it on a miss.

    Args:
        cache_path: CSV cache location, or ``None`` to disable caching entirely.
        compute: Callable producing the frame when the cache is absent.
        index: Whether to persist and restore the frame index.

    Returns:
        The cached or freshly computed frame.
    """
    if cache_path is not None and cache_path.exists():
        return pd.read_csv(cache_path, index_col=0 if index else None)

    frame = compute()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index=index)
    return frame


def cached_array(
    cache_path: Path | None,
    compute: Callable[[], FloatArray],
) -> FloatArray:
    """Return a cached array, computing and writing it on a miss.

    Args:
        cache_path: ``.npy`` cache location, or ``None`` to disable caching.
        compute: Callable producing the array when the cache is absent.

    Returns:
        The cached or freshly computed array.
    """
    if cache_path is not None and cache_path.exists():
        loaded: FloatArray = np.load(cache_path)
        return loaded

    array = compute()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, array)
    return array

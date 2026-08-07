"""Content digests used to prove cached artifacts match the inputs that built them.

The digests are recorded in the ``.meta.json`` sidecar next to each cache. They
are what makes a stale cache detectable rather than silently reusable, so their
byte-level behaviour is a compatibility contract: changing how a value is encoded
invalidates every sidecar written before the change.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Digest a file's contents, reading it in chunks.

    Args:
        path: File to digest.

    Returns:
        Lowercase hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_digest(values: Iterable[Any]) -> str:
    """Digest an ordered sequence so that reordering changes the result.

    Each value is encoded with ``repr`` and followed by a NUL separator, so
    concatenation cannot forge an equal digest from different boundaries.

    Args:
        values: Values in the order they must be preserved.

    Returns:
        Lowercase hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    for value in values:
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def array_digest(array: np.ndarray[Any, Any]) -> str:
    """Digest an array's shape, dtype, and contents.

    The array is made contiguous first, so a view and its base produce the same
    digest when their values agree.

    Args:
        array: Array to digest.

    Returns:
        Lowercase hex SHA-256 digest.
    """
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(repr((contiguous.shape, contiguous.dtype.str)).encode("ascii"))
    digest.update(contiguous.data.cast("B"))
    return digest.hexdigest()

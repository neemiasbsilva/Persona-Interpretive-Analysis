"""Set-overlap metrics for perception tags.

``on_empty_union`` is keyword-only and has no default on purpose. Two conventions
are live in this repository and they are not interchangeable: the within/cross
and profile-coherence paths score a pair of empty tag sets as 0.0, while
per-image label agreement in :mod:`src.convergence` scores it as 1.0 (two
annotators who both produced no tags agreed). Collapsing them would silently
change ``convergence_all_dimensions.csv``, so every call site must state which
convention it means.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable


def jaccard(a: Iterable[str], b: Iterable[str], *, on_empty_union: float) -> float:
    """Return the Jaccard overlap of two tag collections.

    Args:
        a: First tag collection.
        b: Second tag collection.
        on_empty_union: Value to return when both collections are empty.

    Returns:
        ``|a n b| / |a u b|``, or ``on_empty_union`` when the union is empty.
    """
    set_a = a if isinstance(a, set) else set(a)
    set_b = b if isinstance(b, set) else set(b)
    union = len(set_a | set_b)
    if union == 0:
        return on_empty_union
    return len(set_a & set_b) / union


def lenient_tag_set(value: object) -> set[str]:
    """Parse a perception-tag cell, treating anything unparseable as empty.

    Used only by figure code, where a malformed cell should not abort a plot.
    Statistical paths use :func:`src.similarity.parse_perception_tags`, which
    raises instead, so a corrupt corpus cannot silently become zero overlap.

    Args:
        value: A list of tags or its string repr.

    Returns:
        The parsed tag set, empty when the value cannot be parsed.
    """
    if isinstance(value, list):
        return set(value)
    if not isinstance(value, str):
        return set()
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return set()
    if isinstance(parsed, (list, set, tuple)):
        return set(parsed)
    return set()

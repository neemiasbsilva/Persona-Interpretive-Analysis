"""Strict parsing of the perception-tag column."""

import ast

import numpy as np


def parse_perception_tags(value: object, annotation_id: object = "<unknown>") -> set[str] | None:
    """Parse one perception response, treating an empty response as missing.

    A genuinely empty tag list does not encode evidence of zero overlap with
    another response, so callers exclude it from Jaccard pairs. Malformed
    serialized values and non-string tags fail loudly instead of silently
    becoming empty sets.
    """
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    parsed = value
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"malformed predicted_perceptions for {annotation_id!r}: {value!r}"
            ) from exc
    if not isinstance(parsed, (list, tuple, set)):
        raise ValueError(
            "predicted_perceptions must be a list-like value; "
            f"annotation {annotation_id!r} has {type(parsed).__name__}"
        )
    if any(not isinstance(tag, str) for tag in parsed):
        raise ValueError(
            "perception tags must be strings; "
            f"annotation {annotation_id!r} contains a non-string tag"
        )
    tags = {tag.strip() for tag in parsed if tag.strip()}
    return tags or None

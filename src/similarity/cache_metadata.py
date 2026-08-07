"""Provenance sidecars proving a within/cross cache matches the inputs that built it."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..hashing import array_digest, sequence_digest, sha256_file

WITHIN_CROSS_CACHE_SCHEMA = 2
EMPTY_TAG_POLICY = "exclude_pairs_with_empty_perception_responses"


def within_cross_cache_metadata_path(cache_path: Path) -> Path:
    """Return the policy/provenance sidecar for an image-level cache."""
    cache_path = Path(cache_path)
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def validate_within_cross_cache_metadata(
    cache_path: Path,
    expected: dict | None = None,
) -> dict:
    """Validate cache policy, content digest, and optional input provenance."""
    cache_path = Path(cache_path)
    metadata_path = within_cross_cache_metadata_path(cache_path)
    if not cache_path.exists() or not metadata_path.exists():
        raise ValueError(f"{cache_path} and {metadata_path.name} must both exist")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != WITHIN_CROSS_CACHE_SCHEMA:
        raise ValueError("within/cross cache uses an obsolete schema")
    if metadata.get("empty_perception_policy") != EMPTY_TAG_POLICY:
        raise ValueError("within/cross cache uses an obsolete empty-tag policy")
    if metadata.get("csv_sha256") != sha256_file(cache_path):
        raise ValueError("within/cross cache digest does not match its metadata")
    if expected is not None:
        mismatched = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatched:
            raise ValueError(
                "within/cross cache input provenance changed for: " + ", ".join(mismatched)
            )
    return metadata


def within_cross_input_metadata(
    df: pd.DataFrame,
    cap_embs: np.ndarray,
    just_embs: np.ndarray,
    id_index: dict[str, int],
    demo_cols: list,
    *,
    allow_missing_embeddings: bool,
) -> dict:
    """Fingerprint every input a within/cross cache entry depends on.

    Args:
        df: Annotation frame the cache was computed from.
        cap_embs: Caption embedding matrix.
        just_embs: Justification embedding matrix.
        id_index: Mapping from annotation ID to embedding row index.
        demo_cols: Demographic dimensions the cache covers.
        allow_missing_embeddings: Whether rows without embeddings were tolerated.

    Returns:
        Provenance dict compared against the sidecar to decide cache validity.
    """
    relevant = [
        "annotation_id",
        "image_id",
        "predicted_perceptions",
        *demo_cols,
    ]
    mapping_order = [
        annotation_id for annotation_id, _ in sorted(id_index.items(), key=lambda item: item[1])
    ]
    return {
        "schema_version": WITHIN_CROSS_CACHE_SCHEMA,
        "empty_perception_policy": EMPTY_TAG_POLICY,
        "annotation_rows": len(df),
        "annotation_data_sha256": sequence_digest(df[relevant].itertuples(index=False, name=None)),
        "embedding_id_order_sha256": sequence_digest(mapping_order),
        "caption_embeddings_sha256": array_digest(cap_embs),
        "justification_embeddings_sha256": array_digest(just_embs),
        "demographic_columns": list(demo_cols),
        "allow_missing_embeddings": bool(allow_missing_embeddings),
    }

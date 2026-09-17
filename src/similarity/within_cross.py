"""Within-persona versus cross-persona similarity, per image and dimension."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..hashing import sha256_file
from ..metrics import jaccard
from ..schema import DEMO_COLS
from .cache_metadata import (
    validate_within_cross_cache_metadata,
    within_cross_cache_metadata_path,
    within_cross_input_metadata,
)
from .perception import parse_perception_tags


def _validate_annotation_frame(df: pd.DataFrame, demo_cols: list[str]) -> None:
    """Fail unless the annotation frame is complete and uniquely keyed.

    Args:
        df: Annotation frame carrying the ID, image, perception, and demographic columns.
        demo_cols: Demographic columns that must be present and complete.

    Raises:
        ValueError: If a column is absent, a key is missing, or an ID repeats.
    """
    required = {"annotation_id", "image_id", "predicted_perceptions", *demo_cols}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"within/cross input is missing columns: {missing}")
    if df["annotation_id"].isna().any() or df["image_id"].isna().any():
        raise ValueError("annotation_id and image_id must not contain missing values")
    duplicated = df["annotation_id"].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "annotation_id"].astype(str).unique()[:5]
        raise ValueError(f"annotation_id must be unique; duplicate examples: {examples.tolist()}")
    if df[demo_cols].isna().any(axis=None):
        raise ValueError("demographic columns must not contain missing values")


def _validate_embedding_index(
    id_index: dict[str, int],
    n_rows: int,
) -> None:
    """Fail unless the ID mapping is a bijection onto the embedding rows.

    Args:
        id_index: Bijection from annotation ID to embedding row.
        n_rows: Number of embedding rows the mapping must cover.

    Raises:
        ValueError: If the mapping is not a contiguous, in-bounds bijection.
    """
    embedding_indices = list(id_index.values())
    if any(not isinstance(i, (int, np.integer)) for i in embedding_indices):
        raise ValueError("id_index values must be integer row indices")
    if len(set(embedding_indices)) != len(embedding_indices):
        raise ValueError("id_index must map annotation IDs to unique embedding rows")
    if sorted(embedding_indices) != list(range(len(embedding_indices))):
        raise ValueError("id_index row indices must be contiguous from zero")
    if len(embedding_indices) != n_rows:
        raise ValueError("id_index must contain exactly one ID for every embedding row")
    if embedding_indices and (min(embedding_indices) < 0 or max(embedding_indices) >= n_rows):
        raise ValueError("id_index contains an out-of-bounds embedding row")


def validate_similarity_inputs(
    df: pd.DataFrame,
    cap_embs: np.ndarray,
    just_embs: np.ndarray,
    id_index: dict[str, int],
    demo_cols: list[str],
    *,
    allow_missing_embeddings: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the annotation frame, embedding arrays, and ID mapping.

    Shared by every per-image similarity construction so that they cannot drift
    apart in what they consider a well-formed corpus.

    Args:
        df: Annotation frame carrying the ID, image, perception, and demographic columns.
        cap_embs: Caption embedding matrix.
        just_embs: Justification embedding matrix.
        id_index: Bijection from annotation ID to embedding row.
        demo_cols: Demographic columns that must be present and complete.
        allow_missing_embeddings: Permit annotations that have no embedding row.

    Returns:
        The two embedding matrices as arrays.

    Raises:
        ValueError: If any structural or completeness expectation is violated.
    """
    _validate_annotation_frame(df, demo_cols)

    cap_embs = np.asarray(cap_embs)
    just_embs = np.asarray(just_embs)
    if cap_embs.ndim != 2 or just_embs.ndim != 2:
        raise ValueError("caption and justification embeddings must be 2-D")
    if cap_embs.shape[0] != just_embs.shape[0]:
        raise ValueError("caption and justification embedding arrays must have the same row count")
    if not np.isfinite(cap_embs).all() or not np.isfinite(just_embs).all():
        raise ValueError("embedding arrays must contain only finite values")

    _validate_embedding_index(id_index, cap_embs.shape[0])

    missing_embeddings = ~df["annotation_id"].isin(id_index)
    if missing_embeddings.any() and not allow_missing_embeddings:
        examples = df.loc[missing_embeddings, "annotation_id"].astype(str).unique()[:5]
        raise ValueError(
            f"{int(missing_embeddings.sum())} annotations have no embedding row; "
            f"examples: {examples.tolist()}. Set allow_missing_embeddings=True "
            "to exclude those annotations explicitly."
        )
    return cap_embs, just_embs


def _read_valid_cache(
    cache_path: Path | None, input_metadata: dict[str, object]
) -> pd.DataFrame | None:
    """Return the cached within/cross frame when it is present and trustworthy.

    Args:
        cache_path: Cache CSV to read, or None when caching is disabled.
        input_metadata: Provenance the cache sidecar must agree with.

    Returns:
        The cached frame, or None when it is absent, stale or malformed.
    """
    if not cache_path or not cache_path.exists():
        return None
    try:
        validate_within_cross_cache_metadata(cache_path, input_metadata)
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    cached = pd.read_csv(cache_path)
    required_cache = {"image_id", "dimension", "modality", "within_mean", "cross_mean"}
    if not required_cache.issubset(cached.columns):
        return None
    if cached.duplicated(["image_id", "dimension", "modality"]).any():
        return None
    values = cached[["within_mean", "cross_mean"]].to_numpy(dtype=float)
    return cached if np.isfinite(values).all() else None


def compute_within_cross_similarity(
    df: pd.DataFrame,
    cap_embs: np.ndarray,
    just_embs: np.ndarray,
    id_index: dict[str, int],
    demo_cols: list[str] = DEMO_COLS,
    *,
    cache_path: Path | None = None,
    allow_missing_embeddings: bool = False,
) -> pd.DataFrame:
    """Within-group vs. cross-group similarity per demographic dimension.

    Pairs are bucketed on one dimension at a time while the other demographic
    attributes vary freely, so the result is attribute-marginal.

    Covers three modalities:
      - caption       : cosine similarity of Sentence-BERT embeddings
      - justification : cosine similarity of Sentence-BERT embeddings
      - perception    : Jaccard similarity of perception tag sets

    Returns DataFrame with columns:
        image_id, dimension, modality, within_mean, cross_mean
    """
    cache_path = Path(cache_path) if cache_path else None
    cap_embs, just_embs = validate_similarity_inputs(
        df,
        cap_embs,
        just_embs,
        id_index,
        demo_cols,
        allow_missing_embeddings=allow_missing_embeddings,
    )

    input_metadata = within_cross_input_metadata(
        df,
        cap_embs,
        just_embs,
        id_index,
        demo_cols,
        allow_missing_embeddings=allow_missing_embeddings,
    )
    cached = _read_valid_cache(cache_path, input_metadata)
    if cached is not None:
        return cached

    rows = []
    for img_id, grp in df.groupby("image_id"):
        grp_r = grp.loc[grp["annotation_id"].isin(id_index)].reset_index(drop=True)
        idxs = [id_index[a] for a in grp_r["annotation_id"]]
        if len(idxs) < 2:
            continue
        cap_sub = cap_embs[idxs]
        just_sub = just_embs[idxs]
        sim_cap = cosine_similarity(cap_sub)
        sim_just = cosine_similarity(just_sub)
        tag_sets = [
            parse_perception_tags(v, annotation_id)
            for v, annotation_id in zip(
                grp_r["predicted_perceptions"], grp_r["annotation_id"], strict=True
            )
        ]

        for dim in demo_cols:
            labels = grp_r[dim].tolist()
            within_cap, cross_cap = [], []
            within_just, cross_just = [], []
            within_jac, cross_jac = [], []
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    same = labels[i] == labels[j]
                    tags_i, tags_j = tag_sets[i], tag_sets[j]
                    if same:
                        within_cap.append(sim_cap[i, j])
                        within_just.append(sim_just[i, j])
                        if tags_i is not None and tags_j is not None:
                            within_jac.append(jaccard(tags_i, tags_j, on_empty_union=0.0))
                    else:
                        cross_cap.append(sim_cap[i, j])
                        cross_just.append(sim_just[i, j])
                        if tags_i is not None and tags_j is not None:
                            cross_jac.append(jaccard(tags_i, tags_j, on_empty_union=0.0))

            for modality, within, cross in [
                ("caption", within_cap, cross_cap),
                ("justification", within_just, cross_just),
                ("perception", within_jac, cross_jac),
            ]:
                rows.append(
                    {
                        "image_id": img_id,
                        "dimension": dim,
                        "modality": modality,
                        "within_mean": np.mean(within) if within else np.nan,
                        "cross_mean": np.mean(cross) if cross else np.nan,
                    }
                )

    result = pd.DataFrame(rows)
    if cache_path:
        result.to_csv(cache_path, index=False)
        metadata = {
            **input_metadata,
            "csv_sha256": sha256_file(cache_path),
        }
        within_cross_cache_metadata_path(cache_path).write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
    return result

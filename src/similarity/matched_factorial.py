"""Matched-factorial within/cross similarity: one persona dimension varied at a time.

The marginal construction in :mod:`.within_cross` buckets annotation pairs on a
single attribute while the other three vary freely, so a cross-group pair may
differ on more than the attribute under study. Here every contrast is formed
between two *full* persona profiles that agree on the other three attributes and
differ only on the target one, which isolates that attribute's contribution.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..hashing import sha256_file
from ..schema import DEMO_COLS, MODALITIES
from .cache_metadata import (
    validate_within_cross_cache_metadata,
    within_cross_cache_metadata_path,
    within_cross_input_metadata,
)
from .perception import parse_perception_tags
from .within_cross import validate_similarity_inputs

MATCHED_FACTORIAL_CONSTRUCTION = "matched_profile_pairs_one_dimension_varied"
MATCHED_FACTORIAL_SEED_NAMESPACE = "matched_factorial"
MATCHED_FACTORIAL_CONTRAST_SEED_NAMESPACE = "matched_factorial_contrast"

MATCHED_FACTORIAL_FRAME_COLUMNS = [
    "image_id",
    "dimension",
    "modality",
    "within_mean",
    "cross_mean",
    "n_matched_pairs",
]


def persona_profiles(
    df: pd.DataFrame,
    demo_cols: list[str] = DEMO_COLS,
) -> list[tuple[str, ...]]:
    """Enumerate the distinct full persona profiles present in a corpus.

    Args:
        df: Annotation frame carrying the demographic columns.
        demo_cols: Demographic columns, in compute order.

    Returns:
        The distinct profiles as attribute tuples, sorted for determinism.
    """
    values = df.loc[:, demo_cols].astype(str)
    return sorted({tuple(row) for row in values.itertuples(index=False, name=None)})


def matched_profile_pairs(
    profiles: list[tuple[str, ...]],
    dimension: str,
    demo_cols: list[str] = DEMO_COLS,
) -> list[tuple[int, int]]:
    """Index the profile pairs that differ only on one dimension.

    Args:
        profiles: Distinct persona profiles, as returned by :func:`persona_profiles`.
        dimension: The demographic dimension allowed to differ.
        demo_cols: Demographic columns, in the order the profile tuples use.

    Returns:
        Sorted ``(first, second)`` index pairs into ``profiles``.

    Raises:
        ValueError: If the dimension is unknown or the profiles are not distinct.
    """
    if dimension not in demo_cols:
        raise ValueError(f"unknown persona dimension: {dimension!r}")
    if len(set(profiles)) != len(profiles):
        raise ValueError("persona profiles must be distinct")

    target = list(demo_cols).index(dimension)
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, profile in enumerate(profiles):
        key = profile[:target] + profile[target + 1 :]
        groups.setdefault(key, []).append(index)

    pairs: list[tuple[int, int]] = []
    for key in sorted(groups):
        pairs.extend(combinations(sorted(groups[key]), 2))
    return pairs


def _profile_cosine_block(
    embeddings: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Mean cosine similarity within and between profiles, for one image.

    Args:
        embeddings: Rows ordered so that each profile's rows are contiguous.
        starts: First row index of each profile.
        counts: Number of rows belonging to each profile.

    Returns:
        A square profile-by-profile matrix, ``nan`` where a mean is undefined.

    Raises:
        ValueError: If any embedding has zero or non-finite norm.
    """
    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0.0).any():
        raise ValueError("matched-factorial input contains a zero or non-finite embedding")
    unit = values / norms

    n_profiles = len(counts)
    sums = np.zeros((n_profiles, unit.shape[1]), dtype=np.float64)
    self_sums = np.zeros(n_profiles, dtype=np.float64)
    for profile, (start, count) in enumerate(zip(starts, counts, strict=True)):
        if count == 0:
            continue
        chunk = unit[start : start + count]
        sums[profile] = chunk.sum(axis=0)
        self_sums[profile] = float(np.einsum("ij,ij->i", chunk, chunk).sum())

    gram = sums @ sums.T
    sizes = np.asarray(counts, dtype=np.float64)
    pair_counts = np.outer(sizes, sizes)
    block = np.full((n_profiles, n_profiles), np.nan)
    usable = pair_counts > 0
    block[usable] = gram[usable] / pair_counts[usable]

    diagonal = np.arange(n_profiles)
    coherent = sizes >= 2
    block[diagonal, diagonal] = np.nan
    block[diagonal[coherent], diagonal[coherent]] = (
        np.diag(gram)[coherent] - self_sums[coherent]
    ) / (sizes[coherent] * (sizes[coherent] - 1.0))
    return block


def _profile_jaccard_block(
    tag_sets: list[set[str] | None],
    profile_index: np.ndarray,
    n_profiles: int,
) -> np.ndarray:
    """Mean Jaccard overlap within and between profiles, for one image.

    Annotations whose perception response was empty are treated as missing
    rather than as empty sets, matching :func:`.parse_perception_tags`, so they
    take no part in any pair.

    Args:
        tag_sets: Parsed perception tags, one per annotation, ``None`` when missing.
        profile_index: Profile index of each annotation.
        n_profiles: Total number of persona profiles.

    Returns:
        A square profile-by-profile matrix, ``nan`` where a mean is undefined.
    """
    kept = [
        (int(profile), tags) for profile, tags in zip(profile_index, tag_sets, strict=True) if tags
    ]
    block = np.full((n_profiles, n_profiles), np.nan)
    if not kept:
        return block

    vocabulary = sorted({tag for _, tags in kept for tag in tags})
    column_of = {tag: column for column, tag in enumerate(vocabulary)}
    rows: list[int] = []
    columns: list[int] = []
    for position, (_, tags) in enumerate(kept):
        for tag in sorted(tags):
            rows.append(position)
            columns.append(column_of[tag])

    incidence = csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, columns)),
        shape=(len(kept), len(vocabulary)),
    )
    intersection = (incidence @ incidence.T).toarray().astype(np.float64)
    sizes = np.asarray(incidence.sum(axis=1)).ravel().astype(np.float64)
    overlap = intersection / (sizes[:, None] + sizes[None, :] - intersection)

    kept_profiles = np.asarray([profile for profile, _ in kept])
    members = [np.flatnonzero(kept_profiles == profile) for profile in range(n_profiles)]
    for first in range(n_profiles):
        if len(members[first]) >= 2:
            square = overlap[np.ix_(members[first], members[first])]
            block[first, first] = float(square[np.triu_indices(len(members[first]), k=1)].mean())
        for second in range(first + 1, n_profiles):
            if len(members[first]) and len(members[second]):
                value = float(overlap[np.ix_(members[first], members[second])].mean())
                block[first, second] = value
                block[second, first] = value
    return block


def _matched_pair_means(
    block: np.ndarray,
    pairs: list[tuple[int, int]],
) -> tuple[float, float, int]:
    """Mean matched within and cross similarity over the usable profile pairs.

    A pair leaves *both* terms whenever any of its three similarities is
    undefined. The two means therefore average over an identical pair set, which
    is what makes their difference equal the mean of the per-pair contrasts.

    Args:
        block: Profile-by-profile similarity matrix for one image and modality.
        pairs: Matched profile index pairs for one dimension.

    Returns:
        The matched within mean, the matched cross mean, and the pair count.
    """
    within: list[float] = []
    cross: list[float] = []
    for first, second in pairs:
        values = (block[first, first], block[second, second], block[first, second])
        if not np.isfinite(values).all():
            continue
        within.append((float(values[0]) + float(values[1])) / 2.0)
        cross.append(float(values[2]))
    if not within:
        return float("nan"), float("nan"), 0
    return float(np.mean(within)), float(np.mean(cross)), len(within)


def _read_matched_cache(
    cache_path: Path | None,
    input_metadata: dict[str, object],
) -> pd.DataFrame | None:
    """Return a cached frame only when its provenance and contents still hold.

    Args:
        cache_path: Location of the cached CSV, or ``None`` to skip caching.
        input_metadata: Provenance the sidecar must agree with.

    Returns:
        The cached frame, or ``None`` when it cannot be trusted.
    """
    if cache_path is None or not cache_path.exists():
        return None
    try:
        validate_within_cross_cache_metadata(cache_path, input_metadata)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    cached = pd.read_csv(cache_path)
    if not set(MATCHED_FACTORIAL_FRAME_COLUMNS).issubset(cached.columns):
        return None
    if cached.duplicated(["image_id", "dimension", "modality"]).any():
        return None
    values = cached[["within_mean", "cross_mean"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    return cached


def _write_matched_cache(
    result: pd.DataFrame,
    cache_path: Path,
    input_metadata: dict[str, object],
) -> None:
    """Write the frame and the provenance sidecar that guards it.

    Args:
        result: Frame to cache.
        cache_path: Destination CSV path.
        input_metadata: Provenance to record alongside the content digest.
    """
    result.to_csv(cache_path, index=False)
    metadata = {**input_metadata, "csv_sha256": sha256_file(cache_path)}
    within_cross_cache_metadata_path(cache_path).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _image_profile_blocks(
    group: pd.DataFrame,
    cap_embs: np.ndarray,
    just_embs: np.ndarray,
    id_index: dict[str, int],
    *,
    profile_index: np.ndarray,
    n_profiles: int,
) -> dict[str, np.ndarray]:
    """Build the three profile-by-profile similarity matrices for one image.

    Args:
        group: The annotations belonging to a single image.
        cap_embs: Caption embedding matrix.
        just_embs: Justification embedding matrix.
        id_index: Bijection from annotation ID to embedding row.
        profile_index: Profile index of each annotation in ``group``.
        n_profiles: Total number of persona profiles.

    Returns:
        One matrix per modality, keyed by modality name.
    """
    order = np.argsort(profile_index, kind="stable")
    counts = np.bincount(profile_index[order], minlength=n_profiles)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)[:-1]))
    embedding_rows = np.fromiter(
        (id_index[annotation_id] for annotation_id in group["annotation_id"]),
        dtype=np.int64,
        count=len(group),
    )[order]
    tag_sets = [
        parse_perception_tags(value, annotation_id)
        for value, annotation_id in zip(
            group["predicted_perceptions"],
            group["annotation_id"],
            strict=True,
        )
    ]
    return {
        "caption": _profile_cosine_block(cap_embs[embedding_rows], starts, counts),
        "justification": _profile_cosine_block(just_embs[embedding_rows], starts, counts),
        "perception": _profile_jaccard_block(tag_sets, profile_index, n_profiles),
    }


def compute_matched_factorial_similarity(
    df: pd.DataFrame,
    cap_embs: np.ndarray,
    just_embs: np.ndarray,
    id_index: dict[str, int],
    *,
    demo_cols: list[str] = DEMO_COLS,
    cache_path: Path | None = None,
    allow_missing_embeddings: bool = False,
) -> pd.DataFrame:
    """Matched-pair within/cross similarity holding the other attributes fixed.

    For every image, dimension, and pair of profiles that differ only on that
    dimension, the matched within term averages the two profiles' own coherence
    and the matched cross term is their between-profile similarity. Averaging
    over matched pairs leaves one paired value per image, which the existing
    image-level inference consumes unchanged.

    Args:
        df: Annotation frame carrying the ID, image, perception, and demographic columns.
        cap_embs: Caption embedding matrix.
        just_embs: Justification embedding matrix.
        id_index: Bijection from annotation ID to embedding row.
        demo_cols: Demographic columns, in compute order.
        cache_path: Read-through cache location, or ``None`` to always recompute.
        allow_missing_embeddings: Permit annotations that have no embedding row.

    Returns:
        DataFrame with columns image_id, dimension, modality, within_mean,
        cross_mean, n_matched_pairs.
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

    input_metadata = {
        **within_cross_input_metadata(
            df,
            cap_embs,
            just_embs,
            id_index,
            demo_cols,
            allow_missing_embeddings=allow_missing_embeddings,
        ),
        "construction": MATCHED_FACTORIAL_CONSTRUCTION,
    }
    cached = _read_matched_cache(cache_path, input_metadata)
    if cached is not None:
        return cached

    profiles = persona_profiles(df, demo_cols)
    index_of = {profile: index for index, profile in enumerate(profiles)}
    pairs_by_dimension = {
        dimension: matched_profile_pairs(profiles, dimension, demo_cols) for dimension in demo_cols
    }

    rows = []
    for image_id, group in df.groupby("image_id"):
        present = group.loc[group["annotation_id"].isin(id_index)]
        if len(present) < 2:
            continue
        profile_index = np.fromiter(
            (
                index_of[key]
                for key in present.loc[:, demo_cols].astype(str).itertuples(index=False, name=None)
            ),
            dtype=np.int64,
            count=len(present),
        )
        blocks = _image_profile_blocks(
            present,
            cap_embs,
            just_embs,
            id_index,
            profile_index=profile_index,
            n_profiles=len(profiles),
        )
        for dimension in demo_cols:
            for modality in MODALITIES:
                within_mean, cross_mean, n_pairs = _matched_pair_means(
                    blocks[modality],
                    pairs_by_dimension[dimension],
                )
                rows.append(
                    {
                        "image_id": image_id,
                        "dimension": dimension,
                        "modality": modality,
                        "within_mean": within_mean,
                        "cross_mean": cross_mean,
                        "n_matched_pairs": n_pairs,
                    }
                )

    result = pd.DataFrame(rows, columns=MATCHED_FACTORIAL_FRAME_COLUMNS)
    if cache_path:
        _write_matched_cache(result, cache_path, input_metadata)
    return result

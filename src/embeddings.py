"""Sentence embedding: encode or load from cache."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import EMBED_BATCH_SIZE, EMBED_MODEL


def _validate_id_alignment(
    id_df: pd.DataFrame,
    expected_ids: list | None = None,
) -> None:
    """Validate the ordered row identifiers stored beside an embedding cache."""
    if "annotation_id" not in id_df.columns:
        raise ValueError("embedding ID cache must contain an annotation_id column")
    if id_df["annotation_id"].isna().any():
        raise ValueError("embedding ID cache contains missing annotation IDs")
    if id_df["annotation_id"].duplicated().any():
        raise ValueError("embedding ID cache contains duplicate annotation IDs")
    if expected_ids is not None and id_df["annotation_id"].tolist() != expected_ids:
        raise ValueError("embedding ID cache is not in the same row order as the annotation frame")


def _validate_embedding_alignment(
    embs: np.ndarray,
    id_df: pd.DataFrame,
    expected_ids: list | None = None,
) -> None:
    """Fail if an embedding matrix and its row-ID sidecar can drift apart."""
    _validate_id_alignment(id_df, expected_ids)
    if np.asarray(embs).ndim != 2:
        raise ValueError("embedding cache must be a two-dimensional array")
    if len(id_df) != len(embs):
        raise ValueError(
            "embedding cache and ID cache have different row counts: "
            f"{len(embs)} versus {len(id_df)}"
        )
    if not np.isfinite(embs).all():
        raise ValueError("embedding cache contains non-finite values")


def load_or_encode(
    texts: list,
    cache_path: Path,
    model_name: str = EMBED_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    """Return normalized embeddings from cache or encode with SentenceTransformer."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(cache_path, embs)
    return embs


def load_or_encode_with_ids(
    df: pd.DataFrame,
    text_col: str,
    emb_cache: Path,
    id_cache: Path,
    *,
    model_name: str = EMBED_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode *text_col* and also save/load an annotation_id → row-index mapping.

    Returns:
        embs:     (N, D) float32 array
        id_index: dict mapping annotation_id to embedding row index
    """
    emb_cache = Path(emb_cache)
    id_cache = Path(id_cache)

    if emb_cache.exists() and not id_cache.exists():
        raise FileNotFoundError(
            "an embedding cache exists without its row-ID sidecar; regenerate "
            f"{emb_cache} together with {id_cache}"
        )
    if "annotation_id" not in df.columns:
        raise ValueError("annotation frame must contain an annotation_id column")
    expected_ids = df["annotation_id"].tolist()
    if pd.Series(expected_ids).isna().any() or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("annotation IDs must be non-missing and unique")

    if id_cache.exists():
        id_df = pd.read_csv(id_cache)
        _validate_id_alignment(id_df, expected_ids)
    else:
        id_df = df[["annotation_id"]].reset_index(drop=True)
        id_df.to_csv(id_cache, index=False)

    if emb_cache.exists():
        embs = np.load(emb_cache)
    else:
        texts = df[text_col].tolist()
        embs = load_or_encode(texts, emb_cache, model_name, batch_size)

    _validate_embedding_alignment(embs, id_df, expected_ids)
    id_index = {aid: i for i, aid in enumerate(id_df["annotation_id"])}
    return embs, id_index


def build_id_index(id_cache: Path) -> dict[str, int]:
    """Build annotation_id → row-index mapping from a saved CSV."""
    id_df = pd.read_csv(id_cache)
    _validate_id_alignment(id_df)
    return {aid: i for i, aid in enumerate(id_df["annotation_id"])}

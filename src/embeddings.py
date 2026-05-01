"""Sentence embedding: encode or load from cache."""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import EMBED_MODEL, EMBED_BATCH_SIZE


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
    model_name: str = EMBED_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Encode *text_col* and also save/load an annotation_id → row-index mapping.

    Returns:
        embs:     (N, D) float32 array
        id_index: dict mapping annotation_id to embedding row index
    """
    emb_cache = Path(emb_cache)
    id_cache  = Path(id_cache)

    if emb_cache.exists() and id_cache.exists():
        embs    = np.load(emb_cache)
        id_df   = pd.read_csv(id_cache)
    else:
        texts = df[text_col].tolist()
        embs  = load_or_encode(texts, emb_cache, model_name, batch_size)
        id_df = df[["annotation_id"]].reset_index(drop=True)
        id_df.to_csv(id_cache, index=False)

    id_index = {aid: i for i, aid in enumerate(id_df["annotation_id"])}
    return embs, id_index


def build_id_index(id_cache: Path) -> Dict[str, int]:
    """Build annotation_id → row-index mapping from a saved CSV."""
    id_df = pd.read_csv(id_cache)
    return {aid: i for i, aid in enumerate(id_df["annotation_id"])}

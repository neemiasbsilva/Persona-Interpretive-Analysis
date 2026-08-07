"""Per-image mean pairwise cosine similarity across persona agents."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def compute_per_image_similarity(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    id_index: dict[str, int],
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Mean pairwise cosine similarity of caption (or justification) embeddings per image.

    Returns DataFrame with columns:
        image_id, mean_sim, std_sim, min_sim, max_sim, n_personas
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        return pd.read_csv(cache_path)

    rows = []
    for img_id, grp in df.groupby("image_id"):
        idxs = [id_index[a] for a in grp["annotation_id"] if a in id_index]
        if len(idxs) < 2:
            continue
        embs = embeddings[idxs]
        sim = cosine_similarity(embs)
        tri = sim[np.triu_indices(len(idxs), k=1)]
        rows.append(
            {
                "image_id": img_id,
                "mean_sim": tri.mean(),
                "std_sim": tri.std(),
                "min_sim": tri.min(),
                "max_sim": tri.max(),
                "n_personas": len(idxs),
            }
        )

    result = pd.DataFrame(rows)
    if cache_path:
        result.to_csv(cache_path, index=False)
    return result

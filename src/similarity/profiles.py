"""Within-profile coherence and image-conditioned inter-profile similarity."""

import ast
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..metrics import jaccard


def _to_set(pp: object) -> set[str]:
    """Parse one perception response leniently, treating anything unusable as empty.

    Args:
        pp: A tag list, or its repr as stored in CSV.

    Returns:
        The tag set, empty when the value is missing or malformed.
    """
    if isinstance(pp, list):
        return set(pp)
    if isinstance(pp, str):
        try:
            return set(ast.literal_eval(pp))
        except (ValueError, SyntaxError, TypeError):
            return set()
    return set()


def compute_profile_coherence(
    df: pd.DataFrame,
    cap_embs: np.ndarray,
    just_embs: np.ndarray,
    id_index: dict[str, int],
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Median within-profile similarity for each of the 24 full persona profiles.

    Covers pairwise cosine similarity of captions and justifications, and
    Jaccard similarity of perception tags.

    Returns:
        DataFrame with columns profile, n_agents, caption_coherence,
        justification_coherence, perception_jaccard.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        return pd.read_csv(cache_path)

    rows = []
    for profile, grp in df.groupby("profile"):
        aids = grp["annotation_id"].tolist()
        idxs = [id_index[a] for a in aids if a in id_index]
        if len(idxs) < 2:
            continue
        cap_sub = cap_embs[idxs]
        just_sub = just_embs[idxs]
        tri = np.triu_indices(len(idxs), k=1)
        cap_med = float(np.median(cosine_similarity(cap_sub)[tri]))
        just_med = float(np.median(cosine_similarity(just_sub)[tri]))

        img_jac_medians = []
        for _, img_grp in grp.groupby("image_id"):
            img_sets = [_to_set(pp) for pp in img_grp["predicted_perceptions"]]
            if len(img_sets) < 2:
                continue
            jac_vals = []
            for a, b in combinations(img_sets, 2):
                jac_vals.append(jaccard(a, b, on_empty_union=0.0))
            if jac_vals:
                img_jac_medians.append(float(np.median(jac_vals)))
        perc_jac = float(np.median(img_jac_medians)) if img_jac_medians else float("nan")

        rows.append(
            {
                "profile": profile,
                "n_agents": len(idxs),
                "caption_coherence": cap_med,
                "justification_coherence": just_med,
                "perception_jaccard": perc_jac,
            }
        )

    result = pd.DataFrame(rows)
    if cache_path:
        result.to_csv(cache_path, index=False)
    return result


def compute_image_conditioned_profile_sim(
    df: pd.DataFrame,
    cap_embs: np.ndarray,
    just_embs: np.ndarray,
    id_index: dict[str, int],
    *,
    cap_cache: Path | None = None,
    just_cache: Path | None = None,
    labs_cache: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """24x24 image-conditioned cross-profile cosine similarity matrix.

    For each profile pair (p1, p2), compute cosine similarity between their
    embeddings on the SAME image, then average over all shared images.
    This controls for visual content and reveals semantic agreement between
    profiles rather than general embedding-space proximity.

    Diagonal = mean self-similarity (≈ 1.0 for normalized embeddings).

    Returns:
        cap_mat:   (24, 24) mean image-conditioned cosine sim (captions)
        just_mat:  (24, 24) mean image-conditioned cosine sim (justifications)
        prof_labs: list of 24 profile strings (row/col labels)
    """
    cap_cache = Path(cap_cache) if cap_cache else None
    just_cache = Path(just_cache) if just_cache else None
    labs_cache = Path(labs_cache) if labs_cache else None

    if (
        cap_cache
        and cap_cache.exists()
        and just_cache
        and just_cache.exists()
        and labs_cache
        and labs_cache.exists()
    ):
        return (
            np.load(cap_cache),
            np.load(just_cache),
            pd.read_csv(labs_cache)["profile"].tolist(),
        )

    profiles = sorted(df["profile"].unique())
    n = len(profiles)

    cap_by = {p: {} for p in profiles}
    just_by = {p: {} for p in profiles}
    cap_diag_vals = {p: [] for p in profiles}
    just_diag_vals = {p: [] for p in profiles}
    for (prof, img), grp in df.groupby(["profile", "image_id"]):
        aids = [a for a in grp["annotation_id"].tolist() if a in id_index]
        if not aids:
            continue
        idxs = [id_index[a] for a in aids]
        ce = cap_embs[idxs]
        je = just_embs[idxs]
        cap_by[prof][img] = ce.mean(axis=0)
        just_by[prof][img] = je.mean(axis=0)
        if len(idxs) >= 2:
            tri = np.triu_indices(len(idxs), k=1)
            cap_diag_vals[prof].append(float(np.dot(ce, ce.T)[tri].mean()))
            just_diag_vals[prof].append(float(np.dot(je, je.T)[tri].mean()))

    cap_mat = np.zeros((n, n))
    just_mat = np.zeros((n, n))

    for i, p1 in enumerate(profiles):
        for j, p2 in enumerate(profiles):
            if i == j:
                cap_mat[i, j] = (
                    float(np.mean(cap_diag_vals[p1])) if cap_diag_vals[p1] else float("nan")
                )
                just_mat[i, j] = (
                    float(np.mean(just_diag_vals[p1])) if just_diag_vals[p1] else float("nan")
                )
                continue
            shared = sorted(set(cap_by[p1]) & set(cap_by[p2]))
            if not shared:
                cap_mat[i, j] = just_mat[i, j] = float("nan")
                continue
            e1c = np.array([cap_by[p1][img] for img in shared])
            e2c = np.array([cap_by[p2][img] for img in shared])
            e1j = np.array([just_by[p1][img] for img in shared])
            e2j = np.array([just_by[p2][img] for img in shared])
            cap_mat[i, j] = float(np.einsum("ij,ij->i", e1c, e2c).mean())
            just_mat[i, j] = float(np.einsum("ij,ij->i", e1j, e2j).mean())

    if cap_cache:
        np.save(cap_cache, cap_mat)
    if just_cache:
        np.save(just_cache, just_mat)
    if labs_cache:
        pd.DataFrame({"profile": profiles}).to_csv(labs_cache, index=False)

    return cap_mat, just_mat, profiles

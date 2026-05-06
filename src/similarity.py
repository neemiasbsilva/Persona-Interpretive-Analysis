"""Cosine similarity computations: per-image, within/cross-persona, profile matrices."""
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from .config import OUTPUTS, PROFILE_SIM_N_SAMPLE


def _pairwise_mean(embs: np.ndarray) -> float:
    """Mean of the upper-triangle of a cosine similarity matrix."""
    sim = cosine_similarity(embs)
    tri = np.triu_indices(len(embs), k=1)
    return float(sim[tri].mean())


# ── Per-image similarity ───────────────────────────────────────────────────────

def compute_per_image_similarity(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    id_index: Dict[str, int],
    cache_path: Optional[Path] = None,
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
        sim  = cosine_similarity(embs)
        tri  = sim[np.triu_indices(len(idxs), k=1)]
        rows.append({
            "image_id": img_id,
            "mean_sim": tri.mean(),
            "std_sim":  tri.std(),
            "min_sim":  tri.min(),
            "max_sim":  tri.max(),
            "n_personas": len(idxs),
        })

    result = pd.DataFrame(rows)
    if cache_path:
        result.to_csv(cache_path, index=False)
    return result


def compute_per_persona_similarity(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    id_index: Dict[str, int],
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """For each agent, compute mean cosine similarity to all other agents on the same image.

    Returns DataFrame with columns: persona_id, image_id, mean_sim_vs_others
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        return pd.read_csv(cache_path)

    rows = []
    df_r = df.reset_index(drop=True)
    for img_id, grp in df_r.groupby("image_id"):
        idxs = [id_index[a] for a in grp["annotation_id"] if a in id_index]
        if len(idxs) < 2:
            continue
        embs = embeddings[idxs]
        sim  = cosine_similarity(embs)
        for local_i, (_, row) in enumerate(grp.iterrows()):
            others = [sim[local_i, j] for j in range(len(idxs)) if j != local_i]
            rows.append({
                "persona_id":       row["persona_id"],
                "image_id":         img_id,
                "mean_sim_vs_others": float(np.mean(others)),
            })

    result = pd.DataFrame(rows)
    if cache_path:
        result.to_csv(cache_path, index=False)
    return result


# ── Within vs. cross-persona similarity ───────────────────────────────────────

_DEMO_COLS = ["gender", "economic_status", "political_spectrum", "personality"]


def compute_within_cross_similarity(
    df: pd.DataFrame,
    cap_embs: np.ndarray,
    just_embs: np.ndarray,
    id_index: Dict[str, int],
    demo_cols: list = _DEMO_COLS,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Within-group vs. cross-group similarity per demographic dimension.

    Covers three modalities:
      - caption       : cosine similarity of Sentence-BERT embeddings
      - justification : cosine similarity of Sentence-BERT embeddings
      - perception    : Jaccard similarity of perception tag sets

    Returns DataFrame with columns:
        image_id, dimension, modality, within_mean, cross_mean
    """
    import ast as _ast

    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        return pd.read_csv(cache_path)

    def _to_set(v):
        if isinstance(v, list):
            return set(v)
        try:
            return set(_ast.literal_eval(v))
        except Exception:
            return set()

    def _jaccard(a, b):
        u = len(a | b)
        return len(a & b) / u if u > 0 else 0.0

    rows = []
    for img_id, grp in df.groupby("image_id"):
        idxs = [id_index[a] for a in grp["annotation_id"] if a in id_index]
        if len(idxs) < 2:
            continue
        grp_r    = grp.reset_index(drop=True)
        cap_sub  = cap_embs[idxs]
        just_sub = just_embs[idxs]
        sim_cap  = cosine_similarity(cap_sub)
        sim_just = cosine_similarity(just_sub)
        tag_sets = [_to_set(v) for v in grp_r["predicted_perceptions"]]

        for dim in demo_cols:
            labels = grp_r[dim].tolist()
            within_cap, cross_cap   = [], []
            within_just, cross_just = [], []
            within_jac, cross_jac   = [], []
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    same = labels[i] == labels[j]
                    if same:
                        within_cap.append(sim_cap[i, j])
                        within_just.append(sim_just[i, j])
                        within_jac.append(_jaccard(tag_sets[i], tag_sets[j]))
                    else:
                        cross_cap.append(sim_cap[i, j])
                        cross_just.append(sim_just[i, j])
                        cross_jac.append(_jaccard(tag_sets[i], tag_sets[j]))

            for modality, within, cross in [
                ("caption",       within_cap,  cross_cap),
                ("justification", within_just, cross_just),
                ("perception",    within_jac,  cross_jac),
            ]:
                rows.append({
                    "image_id":   img_id,
                    "dimension":  dim,
                    "modality":   modality,
                    "within_mean": np.mean(within) if within else np.nan,
                    "cross_mean":  np.mean(cross)  if cross  else np.nan,
                })

    result = pd.DataFrame(rows)
    if cache_path:
        result.to_csv(cache_path, index=False)
    return result


# ── Within-persona profile coherence ─────────────────────────────────────────

def compute_profile_coherence(
    df: pd.DataFrame,
    cap_embs: np.ndarray,
    just_embs: np.ndarray,
    id_index: Dict[str, int],
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """For each of the 24 full persona profiles, compute median pairwise cosine
    similarity (captions and justifications) and median Jaccard similarity of
    perception tags.

    Returns DataFrame with columns:
        profile, n_agents, caption_coherence, justification_coherence, perception_jaccard
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
        cap_sub  = cap_embs[idxs]
        just_sub = just_embs[idxs]
        tri = np.triu_indices(len(idxs), k=1)
        cap_med  = float(np.median(cosine_similarity(cap_sub)[tri]))
        just_med = float(np.median(cosine_similarity(just_sub)[tri]))

        import ast as _ast
        def _to_set(pp):
            if isinstance(pp, list):
                return set(pp)
            if isinstance(pp, str):
                try:
                    return set(_ast.literal_eval(pp))
                except Exception:
                    pass
            return set()

        img_jac_medians = []
        for _, img_grp in grp.groupby("image_id"):
            img_sets = [_to_set(pp) for pp in img_grp["predicted_perceptions"]]
            if len(img_sets) < 2:
                continue
            jac_vals = []
            for a, b in combinations(img_sets, 2):
                union = len(a | b)
                jac_vals.append(len(a & b) / union if union > 0 else 0.0)
            if jac_vals:
                img_jac_medians.append(float(np.median(jac_vals)))
        perc_jac = float(np.median(img_jac_medians)) if img_jac_medians else float("nan")

        rows.append({
            "profile":                profile,
            "n_agents":               len(idxs),
            "caption_coherence":      cap_med,
            "justification_coherence": just_med,
            "perception_jaccard":     perc_jac,
        })

    result = pd.DataFrame(rows)
    if cache_path:
        result.to_csv(cache_path, index=False)
    return result


# ── 24×24 inter-profile cosine similarity matrices ────────────────────────────

def compute_profile_sim_matrix(
    df: pd.DataFrame,
    cap_embs: np.ndarray,
    just_embs: np.ndarray,
    id_index: Dict[str, int],
    n_sample: int = PROFILE_SIM_N_SAMPLE,
    mean_cap_cache:   Optional[Path] = None,
    mean_just_cache:  Optional[Path] = None,
    med_cap_cache:    Optional[Path] = None,
    med_just_cache:   Optional[Path] = None,
    labs_cache:       Optional[Path] = None,
    # legacy aliases kept for backwards compatibility
    cap_cache:  Optional[Path] = None,
    just_cache: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """24×24 cosine similarity matrices (mean and median) between persona profiles.

    Diagonal = 1.0 for both aggregations.
    Off-diagonal = mean or median of all pairwise cosine similarities.

    Returns:
        mean_cap:   (24, 24) mean cosine similarity for captions
        mean_just:  (24, 24) mean cosine similarity for justifications
        med_cap:    (24, 24) median cosine similarity for captions
        med_just:   (24, 24) median cosine similarity for justifications
        prof_labs:  list of 24 profile strings (row/col labels)
    """
    # legacy alias support
    mean_cap_cache  = Path(mean_cap_cache  or cap_cache)  if (mean_cap_cache  or cap_cache)  else None
    mean_just_cache = Path(mean_just_cache or just_cache) if (mean_just_cache or just_cache) else None
    med_cap_cache   = Path(med_cap_cache)  if med_cap_cache  else None
    med_just_cache  = Path(med_just_cache) if med_just_cache else None
    labs_cache      = Path(labs_cache)     if labs_cache     else None

    caches_exist = all(
        c is not None and c.exists()
        for c in [mean_cap_cache, mean_just_cache, med_cap_cache, med_just_cache, labs_cache]
    )
    if caches_exist:
        return (
            np.load(mean_cap_cache),
            np.load(mean_just_cache),
            np.load(med_cap_cache),
            np.load(med_just_cache),
            pd.read_csv(labs_cache)["profile"].tolist(),
        )

    profiles = sorted(df["profile"].unique())
    n        = len(profiles)
    rng      = np.random.default_rng(42)
    ec, ej   = {}, {}

    for p in profiles:
        mask = (df["profile"] == p).values
        idx  = np.where(mask)[0]
        sel  = rng.choice(idx, min(n_sample, len(idx)), replace=False)
        aids = df.iloc[sel]["annotation_id"].tolist()
        emb_idxs = [id_index[a] for a in aids if a in id_index]
        ec[p] = cap_embs[emb_idxs]
        ej[p] = just_embs[emb_idxs]

    mean_cap  = np.zeros((n, n))
    mean_just = np.zeros((n, n))
    med_cap   = np.zeros((n, n))
    med_just  = np.zeros((n, n))

    for i, p1 in enumerate(profiles):
        for j, p2 in enumerate(profiles):
            if i == j:
                mean_cap[i, j] = med_cap[i, j]  = 1.0
                mean_just[i, j] = med_just[i, j] = 1.0
            else:
                sim_c = cosine_similarity(ec[p1], ec[p2]).ravel()
                sim_j = cosine_similarity(ej[p1], ej[p2]).ravel()
                mean_cap[i, j]  = float(sim_c.mean())
                mean_just[i, j] = float(sim_j.mean())
                med_cap[i, j]   = float(np.median(sim_c))
                med_just[i, j]  = float(np.median(sim_j))

    if mean_cap_cache:  np.save(mean_cap_cache,  mean_cap)
    if mean_just_cache: np.save(mean_just_cache, mean_just)
    if med_cap_cache:   np.save(med_cap_cache,   med_cap)
    if med_just_cache:  np.save(med_just_cache,  med_just)
    if labs_cache:
        pd.DataFrame({"profile": profiles}).to_csv(labs_cache, index=False)

    return mean_cap, mean_just, med_cap, med_just, profiles


def compute_image_conditioned_profile_sim(
    df: pd.DataFrame,
    cap_embs: np.ndarray,
    just_embs: np.ndarray,
    id_index: Dict[str, int],
    cap_cache:  Optional[Path] = None,
    just_cache: Optional[Path] = None,
    labs_cache: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """24×24 image-conditioned cross-profile cosine similarity matrix.

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
    cap_cache  = Path(cap_cache)  if cap_cache  else None
    just_cache = Path(just_cache) if just_cache else None
    labs_cache = Path(labs_cache) if labs_cache else None

    if (cap_cache  and cap_cache.exists() and
            just_cache and just_cache.exists() and
            labs_cache and labs_cache.exists()):
        return (
            np.load(cap_cache),
            np.load(just_cache),
            pd.read_csv(labs_cache)["profile"].tolist(),
        )

    profiles = sorted(df["profile"].unique())
    n = len(profiles)

    # Build profile → {image_id → mean_embedding} lookup
    # Also accumulate per-image within-profile pairwise sims for the diagonal.
    cap_by       = {p: {} for p in profiles}
    just_by      = {p: {} for p in profiles}
    cap_diag_vals  = {p: [] for p in profiles}
    just_diag_vals = {p: [] for p in profiles}
    for (prof, img), grp in df.groupby(["profile", "image_id"]):
        aids    = [a for a in grp["annotation_id"].tolist() if a in id_index]
        if not aids:
            continue
        idxs = [id_index[a] for a in aids]
        ce = cap_embs[idxs]
        je = just_embs[idxs]
        cap_by[prof][img]  = ce.mean(axis=0)
        just_by[prof][img] = je.mean(axis=0)
        if len(idxs) >= 2:
            tri = np.triu_indices(len(idxs), k=1)
            cap_diag_vals[prof].append(float(np.dot(ce, ce.T)[tri].mean()))
            just_diag_vals[prof].append(float(np.dot(je, je.T)[tri].mean()))

    cap_mat  = np.zeros((n, n))
    just_mat = np.zeros((n, n))

    for i, p1 in enumerate(profiles):
        for j, p2 in enumerate(profiles):
            if i == j:
                cap_mat[i, j]  = float(np.mean(cap_diag_vals[p1]))  if cap_diag_vals[p1]  else float("nan")
                just_mat[i, j] = float(np.mean(just_diag_vals[p1])) if just_diag_vals[p1] else float("nan")
                continue
            shared = list(set(cap_by[p1]) & set(cap_by[p2]))
            if not shared:
                cap_mat[i, j] = just_mat[i, j] = float("nan")
                continue
            e1c = np.array([cap_by[p1][img]  for img in shared])
            e2c = np.array([cap_by[p2][img]  for img in shared])
            e1j = np.array([just_by[p1][img] for img in shared])
            e2j = np.array([just_by[p2][img] for img in shared])
            # dot product = cosine sim for L2-normalised embeddings
            cap_mat[i, j]  = float(np.einsum("ij,ij->i", e1c, e2c).mean())
            just_mat[i, j] = float(np.einsum("ij,ij->i", e1j, e2j).mean())

    if cap_cache:  np.save(cap_cache,  cap_mat)
    if just_cache: np.save(just_cache, just_mat)
    if labs_cache:
        pd.DataFrame({"profile": profiles}).to_csv(labs_cache, index=False)

    return cap_mat, just_mat, profiles


def cluster_order(sim_matrix: np.ndarray) -> np.ndarray:
    """Return row/col ordering via Ward hierarchical clustering on 1-cosine distance."""
    dist = np.clip(1.0 - sim_matrix, 0, None)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    link = sch.linkage(ssd.squareform(dist), method="ward")
    return sch.leaves_list(link)

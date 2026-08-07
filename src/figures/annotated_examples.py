"""Locate representative annotation pairs to caption the annotated matrices."""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ..data_loading import create_profiles, load_annotations, parse_demographics
from ..embeddings import build_id_index
from ..metrics import lenient_tag_set
from ..models import annotations_path, outputs_dir

REP_PAIR_SEED = 7

SimFn = Callable[[pd.Series, pd.Series], float]
TextFn = Callable[[pd.Series], str]


def _abbrev_profile(profile: str) -> str:
    """Shorten one full profile string to its four-field abbreviation.

    Args:
        profile: Full ``gender / income / spectrum / personality`` string.

    Returns:
        The abbreviated label used on matrix axes.
    """
    pts = [x.strip() for x in profile.split("/")]
    return f"{pts[0][0]}/{pts[1][:4]}/{pts[2][:4]}/{pts[3][:4]}"


def _short_full_labels(model: str) -> tuple[list[str], list[str]]:
    """Return matrix-aligned profile labels.

    Args:
        model: Model directory id whose cached label order is used.

    Returns:
        The short abbreviations and the full profile strings, both in matrix
        index order.
    """
    full = pd.read_csv(outputs_dir(model) / "ic_profile_sim_labels.csv")["profile"].tolist()
    return [_abbrev_profile(p) for p in full], full


def _profile_short(row: pd.Series) -> str:
    """Build the short profile label for one annotation row.

    Args:
        row: Annotation row carrying the four demographic columns.

    Returns:
        The abbreviated profile label.
    """
    return (
        f"{row.gender[0]}/"
        f"{'High' if 'High' in row.economic_status else 'Low'}/"
        f"{'Cons' if 'Cons' in row.political_spectrum else 'Prog'}/"
        f"{row.personality[:4]}"
    )


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity, guarded against a zero norm.
    """
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _rep_pair(
    by_profile: dict[int, pd.DataFrame],
    i: int,
    j: int,
    *,
    sim_fn: SimFn,
    txt_fn: TextFn,
    target: float,
    lo_len: int,
    hi_len: int,
    seed: int = REP_PAIR_SEED,
) -> tuple[float, float, str, str, str, str] | None:
    """Find the shared-image annotation pair closest to a cell's value.

    Both outputs must be non-empty and within ``[lo_len, hi_len]`` characters.

    Args:
        by_profile: Annotation rows grouped by matrix index.
        i: Matrix row index.
        j: Matrix column index.
        sim_fn: Pairwise similarity for the modality being illustrated.
        txt_fn: Renders the displayed text for one row.
        target: Cell value the pair's similarity should sit closest to.
        lo_len: Minimum acceptable output length in characters.
        hi_len: Maximum acceptable output length in characters.
        seed: Base seed for the per-cell image shuffle.

    Returns:
        ``(distance, similarity, label_a, label_b, text_a, text_b)``, or None
        when the cell has no usable pair.
    """
    gi, gj = by_profile.get(i), by_profile.get(j)
    if gi is None or gj is None:
        return None
    imgs = sorted(set(gi.image_id) & set(gj.image_id))
    if not imgs:
        return None
    rng = np.random.default_rng(seed + i * 31 + j)
    rng.shuffle(imgs)

    best = None
    for img in imgs[:12]:
        ai = gi[gi.image_id == img]
        aj = gj[gj.image_id == img]
        ai = ai.sample(min(4, len(ai)), random_state=seed)
        aj = aj.sample(min(4, len(aj)), random_state=seed)
        for _, r1 in ai.iterrows():
            for _, r2 in aj.iterrows():
                if i == j and r1.annotation_id == r2.annotation_id:
                    continue
                ta, tb = txt_fn(r1), txt_fn(r2)
                if not ta or not tb:
                    continue
                if not (lo_len <= len(ta) <= hi_len and lo_len <= len(tb) <= hi_len):
                    continue
                s = sim_fn(r1, r2)
                key = abs(s - target)
                if best is None or key < best[0]:
                    best = (key, s, r1.short, r2.short, ta, tb)
    return best


def _band_examples(
    matrix: np.ndarray,
    by_profile: dict[int, pd.DataFrame],
    *,
    sim_fn: SimFn,
    txt_fn: TextFn,
    lo_len: int,
    hi_len: int,
    n: int,
) -> list[dict[str, Any]]:
    """Pick examples spanning the lower-triangle cell-value distribution.

    Targets sit at evenly spaced percentiles of the lower triangle, diagonal
    included, so the chosen cells span low to high similarity.

    Args:
        matrix: Square similarity matrix for one modality.
        by_profile: Annotation rows grouped by matrix index.
        sim_fn: Pairwise similarity for the modality being illustrated.
        txt_fn: Renders the displayed text for one row.
        lo_len: Minimum acceptable output length in characters.
        hi_len: Maximum acceptable output length in characters.
        n: Number of examples to return.

    Returns:
        One example dict per band that yielded a usable pair.
    """
    size = matrix.shape[0]
    tri = [matrix[i, j] for i in range(size) for j in range(i + 1)]
    targets = np.percentile(tri, np.linspace(5, 95, n))
    used: set[tuple[int, int]] = set()
    out: list[dict[str, Any]] = []
    for t in targets:
        cells = sorted(
            ((abs(matrix[i, j] - t), i, j) for i in range(size) for j in range(i + 1)),
            key=lambda x: x[0],
        )
        for _, i, j in cells:
            if (i, j) in used:
                continue
            rp = _rep_pair(
                by_profile,
                i,
                j,
                sim_fn=sim_fn,
                txt_fn=txt_fn,
                target=float(matrix[i, j]),
                lo_len=lo_len,
                hi_len=hi_len,
            )
            if rp is None:
                continue
            _, s, pa, pb, ta, tb = rp
            used.add((i, j))
            out.append(
                {
                    "i": i,
                    "j": j,
                    "pa": pa,
                    "pb": pb,
                    "ta": ta,
                    "tb": tb,
                    "sim": s,
                    "cellval": float(matrix[i, j]),
                }
            )
            break
    return out


def _caption_example(
    df: pd.DataFrame,
    caption_matrix: np.ndarray,
    sim_fn: SimFn,
) -> list[dict[str, Any]]:
    """Pick the most similar high-income/low-income caption pair.

    Args:
        df: Annotation frame with embedding indices and short labels.
        caption_matrix: Cached caption similarity matrix.
        sim_fn: Pairwise caption similarity.

    Returns:
        A single-element list, or an empty list when no pair qualifies.
    """
    best = None
    for _img, g in df.groupby("image_id"):
        hi = g[g.economic_status.str.contains("High")]
        lo = g[g.economic_status.str.contains("Low")]
        if hi.empty or lo.empty:
            continue
        hi = hi.sample(min(6, len(hi)), random_state=1)
        lo = lo.sample(min(6, len(lo)), random_state=1)
        for _, r1 in hi.iterrows():
            for _, r2 in lo.iterrows():
                if not (r1.cap and r2.cap) or r1.short == r2.short:
                    continue
                s = sim_fn(r1, r2)
                key = (-round(s, 3), len(r1.cap) + len(r2.cap))
                if best is None or key < best[0]:
                    best = (key, s, r1, r2)
    if best is None:
        return []

    _, s, r1, r2 = best
    i, j = max(r1.midx, r2.midx), min(r1.midx, r2.midx)
    a, b = (r1, r2) if r1.midx >= r2.midx else (r2, r1)
    return [
        {
            "i": i,
            "j": j,
            "pa": a.short,
            "pb": b.short,
            "ta": a.cap,
            "tb": b.cap,
            "sim": s,
            "cellval": float(caption_matrix[i, j]),
        }
    ]


def _load_example_frame(model: str, full_labels: list[str]) -> pd.DataFrame:
    """Load the annotation frame with matrix indices and embedding rows attached.

    Args:
        model: Model directory id to load.
        full_labels: Full profile strings in matrix index order.

    Returns:
        Annotations restricted to rows that have both a matrix index and an
        embedding row.
    """
    idx_of = {p: i for i, p in enumerate(full_labels)}
    df = create_profiles(parse_demographics(load_annotations(annotations_path(model))))
    df["short"] = df.apply(_profile_short, axis=1)
    df["midx"] = df.profile.map(idx_of)
    df = df.dropna(subset=["midx"])
    df["midx"] = df.midx.astype(int)
    df["cap"] = df.caption.fillna("")
    df["just"] = df.justification.fillna("")
    df["tagset"] = df.predicted_perceptions.apply(lenient_tag_set)

    ei = build_id_index(outputs_dir(model) / "caption_embeddings_ids.csv")
    df["ei"] = df.annotation_id.map(ei)
    df = df.dropna(subset=["ei"])
    df["ei"] = df.ei.astype(int)
    return df


def _find_examples(model: str, full_labels: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Worked examples behind specific cells, computed live from cached data.

    Returns a dict keyed by modality. Each value is a *list* of example dicts,
    ordered low -> high similarity:
      - 'caption'    : one example (captions barely vary, so the range is flat).
      - 'just'       : four examples whose heat-map cells span the similarity
                       range (low, slightly-low, slightly-high, high).
      - 'perception' : four examples, same banding on Jaccard.

    Each dict has matrix indices (i>=j, a lower-triangle cell), the two short
    profile labels, the two actual outputs for one shared image, the single-image
    example similarity (`sim`), and the heat-map cell's image-averaged value
    (`cellval`) so the arrow-target value matches the cell it points to.

    Args:
        model: Model directory id to draw examples from.
        full_labels: Full profile strings in matrix index order.

    Returns:
        Example lists keyed by modality.
    """
    df = _load_example_frame(model, full_labels)

    cap = np.load(outputs_dir(model) / "caption_embeddings.npy")
    jus = np.load(outputs_dir(model) / "justification_embeddings.npy")
    cell_mat = {
        "caption": np.load(outputs_dir(model) / "ic_profile_sim_caption.npy"),
        "just": np.load(outputs_dir(model) / "ic_profile_sim_just.npy"),
        "perception": np.load(outputs_dir(model) / "ic_profile_sim_jaccard.npy"),
    }

    def sim_cap(r1: pd.Series, r2: pd.Series) -> float:
        return _cos(cap[r1.ei], cap[r2.ei])

    def sim_just(r1: pd.Series, r2: pd.Series) -> float:
        return _cos(jus[r1.ei], jus[r2.ei])

    def sim_perc(r1: pd.Series, r2: pd.Series) -> float:
        tags_a, tags_b = r1.tagset, r2.tagset
        union = tags_a | tags_b
        return len(tags_a & tags_b) / len(union) if union else 0.0

    def txt_just(r: pd.Series) -> str:
        return str(r.just)

    def txt_perc(r: pd.Series) -> str:
        return "; ".join(sorted(r.tagset))

    by_profile = dict(df.groupby("midx"))

    return {
        "caption": _caption_example(df, cell_mat["caption"], sim_cap),
        "just": _band_examples(
            cell_mat["just"],
            by_profile,
            sim_fn=sim_just,
            txt_fn=txt_just,
            lo_len=40,
            hi_len=150,
            n=4,
        ),
        "perception": _band_examples(
            cell_mat["perception"],
            by_profile,
            sim_fn=sim_perc,
            txt_fn=txt_perc,
            lo_len=5,
            hi_len=90,
            n=4,
        ),
    }


def _wrap(text: str, width: int = 46, limit: int = 200) -> str:
    """Truncate and hard-wrap one output for display inside a callout box.

    Args:
        text: Text to render.
        width: Wrap column.
        limit: Maximum characters before an ellipsis is appended.

    Returns:
        The wrapped text.
    """
    t = text.strip()
    if len(t) > limit:
        t = t[:limit].rsplit(" ", 1)[0] + "…"
    return "\n".join(textwrap.wrap(t, width=width)) or t

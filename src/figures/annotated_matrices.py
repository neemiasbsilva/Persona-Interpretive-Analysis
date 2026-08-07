"""Figure C: annotated inter-profile similarity matrices per model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import PROJECT_ROOT
from ..models import outputs_dir
from .annotated_examples import _find_examples, _short_full_labels, _wrap


def _annotated_matrix(
    model: str,
    matrix_file: str,
    cbar_label: str,
    out_name: str,
    kind: str,
    *,
    example: dict[str, list[dict[str, Any]]] | None,
    out_dir: Path,
    vmin: float,
    vmax: float,
) -> None:
    """Redraw a cached 24x24 similarity matrix as an annotated heatmap.

    The matrix is drawn lower-triangular, with worked-example and structural
    callouts placed in the freed upper-right.

    ``vmin``/``vmax`` are the *shared* colour limits used by the plain matrices
    (generate_figures.py), so each annotated version matches its un-annotated
    counterpart (captions render uniformly dark, justifications span the scale).

    Profiles are sorted Gender -> Economic -> Political -> Personality, so block
    coordinates are fixed: Female rows 0-11, Male 12-23; High income first six of
    each gender, Low next six; Conservative then Progressive in threes. Cell
    (row r, col c) is centred at (c+0.5, r+0.5); y grows downward.

    Args:
        model: Model directory id the cached matrix belongs to.
        matrix_file: Cached ``.npy`` matrix filename.
        cbar_label: Colour-bar label.
        out_name: Output stem, written as both PDF and PNG.
        kind: One of ``caption``, ``just`` or ``perception``.
        example: Worked examples keyed by kind, or None when unavailable.
        out_dir: Directory to write the figures into.
        vmin: Shared colour-scale minimum.
        vmax: Shared colour-scale maximum.
    """
    mat = np.load(outputs_dir(model) / matrix_file)
    short, _ = _short_full_labels(model)
    mat_df = pd.DataFrame(mat, index=short, columns=short)
    upper = np.triu(np.ones_like(mat, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(15, 11.5))
    sns.heatmap(
        mat_df,
        ax=ax,
        cmap="Reds",
        mask=upper,
        annot=False,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.3,
        linecolor="white",
        square=True,
        cbar_kws={"label": cbar_label, "shrink": 0.45, "anchor": (0.0, 0.0)},
    )
    ax.tick_params(axis="x", rotation=90, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

    def box(
        text: str,
        xy: tuple[float, float],
        xytext: tuple[float, float],
        *,
        fc: str = "#FFF8E1",
        fontsize: float = 11.5,
        rad: float = -0.2,
        weight: str | None = None,
    ) -> None:
        ax.annotate(
            text,
            xy=xy,
            xycoords="data",
            xytext=xytext,
            textcoords="data",
            annotation_clip=False,
            fontsize=fontsize,
            va="top",
            ha="left",
            fontweight=weight,
            bbox={"boxstyle": "round,pad=0.45", "fc": fc, "ec": "0.35"},
            arrowprops={
                "arrowstyle": "->",
                "color": "0.2",
                "lw": 1.6,
                "connectionstyle": f"arc3,rad={rad}",
            },
        )

    metric = "Jaccard" if kind == "perception" else "sim"
    quote = "" if kind == "perception" else "“"
    endq = "" if kind == "perception" else "”"
    exs = (example.get(kind) if example else None) or []

    def _example_body(ex: dict[str, Any], wrap: int = 44, limit: int = 190) -> str:
        head = (
            f"cell (avg) = {ex['cellval']:.2f}  ⟵ arrow  |  one image: {metric} = {ex['sim']:.2f}"
        )
        return (
            f"{head}\n"
            f"[{ex['pa']}]  {quote}{_wrap(ex['ta'], wrap, limit)}{endq}\n"
            f"[{ex['pb']}]  {quote}{_wrap(ex['tb'], wrap, limit)}{endq}"
        )

    if kind == "caption":
        box(
            "Narrow range, no block structure:\n"
            "captions barely move with persona\n(descriptive grounding is invariant)",
            xy=(3.0, 12.0),
            xytext=(12.3, 12.6),
            rad=0.22,
        )
        if exs:
            box(
                _example_body(exs[0]),
                xy=(exs[0]["j"] + 0.5, exs[0]["i"] + 0.5),
                xytext=(10.5, 0.4),
                fc="#E8F8F5",
                fontsize=10.5,
                rad=-0.28,
            )
    else:
        anchors = [(12.6, 0.3), (12.6, 4.5), (12.6, 8.7), (12.6, 12.9)]
        rads = [-0.30, -0.18, 0.18, 0.30]
        for k, ex in enumerate(exs):
            box(
                _example_body(ex, wrap=40, limit=110),
                xy=(ex["j"] + 0.5, ex["i"] + 0.5),
                xytext=anchors[k % len(anchors)],
                fc="#E8F8F5",
                fontsize=9.2,
                rad=rads[k % len(rads)],
            )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{out_name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name} (annotated, {model})")


def figure_annotated_matrices(model: str) -> None:
    """Write annotated backup versions of the plain similarity matrices.

    The lower-triangular, worked-example figures are written next to the plain
    matrices, which are left untouched.

    Args:
        model: Model directory id to draw the matrices for.
    """
    out_dir = PROJECT_ROOT / "figures" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    _, full = _short_full_labels(model)
    example = _find_examples(model, full)

    mats = [
        np.load(outputs_dir(model) / f)
        for f in (
            "ic_profile_sim_caption.npy",
            "ic_profile_sim_just.npy",
            "ic_profile_sim_jaccard.npy",
        )
    ]
    all_vals = np.concatenate([m.ravel() for m in mats])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    _annotated_matrix(
        model,
        "ic_profile_sim_caption.npy",
        "Mean cosine similarity",
        "fig_cosine_cross_profile_heatmap_caption_annotated",
        "caption",
        example=example,
        out_dir=out_dir,
        vmin=vmin,
        vmax=vmax,
    )
    _annotated_matrix(
        model,
        "ic_profile_sim_just.npy",
        "Mean cosine similarity",
        "fig_cosine_cross_profile_heatmap_just_annotated",
        "just",
        example=example,
        out_dir=out_dir,
        vmin=vmin,
        vmax=vmax,
    )
    _annotated_matrix(
        model,
        "ic_profile_sim_jaccard.npy",
        "Mean Jaccard similarity",
        "fig_cross_profile_perception_jaccard_annotated",
        "perception",
        example=example,
        out_dir=out_dir,
        vmin=vmin,
        vmax=vmax,
    )

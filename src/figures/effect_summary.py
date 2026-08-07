"""Figure A (persona-effect dot plot) and Figure B (topic contrasts)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..data_loading import load_annotations, parse_demographics
from ..models import annotations_path, display_name, outputs_dir
from ..similarity import EPS, run_within_cross_significance
from .style import DIM_LABELS, DIMS, MOD_COLORS, MOD_LABELS, MODALITIES, TOPIC_CONTRASTS


def _effect_table(model: str) -> pd.DataFrame:
    """Per (dimension, modality): within-minus-cross Delta, bootstrap CI, verdict.

    Thin adapter over similarity.run_within_cross_significance, which owns the
    statistics quoted in Section 4.2 (paired Wilcoxon signed-rank on the
    per-image gaps, plus the bootstrap CI and the practical-equivalence
    verdict). Keeping a single implementation means this figure cannot drift
    from the reported numbers.
    """
    d = pd.read_csv(outputs_dir(model) / "within_cross_persona_sim.csv")
    stats = run_within_cross_significance(d, demo_cols=DIMS, modalities=MODALITIES)
    return stats.rename(columns={"ci_lo": "lo", "ci_hi": "hi", "p_wilcoxon": "p"})


def figure_persona_effect_summary(model_a: str, model_b: str, out_dir: Path) -> None:
    """Plot Figure A: within-minus-cross effect by dimension and modality.

    Every cell is significant under the paired test, so significance stars would
    carry no information. The shaded equivalence band replaces them: a confidence
    interval contained in the band is a persona effect too small to matter.

    Args:
        model_a: Model shown in the left panel.
        model_b: Model shown in the right panel.
        out_dir: Directory to write the PDF and PNG into.
    """
    tables = {m: _effect_table(m) for m in (model_a, model_b)}
    xmax = max(t.hi.max() for t in tables.values())
    xmin = min(t.lo.min() for t in tables.values())
    pad = 0.05 * (xmax - xmin)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    band = 1.0
    offsets = {
        "caption": +band * 0.28,
        "perception": 0.0,
        "justification": -band * 0.28,
    }
    y_base = {dim: len(DIMS) - 1 - i for i, dim in enumerate(DIMS)}

    for ax, model in zip(axes, (model_a, model_b), strict=True):
        tbl = tables[model]
        ax.axvspan(-EPS, EPS, color="0.85", zorder=0, label=rf"Negligible ($|\Delta| < {EPS}$)")
        ax.axvline(0.0, color="0.6", lw=1.0, zorder=1)
        for mod in MODALITIES:
            sub = tbl[tbl.modality == mod]
            ys = [y_base[dim] + offsets[mod] for dim in sub.dimension]
            ax.errorbar(
                sub.delta,
                ys,
                xerr=[sub.delta - sub.lo, sub.hi - sub.delta],
                fmt="o",
                ms=8,
                color=MOD_COLORS[mod],
                ecolor=MOD_COLORS[mod],
                elinewidth=1.6,
                capsize=3,
                label=MOD_LABELS[mod],
                zorder=3,
            )
        ax.set_yticks([y_base[d] for d in DIMS])
        ax.set_yticklabels([DIM_LABELS[d] for d in DIMS])
        ax.set_xlim(min(xmin - pad, -pad), xmax + 2 * pad)
        ax.set_xlabel(r"Within$-$cross similarity $\Delta$")
        ax.set_title(display_name(model))
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06)
    )
    fig.suptitle(
        "Persona effect by dimension and modality (higher = stronger persona signal)",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_persona_effect_summary.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_persona_effect_summary")


def _topic_demo(model: str) -> pd.DataFrame:
    """annotations_with_topics joined to parsed persona demographics."""
    ann = pd.read_csv(outputs_dir(model) / "annotations_with_topics.csv")
    demo = parse_demographics(load_annotations(annotations_path(model)))
    cols = ["annotation_id", "economic_status", "political_spectrum"]
    return ann.merge(demo[cols].drop_duplicates("annotation_id"), on="annotation_id", how="inner")


def _labeled_coverage(model: str, labels: dict[int, str]) -> float:
    """Share of all justifications that fall in a nicknamed (top) topic."""
    ann = pd.read_csv(outputs_dir(model) / "annotations_with_topics.csv")
    return (ann.just_topic.isin(labels)).mean()


def figure_topic_contrasts(model_a: str, model_b: str, out_dir: Path) -> None:
    """Plot Figure B: topic-prevalence contrasts as diverging bars.

    Args:
        model_a: Model shown in the left column.
        model_b: Model shown in the right column.
        out_dir: Directory to write the PDF and PNG into.
    """
    fig, axes = plt.subplots(
        len(TOPIC_CONTRASTS),
        2,
        figsize=(15, 11),
        gridspec_kw={"wspace": 0.55, "hspace": 0.35},
    )
    for col, model in enumerate((model_a, model_b)):
        labels = (
            pd.read_csv(outputs_dir(model) / "topic_labels_just.csv")
            .set_index("topic")["label"]
            .to_dict()
        )
        d = _topic_demo(model)
        d = d[d.just_topic.isin(labels)]
        coverage = _labeled_coverage(model, labels)
        for row, (dim, pos, neg) in enumerate(TOPIC_CONTRASTS):
            ax = axes[row, col]
            pa = d[d[dim] == pos].just_topic.value_counts(normalize=True)
            pb = d[d[dim] == neg].just_topic.value_counts(normalize=True)
            diff = (pa.reindex(labels).fillna(0) - pb.reindex(labels).fillna(0)) * 100
            diff = diff.sort_values()
            names = [labels[t] for t in diff.index]
            colors = ["#CC6677" if v > 0 else "#4477AA" for v in diff.to_numpy()]
            ax.barh(range(len(diff)), diff.to_numpy(), color=colors)
            ax.axvline(0.0, color="0.5", lw=0.9)
            ax.set_yticks(range(len(diff)))
            ax.set_yticklabels(names, fontsize=10)
            ax.set_xlabel(f"{pos} $-$ {neg}  (percentage points)", fontsize=10)
            if row == 0:
                ax.set_title(
                    f"{display_name(model)}\n(top-topic coverage {coverage:.0%})", fontsize=12
                )
            ax.grid(axis="x", alpha=0.3)
            ax.set_axisbelow(True)
    fig.suptitle(
        "Topic-prevalence contrasts along economic status and political orientation",
        y=0.98,
        fontsize=13,
    )
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_topic_contrasts.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_topic_contrasts")

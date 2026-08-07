"""Standalone figure generation for the analyzing-persona-effects-mllm paper.

Uses only cached data (embeddings, CSVs, JSONL) — no BERTopic refitting needed.
The active model is selected by config.MODEL (env PERSONA_MODEL); all inputs and
outputs are namespaced under that model.
Run from project root: uv run python -m src.generate_figures
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .config import FIGURES, FONT_SCALE, OUTPUTS, SENT_COLORS_3
from .data_loading import create_profiles, load_annotations, parse_demographics
from .embeddings import build_id_index
from .metrics import jaccard, lenient_tag_set
from .similarity import compute_image_conditioned_profile_sim
from .topic_divergence import compute_topic_jsd
from .topic_modeling import load_topic_labels

HEATMAP_CMAP = "Reds"
HEATMAP_TICK_FONT = 16
HEATMAP_CBAR_FONT = 18
HEATMAP_CBAR_TICK_FONT = 15


def _abbrev(p: str) -> str:
    pts = [x.strip() for x in p.split("/")]
    return f"{pts[0][0]}/{pts[1][:4]}/{pts[2][:4]}/{pts[3][:4]}"


def _topic_persona_proportions(
    df: pd.DataFrame,
    ann_topics: pd.DataFrame,
    topic_labels: dict[int, str],
) -> pd.DataFrame:
    """Column-normalized topic proportions per abbreviated persona profile.

    Args:
        df: Annotation frame carrying ``profile_abbr``.
        ann_topics: Per-annotation topic assignments.
        topic_labels: Curated nickname per labelled topic.

    Returns:
        Topics by personas, each column summing to 1.
    """
    merged_tp = ann_topics.merge(
        df[["annotation_id", "profile_abbr"]], on="annotation_id", how="left"
    ).dropna(subset=["profile_abbr"])
    tp = (
        merged_tp[merged_tp["just_topic"].isin(list(topic_labels))]
        .groupby(["just_topic", "profile_abbr"])
        .size()
        .reset_index(name="count")
    )
    piv = tp.pivot_table(
        index="just_topic", columns="profile_abbr", values="count", aggfunc="sum"
    ).fillna(0)
    piv_n = piv.div(piv.sum(axis=0), axis=1)
    piv_n.index = [topic_labels[i] for i in piv_n.index]
    return piv_n


def _figure_topic_persona_heatmap(piv_n: pd.DataFrame) -> None:
    """Draw the topic-by-persona proportion heatmap.

    Args:
        piv_n: Column-normalized topic proportions.
    """
    fig, ax = plt.subplots(figsize=(18, 7))
    sns.heatmap(
        piv_n,
        ax=ax,
        cmap=HEATMAP_CMAP,
        vmin=piv_n.to_numpy().min(),
        vmax=piv_n.to_numpy().max(),
        annot=False,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Topic proportion", "shrink": 0.6},
    )
    ax.set_xlabel("Persona", fontsize=20)
    ax.set_ylabel("Topic", fontsize=20)
    ax.tick_params(axis="x", rotation=90, labelsize=15, pad=1)
    ax.tick_params(axis="y", rotation=0, labelsize=15, pad=1)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=HEATMAP_CBAR_TICK_FONT)
    cbar.set_label("Topic proportion", fontsize=HEATMAP_CBAR_FONT)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_topic_persona_heatmap_just.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_topic_persona_heatmap_just.png", bbox_inches="tight")
    plt.close()


def _write_topic_jsd(df: pd.DataFrame, ann_topics: pd.DataFrame) -> None:
    """Compute and persist per-dimension topic divergence.

    Args:
        df: Annotation frame carrying the demographic columns.
        ann_topics: Per-annotation topic assignments.
    """
    jsd_df = compute_topic_jsd(df, ann_topics, topics=None, n_perm=1000, seed=42)
    jsd_df.to_csv(OUTPUTS / "topic_jsd_by_dimension.csv", index=False)
    for _, r in jsd_df.iterrows():
        print(f"  {r['dimension']:<20s} JSD={r['jsd']:.4f}  p_perm={r['p_perm']:.4f}")
    print("  Saved topic_jsd_by_dimension.csv")


def _figure_topic_sentiment(ann_topics: pd.DataFrame, topic_labels: dict[int, str]) -> None:
    """Draw the stacked sentiment composition per justification topic.

    Args:
        ann_topics: Per-annotation topic assignments with predicted sentiment.
        topic_labels: Curated nickname per labelled topic.
    """
    order3 = ["Positive", "Neutral", "Negative"]
    sent_tp = ann_topics[
        ann_topics["just_topic"].isin(list(topic_labels))
        & ann_topics["predicted_sentiment"].isin(order3)
    ]
    pivot = (
        sent_tp.groupby(["just_topic", "predicted_sentiment"])
        .size()
        .reset_index(name="count")
        .pivot_table(
            index="just_topic", columns="predicted_sentiment", values="count", aggfunc="sum"
        )
        .fillna(0)
    )
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)[order3]
    pivot_norm.index = [topic_labels[i] for i in pivot_norm.index]
    pivot_norm = pivot_norm.sort_values("Negative", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    left = np.zeros(len(pivot_norm))
    for col, color in zip(order3, [SENT_COLORS_3[c] for c in order3], strict=True):
        vals = pivot_norm[col].to_numpy()
        ax.barh(
            range(len(pivot_norm)),
            vals,
            left=left,
            color=color,
            label=col,
            height=0.62,
            edgecolor="white",
            linewidth=0.5,
        )
        for i, (val, base) in enumerate(zip(vals, left, strict=True)):
            if val > 0.09:
                ax.text(
                    base + val / 2,
                    i,
                    f"{val:.0%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
        left += vals
    ax.set_yticks(range(len(pivot_norm)))
    ax.set_yticklabels(pivot_norm.index, fontsize=11)
    ax.set_xlabel("Proportion", fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.tick_params(axis="x", labelsize=10)
    ax.xaxis.grid(visible=True, linestyle="--", alpha=0.35, color="gray", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        title="Sentiment",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        fontsize=10,
        title_fontsize=10,
        framealpha=0.9,
    )
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_topic_just_sentiment_composition.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_topic_just_sentiment_composition.png", bbox_inches="tight")
    plt.close()


def _parse_profile(profile: str) -> tuple[str, str]:
    """Reduce a full profile string to its income and political levels.

    Args:
        profile: Full ``gender / income / spectrum / personality`` string.

    Returns:
        The income level and the political level.
    """
    parts = [x.strip() for x in profile.split("/")]
    econ = "High" if any("High" in x or "high" in x for x in parts) else "Low"
    polit = "Con" if any("Con" in x or "con" in x.lower() for x in parts) else "Pro"
    return econ, polit


def _figure_persona_topic_tsne(piv_n: pd.DataFrame) -> None:
    """Draw the t-SNE projection of personas in topic-proportion space.

    Args:
        piv_n: Column-normalized topic proportions.
    """
    piv_tsne = piv_n.T
    profiles_list = piv_tsne.index.tolist()
    scaled = StandardScaler().fit_transform(piv_tsne.to_numpy())
    emb = TSNE(n_components=2, perplexity=5, random_state=42, max_iter=2000).fit_transform(scaled)

    attrs = [_parse_profile(p) for p in profiles_list]
    econ_colors = {"High": "#2166ac", "Low": "#d73027"}
    polit_shapes = {"Con": "^", "Pro": "o"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (econ, polit) in enumerate(attrs):
        ax.scatter(
            emb[i, 0],
            emb[i, 1],
            c=econ_colors[econ],
            marker=polit_shapes[polit],
            s=120,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
    legend_handles = [
        mpatches.Patch(color=econ_colors["High"], label="High income"),
        mpatches.Patch(color=econ_colors["Low"], label="Low income"),
        plt.Line2D(
            [0], [0], marker="^", color="gray", markersize=9, linestyle="", label="Conservative"
        ),
        plt.Line2D(
            [0], [0], marker="o", color="gray", markersize=9, linestyle="", label="Progressive"
        ),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="best")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(visible=True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_persona_topic_tsne.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_persona_topic_tsne.png", bbox_inches="tight")
    plt.close()


def _perception_jaccard_matrix(df: pd.DataFrame, profiles: list[str]) -> np.ndarray:
    """Image-conditioned mean Jaccard similarity between every profile pair.

    Args:
        df: Annotation frame carrying ``profile`` and ``predicted_perceptions``.
        profiles: Profiles in matrix row and column order.

    Returns:
        Square matrix of mean per-image Jaccard similarity.
    """
    by_profile = {
        profile: df[df["profile"] == profile]
        .groupby("image_id")["predicted_perceptions"]
        .apply(list)
        for profile in profiles
    }
    n = len(profiles)
    jac_mat = np.zeros((n, n))
    for i, pi in enumerate(profiles):
        gi = by_profile[pi]
        for j, pj in enumerate(profiles):
            vals = []
            if i == j:
                for img in gi.index:
                    si = [lenient_tag_set(x) for x in gi[img]]
                    if len(si) < 2:
                        continue
                    pairs = [
                        jaccard(a, b, on_empty_union=0.0)
                        for idx_a, a in enumerate(si)
                        for idx_b, b in enumerate(si)
                        if idx_a < idx_b
                    ]
                    if pairs:
                        vals.append(float(np.mean(pairs)))
            else:
                gj = by_profile[pj]
                for img in gi.index.intersection(gj.index):
                    si = [lenient_tag_set(x) for x in gi[img]]
                    sj = [lenient_tag_set(x) for x in gj[img]]
                    pairs = [jaccard(a, b, on_empty_union=0.0) for a in si for b in sj]
                    if pairs:
                        vals.append(float(np.mean(pairs)))
            jac_mat[i, j] = float(np.mean(vals)) if vals else 0.0
    return jac_mat


def _figure_profile_matrix(
    mat: np.ndarray,
    short_labs: list[str],
    fname: str,
    cbar_label: str,
    *,
    vmin: float,
    vmax: float,
) -> None:
    """Draw one cross-profile similarity heatmap on the shared colour scale.

    Args:
        mat: Square similarity matrix.
        short_labs: Abbreviated profile labels for both axes.
        fname: Output stem, written as both PDF and PNG.
        cbar_label: Colour-bar label.
        vmin: Shared colour-scale minimum.
        vmax: Shared colour-scale maximum.
    """
    mat_df = pd.DataFrame(mat, index=short_labs, columns=short_labs)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        mat_df,
        ax=ax,
        cmap=HEATMAP_CMAP,
        vmin=vmin,
        vmax=vmax,
        annot=False,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": cbar_label, "shrink": 0.7},
    )
    ax.tick_params(axis="x", rotation=90, labelsize=HEATMAP_TICK_FONT)
    ax.tick_params(axis="y", rotation=0, labelsize=HEATMAP_TICK_FONT)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=HEATMAP_CBAR_TICK_FONT)
    cbar.set_label(cbar_label, fontsize=HEATMAP_CBAR_FONT)
    plt.tight_layout()
    fig.savefig(FIGURES / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{fname}.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


def main() -> None:
    """Regenerate every cached-data figure for the active model."""
    sns.set_theme(style="whitegrid", font_scale=FONT_SCALE)
    FIGURES.mkdir(exist_ok=True)

    print("Loading annotations...")
    df = load_annotations()
    df = parse_demographics(df)
    df = create_profiles(df)
    print(f"  {len(df):,} records, {df['profile'].nunique()} profiles")

    print("Loading embeddings...")
    cap_embs = np.load(OUTPUTS / "caption_embeddings.npy")
    just_embs = np.load(OUTPUTS / "justification_embeddings.npy")
    cap_idx = build_id_index(OUTPUTS / "caption_embeddings_ids.csv")

    topic_labels = load_topic_labels(OUTPUTS / "topic_labels_just.csv")
    ann_topics = pd.read_csv(OUTPUTS / "annotations_with_topics.csv")

    print("\n[1] fig_topic_persona_heatmap_just ...")
    piv_n = _topic_persona_proportions(df, ann_topics, topic_labels)
    _figure_topic_persona_heatmap(piv_n)
    print("  Saved.")

    print("\n[1b] topic_jsd_by_dimension ...")
    _write_topic_jsd(df, ann_topics)

    print("\n[2] fig_topic_just_sentiment_composition ...")
    _figure_topic_sentiment(ann_topics, topic_labels)
    print("  Saved.")

    print("\n[3] fig_persona_topic_tsne ...")
    _figure_persona_topic_tsne(piv_n)
    print("  Saved.")

    print("\n[2-3] fig_cosine_cross_profile_heatmap ...")
    ic_cap, ic_just, ic_labs = compute_image_conditioned_profile_sim(
        df,
        cap_embs,
        just_embs,
        cap_idx,
        cap_cache=OUTPUTS / "ic_profile_sim_caption.npy",
        just_cache=OUTPUTS / "ic_profile_sim_just.npy",
        labs_cache=OUTPUTS / "ic_profile_sim_labels.csv",
    )
    print(
        f"  Matrix: {ic_cap.shape}, "
        f"cap [{ic_cap.min():.3f}, {ic_cap.max():.3f}], "
        f"just [{ic_just.min():.3f}, {ic_just.max():.3f}]"
    )

    short_labs = [_abbrev(p) for p in ic_labs]

    print("\n[4] fig_cross_profile_perception_jaccard (computing) ...")
    jac_cache = OUTPUTS / "ic_profile_sim_jaccard.npy"
    if jac_cache.exists():
        jac_mat = np.load(jac_cache)
        print("  Loaded from cache.")
    else:
        jac_mat = _perception_jaccard_matrix(df, list(ic_labs))
        np.save(jac_cache, jac_mat)
        print(f"  Computed and cached. Range [{jac_mat.min():.3f}, {jac_mat.max():.3f}]")

    all_vals = np.concatenate([ic_cap.ravel(), ic_just.ravel(), jac_mat.ravel()])
    shared_vmin, shared_vmax = float(all_vals.min()), float(all_vals.max())
    print(f"  Shared scale: [{shared_vmin:.3f}, {shared_vmax:.3f}]")

    for mat, fname in [
        (ic_cap, "fig_cosine_cross_profile_heatmap_caption"),
        (ic_just, "fig_cosine_cross_profile_heatmap_just"),
    ]:
        _figure_profile_matrix(
            mat,
            short_labs,
            fname,
            "Mean cosine similarity",
            vmin=shared_vmin,
            vmax=shared_vmax,
        )
    _figure_profile_matrix(
        jac_mat,
        short_labs,
        "fig_cross_profile_perception_jaccard",
        "Mean Jaccard similarity",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )

    print("\nAll figures generated successfully.")
    print("Files in figures/:")
    for f in sorted(FIGURES.glob("fig_*.pdf")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()

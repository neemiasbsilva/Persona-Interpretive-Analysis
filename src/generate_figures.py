"""Standalone figure generation for percepta_urbana paper.

Uses only cached data (embeddings, CSVs, JSONL) — no BERTopic refitting needed.
Run from project root: uv run python -m src.generate_figures
"""
import ast as _ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

HEATMAP_CMAP = "Reds"
HEATMAP_ANNOT_FONT = 9.5
HEATMAP_TICK_FONT = 11
HEATMAP_CBAR_FONT = 12
HEATMAP_CBAR_TICK_FONT = 11

_TOPIC_LABELS = {
    0: "Natural landscape & beauty",
    1: "Work & development",
    2: "Accident & police response",
    3: "Rural & quiet road",
    4: "Historical & preserved",
    5: "Destruction & conflict",
    6: "Streets & orderly urban",
    7: "City & nature",
    8: "Home & peaceful scene",
    9: "Typical urban scene",
}

from .config import FIGURES, OUTPUTS, FONT_SCALE, SENT_COLORS_3
from .data_loading import load_annotations, parse_demographics, create_profiles
from .embeddings import build_id_index
from .similarity import compute_image_conditioned_profile_sim


def _abbrev(p: str) -> str:
    pts = [x.strip() for x in p.split("/")]
    return f"{pts[0][0]}/{pts[1][:4]}/{pts[2][:4]}/{pts[3][:4]}"


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=FONT_SCALE)
    FIGURES.mkdir(exist_ok=True)

    print("Loading annotations...")
    df = load_annotations()
    df = parse_demographics(df)
    df = create_profiles(df)
    print(f"  {len(df):,} records, {df['profile'].nunique()} profiles")

    print("Loading embeddings...")
    cap_embs  = np.load(OUTPUTS / "caption_embeddings.npy")
    just_embs = np.load(OUTPUTS / "justification_embeddings.npy")
    cap_idx   = build_id_index(OUTPUTS / "caption_embeddings_ids.csv")

    # ── Fig 1: topic × persona profile heatmap (justifications) ─────────────
    print("\n[1] fig_topic_persona_heatmap_just ...")
    ann_topics = pd.read_csv(OUTPUTS / "annotations_with_topics.csv")
    merged_tp  = ann_topics.merge(
        df[["annotation_id", "profile_abbr"]], on="annotation_id", how="left"
    ).dropna(subset=["profile_abbr"])

    labeled_topics = list(_TOPIC_LABELS.keys())
    tp    = (merged_tp[merged_tp["just_topic"].isin(labeled_topics)]
             .groupby(["just_topic", "profile_abbr"]).size().reset_index(name="count"))
    piv   = tp.pivot(index="just_topic", columns="profile_abbr", values="count").fillna(0)
    piv_n = piv.div(piv.sum(axis=0), axis=1)
    piv_n.index = [_TOPIC_LABELS[i] for i in piv_n.index]

    fig, ax = plt.subplots(figsize=(18, 7))
    sns.heatmap(piv_n, ax=ax, cmap=HEATMAP_CMAP,
                vmin=piv_n.values.min(), vmax=piv_n.values.max(),
                annot=True, fmt=".2f", annot_kws={"fontsize": 5.5},
                linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Topic proportion", "shrink": 0.6})
    ax.set_xlabel("Persona", fontsize=14)
    ax.set_ylabel("Topic", fontsize=14)
    ax.tick_params(axis="x", rotation=90, labelsize=11, pad=1)
    ax.tick_params(axis="y", rotation=0,  labelsize=11, pad=1)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_topic_persona_heatmap_just.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_topic_persona_heatmap_just.png", bbox_inches="tight")
    plt.close()
    print("  Saved.")

    # ── Fig 2: justification topic × sentiment composition ──────────────────
    print("\n[2] fig_topic_just_sentiment_composition ...")
    order3 = ["Positive", "Neutral", "Negative"]
    sent_tp = ann_topics[ann_topics["just_topic"].isin(labeled_topics)
                         & ann_topics["predicted_sentiment"].isin(order3)]
    pivot = (sent_tp.groupby(["just_topic", "predicted_sentiment"]).size()
             .reset_index(name="count")
             .pivot(index="just_topic", columns="predicted_sentiment", values="count")
             .fillna(0))
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)[order3]
    pivot_norm.index = [_TOPIC_LABELS[i] for i in pivot_norm.index]
    pivot_norm = pivot_norm.sort_values("Negative", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    left = np.zeros(len(pivot_norm))
    for col, color in zip(order3, [SENT_COLORS_3[c] for c in order3]):
        vals = pivot_norm[col].values
        ax.barh(range(len(pivot_norm)), vals, left=left,
                color=color, label=col, height=0.62,
                edgecolor="white", linewidth=0.5)
        for i, (val, l) in enumerate(zip(vals, left)):
            if val > 0.09:
                ax.text(l + val / 2, i, f"{val:.0%}",
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
        left += vals
    ax.set_yticks(range(len(pivot_norm)))
    ax.set_yticklabels(pivot_norm.index, fontsize=11)
    ax.set_xlabel("Proportion", fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.tick_params(axis="x", labelsize=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, color="gray", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 0.5), loc="center left",
              fontsize=10, title_fontsize=10, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_topic_just_sentiment_composition.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_topic_just_sentiment_composition.png", bbox_inches="tight")
    plt.close()
    print("  Saved.")

    # ── Fig 3: t-SNE of persona profiles in topic-distribution space ─────────
    print("\n[3] fig_persona_topic_tsne ...")
    piv_tsne = piv_n.T  # (n_profiles × n_topics)
    profiles_list = piv_tsne.index.tolist()
    X = StandardScaler().fit_transform(piv_tsne.values)
    emb = TSNE(n_components=2, perplexity=5, random_state=42, max_iter=2000).fit_transform(X)

    def _parse_profile(p):
        parts = [x.strip() for x in p.split("/")]
        econ  = "High" if any("High" in x or "high" in x for x in parts) else "Low"
        polit = "Con"  if any("Con"  in x or "con"  in x.lower() for x in parts) else "Pro"
        return econ, polit

    attrs      = [_parse_profile(p) for p in profiles_list]
    econ_vals  = [a[0] for a in attrs]
    polit_vals = [a[1] for a in attrs]
    econ_colors  = {"High": "#2166ac", "Low": "#d73027"}
    polit_shapes = {"Con": "^", "Pro": "o"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (prof, econ, polit) in enumerate(zip(profiles_list, econ_vals, polit_vals)):
        ax.scatter(emb[i, 0], emb[i, 1],
                   c=econ_colors[econ], marker=polit_shapes[polit],
                   s=120, edgecolors="white", linewidths=0.6, zorder=3)
    legend_handles = [
        mpatches.Patch(color=econ_colors["High"], label="High income"),
        mpatches.Patch(color=econ_colors["Low"],  label="Low income"),
        plt.Line2D([0], [0], marker="^", color="gray", markersize=9, linestyle="", label="Conservative"),
        plt.Line2D([0], [0], marker="o", color="gray", markersize=9, linestyle="", label="Progressive"),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="best")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_persona_topic_tsne.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_persona_topic_tsne.png", bbox_inches="tight")
    plt.close()
    print("  Saved.")

    # ── Figs 4-5: image-conditioned cross-profile cosine similarity ──────────
    print("\n[2-3] fig_cosine_cross_profile_heatmap ...")
    ic_cap, ic_just, ic_labs = compute_image_conditioned_profile_sim(
        df, cap_embs, just_embs, cap_idx,
        cap_cache=OUTPUTS  / "ic_profile_sim_caption.npy",
        just_cache=OUTPUTS / "ic_profile_sim_just.npy",
        labs_cache=OUTPUTS / "ic_profile_sim_labels.csv",
    )
    print(f"  Matrix: {ic_cap.shape}, "
          f"cap [{ic_cap.min():.3f}, {ic_cap.max():.3f}], "
          f"just [{ic_just.min():.3f}, {ic_just.max():.3f}]")

    short_labs = [_abbrev(p) for p in ic_labs]

    # ── Compute Jaccard first so all three matrices are available for shared scale
    print("\n[4] fig_cross_profile_perception_jaccard (computing) ...")
    JAC_CACHE = OUTPUTS / "ic_profile_sim_jaccard.npy"
    if JAC_CACHE.exists():
        jac_mat = np.load(JAC_CACHE)
        print("  Loaded from cache.")
    else:
        def _tag_set(v):
            if isinstance(v, list):
                return set(v)
            try:
                return set(_ast.literal_eval(v))
            except Exception:
                return set()

        def _jaccard(a, b):
            u = len(a | b)
            return len(a & b) / u if u > 0 else 0.0

        profiles = ic_labs
        n = len(profiles)
        jac_mat = np.zeros((n, n))
        for i, pi in enumerate(profiles):
            for j, pj in enumerate(profiles):
                gi = df[df["profile"] == pi].groupby("image_id")["predicted_perceptions"].apply(list)
                if i == j:
                    vals = []
                    for img in gi.index:
                        si = [_tag_set(x) for x in gi[img]]
                        if len(si) >= 2:
                            pairs = [_jaccard(a, b) for idx_a, a in enumerate(si)
                                     for idx_b, b in enumerate(si) if idx_a < idx_b]
                            if pairs:
                                vals.append(float(np.mean(pairs)))
                    jac_mat[i, j] = float(np.mean(vals)) if vals else 0.0
                    continue
                gj = df[df["profile"] == pj].groupby("image_id")["predicted_perceptions"].apply(list)
                shared = gi.index.intersection(gj.index)
                vals = []
                for img in shared:
                    si = [_tag_set(x) for x in gi[img]]
                    sj = [_tag_set(x) for x in gj[img]]
                    pairs = [_jaccard(a, b) for a in si for b in sj]
                    if pairs:
                        vals.append(float(np.mean(pairs)))
                jac_mat[i, j] = float(np.mean(vals)) if vals else 0.0
        np.save(JAC_CACHE, jac_mat)
        print(f"  Computed and cached. Range [{jac_mat.min():.3f}, {jac_mat.max():.3f}]")

    # Shared scale across all three cross-profile matrices
    all_vals = np.concatenate([ic_cap.ravel(), ic_just.ravel(), jac_mat.ravel()])
    shared_vmin, shared_vmax = float(all_vals.min()), float(all_vals.max())
    print(f"  Shared scale: [{shared_vmin:.3f}, {shared_vmax:.3f}]")

    # ── Figs 2-3: cosine cross-profile heatmaps ──────────────────────────────
    for mat, fname, cbar_label in [
        (ic_cap,  "fig_cosine_cross_profile_heatmap_caption", "Mean cosine similarity"),
        (ic_just, "fig_cosine_cross_profile_heatmap_just",    "Mean cosine similarity"),
    ]:
        mat_df = pd.DataFrame(mat, index=short_labs, columns=short_labs)
        fig, ax = plt.subplots(figsize=(13, 10))
        sns.heatmap(
            mat_df, ax=ax, cmap=HEATMAP_CMAP,
            vmin=shared_vmin, vmax=shared_vmax,
            annot=True, fmt=".2f", annot_kws={"fontsize": HEATMAP_ANNOT_FONT},
            linewidths=0.3, linecolor="white",
            cbar_kws={"label": cbar_label, "shrink": 0.7},
        )
        ax.tick_params(axis="x", rotation=90, labelsize=HEATMAP_TICK_FONT)
        ax.tick_params(axis="y", rotation=0,  labelsize=HEATMAP_TICK_FONT)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=HEATMAP_CBAR_TICK_FONT)
        cbar.set_label(cbar_label, fontsize=HEATMAP_CBAR_FONT)
        plt.tight_layout()
        fig.savefig(FIGURES / f"{fname}.pdf", bbox_inches="tight")
        fig.savefig(FIGURES / f"{fname}.png", bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")

    # ── Fig 4: Jaccard cross-profile heatmap ─────────────────────────────────
    jac_df = pd.DataFrame(jac_mat, index=short_labs, columns=short_labs)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        jac_df, ax=ax, cmap=HEATMAP_CMAP,
        vmin=shared_vmin, vmax=shared_vmax,
        annot=True, fmt=".2f", annot_kws={"fontsize": HEATMAP_ANNOT_FONT},
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Mean Jaccard similarity", "shrink": 0.7},
    )
    ax.tick_params(axis="x", rotation=90, labelsize=HEATMAP_TICK_FONT)
    ax.tick_params(axis="y", rotation=0,  labelsize=HEATMAP_TICK_FONT)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=HEATMAP_CBAR_TICK_FONT)
    cbar.set_label("Mean Jaccard similarity", fontsize=HEATMAP_CBAR_FONT)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_cross_profile_perception_jaccard.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_cross_profile_perception_jaccard.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig_cross_profile_perception_jaccard")

    print("\nAll figures generated successfully.")
    print("Files in figures/:")
    for f in sorted(FIGURES.glob("fig_*.pdf")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()

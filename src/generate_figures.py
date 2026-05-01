"""Standalone figure generation for percepta_urbana paper.

Uses only cached data (embeddings, CSVs, JSONL) — no BERTopic refitting needed.
Run from project root: uv run python -m src.generate_figures
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES, OUTPUTS, FONT_SCALE, SENT_COLORS_3
from .data_loading import load_annotations, parse_demographics, create_profiles
from .embeddings import build_id_index
from .similarity import (
    compute_profile_coherence, compute_profile_sim_matrix,
    compute_image_conditioned_profile_sim, cluster_order,
)
from .topic_modeling import topic_sentiment_composition


def _abbrev(p: str) -> str:
    pts = [x.strip() for x in p.split("/")]
    return f"{pts[0][0]}/{pts[1][:4]}/{pts[2][:4]}/{pts[3][:4]}"


def main() -> None:
    sns.set_theme(font_scale=FONT_SCALE)
    FIGURES.mkdir(exist_ok=True)
    MUTED = sns.color_palette("muted")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading annotations...")
    df = load_annotations()
    df = parse_demographics(df)
    df = create_profiles(df)
    print(f"  {len(df):,} records, {df['profile'].nunique()} profiles")

    print("Loading embeddings...")
    cap_embs  = np.load(OUTPUTS / "caption_embeddings.npy")
    just_embs = np.load(OUTPUTS / "justification_embeddings.npy")
    cap_idx   = build_id_index(OUTPUTS / "caption_embeddings_ids.csv")

    # ── Fig 1: justification similarity by sentiment ───────────────────────────
    print("\n[1] fig_sim_by_sentiment_just ...")
    just_sim_df = pd.read_csv(OUTPUTS / "per_image_just_similarity.csv")
    img_sent = (
        df.groupby("image_id")["predicted_sentiment"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"predicted_sentiment": "majority_sentiment"})
    )
    just_sim_sent = just_sim_df.merge(img_sent, on="image_id")
    order = ["Positive", "Neutral", "Negative"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=just_sim_sent[just_sim_sent["majority_sentiment"].isin(order)],
        x="majority_sentiment", y="mean_just_sim", order=order,
        hue="majority_sentiment", palette=SENT_COLORS_3, legend=False, ax=ax,
    )
    ax.set_xlabel("Majority predicted sentiment (per image)")
    ax.set_ylabel("Mean pairwise cosine similarity (justifications)")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_sim_by_sentiment_just.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_sim_by_sentiment_just.png", bbox_inches="tight")
    plt.close()
    print("  Saved.")

    # ── Figs 2-3: within-profile coherence barplot + heatmap ─────────────────
    print("\n[2-3] fig_within_profile_coherence ...")
    WP_CACHE = OUTPUTS / "within_profile_coherence.csv"
    wp_df = compute_profile_coherence(df, cap_embs, just_embs, cap_idx, WP_CACHE)
    print(f"  {len(wp_df)} profiles")

    metrics = [
        ("caption_coherence",       "Median pairwise cosine similarity\n(captions)"),
        ("justification_coherence", "Median pairwise cosine similarity\n(justifications)"),
        ("perception_jaccard",      "Median pairwise Jaccard\n(perception tags)"),
    ]

    wp_plot = wp_df.sort_values("caption_coherence", ascending=True)
    sorted_profiles = wp_plot["profile"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(28, 8), sharey=True)
    for ax_i, (ax, (col, xlabel)) in enumerate(zip(axes, metrics)):
        vals = wp_plot.set_index("profile").loc[sorted_profiles, col]
        ax.barh(sorted_profiles, vals.values, color=MUTED[0], edgecolor="white", linewidth=0.4)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.tick_params(axis="x", labelsize=8)
        if ax_i > 0:
            ax.set_yticklabels([])
        mean_val = vals.mean()
        ax.axvline(mean_val, color="#d73027", linestyle="--", linewidth=1.2,
                   label=f"Mean={mean_val:.3f}")
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.subplots_adjust(left=0.22)
    fig.savefig(FIGURES / "fig_within_profile_coherence.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_within_profile_coherence.png", bbox_inches="tight")
    plt.close()

    wp_sorted = wp_df.sort_values("caption_coherence", ascending=False).copy()
    wp_sorted["short_label"] = wp_sorted["profile"].apply(
        lambda p: " · ".join(x.strip()[:5] for x in p.split("/")))
    hm_data = wp_sorted.set_index("short_label")[
        ["caption_coherence", "justification_coherence", "perception_jaccard"]]
    hm_data.columns = ["Caption\ncoherence", "Justif.\ncoherence", "Perception\nJaccard"]
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(hm_data, ax=ax, cmap="YlOrRd", annot=True, fmt=".3f",
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": "Median coherence score", "shrink": 0.6},
                annot_kws={"size": 8})
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_within_profile_coherence_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_within_profile_coherence_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  Saved.")

    # ── Figs 4-7: 24×24 inter-profile cosine similarity heatmaps (mean + median)
    print("\n[4-7] fig_cosine_sim_heatmap ...")
    mean_cap, mean_just, med_cap, med_just, prof_labs = compute_profile_sim_matrix(
        df, cap_embs, just_embs, cap_idx,
        mean_cap_cache=OUTPUTS / "profile_sim_matrix_caption_mean.npy",
        mean_just_cache=OUTPUTS / "profile_sim_matrix_just_mean.npy",
        med_cap_cache=OUTPUTS  / "profile_sim_matrix_caption_median.npy",
        med_just_cache=OUTPUTS / "profile_sim_matrix_just_median.npy",
        labs_cache=OUTPUTS     / "profile_sim_labels.csv",
    )
    print(f"  Matrix: {mean_cap.shape}")

    def _profile_heatmap(sim_mat, labels, colorbar_label, fname):
        short  = [_abbrev(p) for p in labels]
        order  = cluster_order(sim_mat)
        mat_o  = sim_mat[np.ix_(order, order)]
        labs_o = [short[i] for i in order]
        mat_df = pd.DataFrame(mat_o, index=labs_o, columns=labs_o)
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            mat_df, ax=ax, cmap="YlOrRd",
            annot=True, fmt=".2f", annot_kws={"size": 5},
            linewidths=0.3, linecolor="white",
            cbar_kws={"label": colorbar_label, "shrink": 0.7},
        )
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)
        plt.tight_layout()
        fig.savefig(FIGURES / f"{fname}.pdf", bbox_inches="tight")
        fig.savefig(FIGURES / f"{fname}.png", bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")

    _profile_heatmap(mean_cap,  prof_labs, "Mean cosine similarity (captions)",          "fig_cosine_sim_heatmap_caption_mean")
    _profile_heatmap(mean_just, prof_labs, "Mean cosine similarity (justifications)",     "fig_cosine_sim_heatmap_just_mean")
    _profile_heatmap(med_cap,   prof_labs, "Median cosine similarity (captions)",         "fig_cosine_sim_heatmap_caption_median")
    _profile_heatmap(med_just,  prof_labs, "Median cosine similarity (justifications)",   "fig_cosine_sim_heatmap_just_median")

    # ── Fig 6: caption topic × sentiment composition ──────────────────────────
    print("\n[6] fig_topic_caption_sentiment_composition ...")
    ann_topics     = pd.read_csv(OUTPUTS / "annotations_with_topics.csv")
    cap_topic_info = pd.read_csv(OUTPUTS / "caption_topic_info.csv")

    pivot_cap_norm = topic_sentiment_composition(ann_topics, "caption_topic", top_n=10)
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_cap_norm[["Positive", "Neutral", "Negative"]].plot(
        kind="bar", stacked=True, ax=ax,
        color=[SENT_COLORS_3["Positive"], SENT_COLORS_3["Neutral"], SENT_COLORS_3["Negative"]],
    )
    ax.set_xlabel("Caption topic ID")
    ax.set_ylabel("Proportion")
    ax.legend(title="Predicted sentiment", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    for bar_stack in ax.containers:
        ax.bar_label(bar_stack, fmt=lambda v: f"{v:.0%}" if v > 0.08 else "",
                     label_type="center", fontsize=8, color="white", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_topic_caption_sentiment_composition.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_topic_caption_sentiment_composition.png", bbox_inches="tight")
    plt.close()
    print("  Saved.")

    # ── Figs 7-8: topic × persona profile heatmaps ───────────────────────────
    print("\n[7-8] fig_topic_persona_heatmap ...")
    just_topic_info = pd.read_csv(OUTPUTS / "just_topic_info.csv")

    merged_tp = ann_topics.merge(
        df[["annotation_id", "profile_abbr"]], on="annotation_id", how="left")
    merged_tp = merged_tp.dropna(subset=["profile_abbr"])

    def _tp_heatmap(pivot_norm, fname):
        fig, ax = plt.subplots(figsize=(18, 7))
        sns.heatmap(pivot_norm, ax=ax, cmap="Blues", annot=False,
                    linewidths=0.3, linecolor="white",
                    cbar_kws={"label": "Proportion of profile annotations", "shrink": 0.6})
        ax.set_xlabel("Persona profile", fontsize=11)
        ax.set_ylabel("Topic ID", fontsize=11)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
        plt.tight_layout()
        fig.savefig(FIGURES / f"{fname}.pdf", bbox_inches="tight")
        fig.savefig(FIGURES / f"{fname}.png", bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")

    for modality, tcol, info_df in [
        ("caption", "caption_topic", cap_topic_info),
        ("just",    "just_topic",    just_topic_info),
    ]:
        top = info_df[info_df["Topic"] != -1].head(15)["Topic"].tolist()
        tp  = (merged_tp[merged_tp[tcol].isin(top)]
               .groupby([tcol, "profile_abbr"]).size().reset_index(name="count"))
        piv   = tp.pivot(index=tcol, columns="profile_abbr", values="count").fillna(0)
        piv_n = piv.div(piv.sum(axis=0), axis=1)
        _tp_heatmap(piv_n, f"fig_topic_persona_heatmap_{modality}")

    # ── Fig 9: image-conditioned cross-profile similarity (captions + justifs) ──
    print("\n[9] fig_cosine_cross_profile_heatmap ...")
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
    vmin_c = max(0.4, float(np.nanmin(ic_cap))  - 0.02)
    vmin_j = max(0.4, float(np.nanmin(ic_just)) - 0.02)

    fig, axes = plt.subplots(1, 2, figsize=(26, 10))
    for ax, mat, vmin, modality in [
        (axes[0], ic_cap,  vmin_c, "Captions"),
        (axes[1], ic_just, vmin_j, "Justifications"),
    ]:
        mat_df = pd.DataFrame(mat, index=short_labs, columns=short_labs)
        sns.heatmap(
            mat_df, ax=ax, cmap="coolwarm_r",
            vmin=vmin, vmax=1.0,
            annot=True, fmt=".2f", annot_kws={"fontsize": 5},
            linewidths=0.3, linecolor="white",
            cbar_kws={"label": "Mean cosine similarity", "shrink": 0.7},
        )
        ax.set_title(modality, fontsize=12)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", rotation=0,  labelsize=6)

    plt.suptitle("Cross-Profile Semantic Similarity — Image-Conditioned (Baseline)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_cosine_cross_profile_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_cosine_cross_profile_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig_cosine_cross_profile_heatmap")

    print("\nAll figures generated successfully.")
    print("Files in figures/:")
    for f in sorted(FIGURES.glob("fig_*.pdf")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()

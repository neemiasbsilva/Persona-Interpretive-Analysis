"""Convergence metrics: sentiment agreement, perception Jaccard, merged dimensions."""

from itertools import combinations

import numpy as np
import pandas as pd

from .metrics import jaccard


def compute_sentiment_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Per-image pairwise sentiment agreement (proportion of identical label pairs).

    Returns DataFrame with columns: image_id, sentiment_agreement, majority_sentiment.
    """
    rows = []
    for img_id, grp in df.groupby("image_id"):
        labels = grp["predicted_sentiment"].tolist()
        pairs = list(combinations(labels, 2))
        agree = float(np.mean([1.0 if a == b else 0.0 for a, b in pairs])) if pairs else 1.0
        majority = grp["predicted_sentiment"].mode().iloc[0]
        rows.append(
            {
                "image_id": img_id,
                "sentiment_agreement": agree,
                "majority_sentiment": majority,
            }
        )
    return pd.DataFrame(rows)


def compute_label_jaccard(df: pd.DataFrame) -> pd.DataFrame:
    """Per-image mean pairwise Jaccard similarity of predicted perception tag sets.

    Returns DataFrame with columns: image_id, label_jaccard.
    """
    rows = []
    for img_id, grp in df.groupby("image_id"):
        tag_lists = grp["predicted_perceptions"].tolist()
        scores = [jaccard(a, b, on_empty_union=1.0) for a, b in combinations(tag_lists, 2)]
        rows.append(
            {
                "image_id": img_id,
                "label_jaccard": float(np.mean(scores)) if scores else 1.0,
            }
        )
    return pd.DataFrame(rows)


def merge_convergence_dimensions(
    cap_sim_df: pd.DataFrame,
    sent_agree_df: pd.DataFrame,
    label_agree_df: pd.DataFrame,
    just_sim_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge all convergence dimensions into one per-image DataFrame.

    cap_sim_df must have column 'mean_sim'; rename to 'caption_sim'.
    just_sim_df (optional) must have column 'mean_just_sim'; rename to 'just_sim'.

    Returns DataFrame with: image_id, caption_sim, sentiment_agreement,
    majority_sentiment, label_jaccard, [just_sim].
    """
    result = (
        cap_sim_df.rename(columns={"mean_sim": "caption_sim"})[
            ["image_id", "caption_sim", "n_personas"]
        ]
        .merge(sent_agree_df, on="image_id")
        .merge(label_agree_df, on="image_id")
    )
    if just_sim_df is not None:
        result = result.merge(
            just_sim_df.rename(columns={"mean_just_sim": "just_sim"})[["image_id", "just_sim"]],
            on="image_id",
            how="left",
        )
    return result

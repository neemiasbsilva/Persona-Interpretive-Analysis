"""BERTopic fitting, topic–sentiment composition, topic–persona proportions."""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    BERTOPIC_MIN_CLUSTER_SIZE, BERTOPIC_N_COMPONENTS,
    BERTOPIC_N_NEIGHBORS, BERTOPIC_TOP_N_WORDS, BERTOPIC_MIN_DF,
)


def fit_bertopic(
    texts: List[str],
    embeddings: np.ndarray,
    min_cluster_size: int = BERTOPIC_MIN_CLUSTER_SIZE,
    n_components:     int = BERTOPIC_N_COMPONENTS,
    n_neighbors:      int = BERTOPIC_N_NEIGHBORS,
    top_n_words:      int = BERTOPIC_TOP_N_WORDS,
    min_df:           int = BERTOPIC_MIN_DF,
    random_state:     int = 42,
):
    """Fit a BERTopic model and return (model, topics, probs).

    Hyperparameters are sourced from config.py by default.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=n_neighbors, n_components=n_components,
        min_dist=0.0, metric="cosine", random_state=random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size, metric="euclidean",
        cluster_selection_method="eom", prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=min_df,
    )
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=top_n_words,
        verbose=True,
    )
    topics, probs = model.fit_transform(texts, embeddings)
    return model, topics, probs


def topic_sentiment_composition(
    df: pd.DataFrame,
    topic_col: str,
    top_n: int = 10,
    sent_col: str = "predicted_sentiment",
    sent_classes: List[str] = ("Positive", "Neutral", "Negative"),
) -> pd.DataFrame:
    """Compute normalized sentiment composition per topic (top *top_n* topics).

    Returns a pivot DataFrame: rows = topic IDs, cols = sentiment classes, values = proportion.
    """
    topic_info_col = topic_col  # the int topic column in df
    top_topics = (
        df[df[topic_info_col] != -1][topic_info_col]
        .value_counts()
        .head(top_n)
        .index.tolist()
    )
    sub = df[df[topic_info_col].isin(top_topics) & df[sent_col].isin(sent_classes)]
    counts = sub.groupby([topic_info_col, sent_col]).size().reset_index(name="count")
    pivot  = counts.pivot(index=topic_info_col, columns=sent_col, values="count").fillna(0)
    for cls in sent_classes:
        if cls not in pivot.columns:
            pivot[cls] = 0.0
    pivot_norm = pivot[list(sent_classes)].div(pivot[list(sent_classes)].sum(axis=1), axis=0)
    return pivot_norm


def topic_persona_proportions(
    ann_topics_df: pd.DataFrame,
    topic_col: str,
    top_n: int = 15,
    topic_info_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Column-normalized topic × persona profile proportion matrix.

    ann_topics_df must have columns: *topic_col*, profile_abbr.
    topic_info_df (optional) provides topic ordering by size; if None, uses value_counts.

    Returns pivot: rows = topic IDs, cols = profile_abbr strings, values = proportion.
    """
    if topic_info_df is not None:
        top = topic_info_df[topic_info_df["Topic"] != -1].head(top_n)["Topic"].tolist()
    else:
        top = (
            ann_topics_df[ann_topics_df[topic_col] != -1][topic_col]
            .value_counts()
            .head(top_n)
            .index.tolist()
        )
    sub   = ann_topics_df[ann_topics_df[topic_col].isin(top)]
    counts = sub.groupby([topic_col, "profile_abbr"]).size().reset_index(name="count")
    pivot  = counts.pivot(index=topic_col, columns="profile_abbr", values="count").fillna(0)
    pivot_norm = pivot.div(pivot.sum(axis=0), axis=1)  # column-normalize
    return pivot_norm


def save_topic_assignments(
    df: pd.DataFrame,
    output_path: Path,
    cap_topic_info,
    just_topic_info,
    cap_info_path: Path,
    just_info_path: Path,
) -> None:
    """Save topic assignment CSV and topic info CSVs."""
    cols = ["annotation_id", "persona_id", "image_id",
            "predicted_sentiment", "caption_topic", "just_topic"]
    df[cols].to_csv(output_path, index=False)
    cap_topic_info.to_csv(cap_info_path, index=False)
    just_topic_info.to_csv(just_info_path, index=False)

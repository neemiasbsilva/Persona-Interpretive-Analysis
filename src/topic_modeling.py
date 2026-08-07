"""BERTopic fitting, topic-sentiment composition, topic-persona proportions."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BERTOPIC_MIN_CLUSTER_SIZE,
    BERTOPIC_MIN_DF,
    BERTOPIC_N_COMPONENTS,
    BERTOPIC_N_NEIGHBORS,
    BERTOPIC_TOP_N_WORDS,
)


def fit_bertopic(
    texts: list[str],
    embeddings: np.ndarray,
    *,
    min_cluster_size: int = BERTOPIC_MIN_CLUSTER_SIZE,
    n_components: int = BERTOPIC_N_COMPONENTS,
    n_neighbors: int = BERTOPIC_N_NEIGHBORS,
    top_n_words: int = BERTOPIC_TOP_N_WORDS,
    min_df: int = BERTOPIC_MIN_DF,
    random_state: int = 42,
) -> tuple[Any, list[int], np.ndarray | None]:
    """Fit a BERTopic model and return (model, topics, probs).

    Hyperparameters are sourced from config.py by default.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
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


def auto_topic_labels(
    topic_info_df: pd.DataFrame,
    n_words: int = 3,
    top_n: int | None = None,
) -> dict[int, str]:
    """Derive a short human-readable label per topic from its top keywords.

    Model-agnostic replacement for hand-curated topic names: each label is the
    topic's leading keywords joined by commas (e.g. ``"street, road, traffic"``).

    Args:
        topic_info_df: BERTopic ``get_topic_info()`` output. Must contain a
            ``Topic`` column and either a ``Representation`` (list of words) or a
            ``Name`` (``"0_word1_word2_..."``) column.
        n_words: number of keywords to include per label.
        top_n: if given, only label the ``top_n`` largest non-noise topics.

    Returns:
        dict mapping int topic id -> label string (noise topic -1 excluded).
    """
    info = topic_info_df[topic_info_df["Topic"] != -1].copy()
    if top_n is not None:
        info = info.head(top_n)

    def _words(row: pd.Series) -> list[str]:
        rep = row.get("Representation")
        if isinstance(rep, (list, tuple)) and rep:
            return [str(w) for w in rep]
        name = str(row.get("Name", ""))
        parts = name.split("_")[1:] if "_" in name else [name]
        return [p for p in parts if p]

    return {int(row["Topic"]): ", ".join(_words(row)[:n_words]) for _, row in info.iterrows()}


def save_topic_labels(labels: dict[int, str], path: Path) -> None:
    """Persist a {topic_id: label} mapping to a ``topic,label`` CSV."""
    (
        pd.DataFrame({"topic": list(labels.keys()), "label": list(labels.values())})
        .sort_values("topic")
        .to_csv(Path(path), index=False)
    )


def load_topic_labels(path: Path) -> dict[int, str]:
    """Load a ``topic,label`` CSV into a {int topic_id: label} mapping."""
    df = pd.read_csv(Path(path))
    return {int(topic): str(label) for topic, label in zip(df["topic"], df["label"], strict=True)}


def save_topic_assignments(
    df: pd.DataFrame,
    output_path: Path,
    cap_topic_info: pd.DataFrame,
    just_topic_info: pd.DataFrame,
    *,
    cap_info_path: Path,
    just_info_path: Path,
) -> None:
    """Save topic assignment CSV and topic info CSVs."""
    cols = [
        "annotation_id",
        "persona_id",
        "image_id",
        "predicted_sentiment",
        "caption_topic",
        "just_topic",
    ]
    df[cols].to_csv(output_path, index=False)
    cap_topic_info.to_csv(cap_info_path, index=False)
    just_topic_info.to_csv(just_info_path, index=False)

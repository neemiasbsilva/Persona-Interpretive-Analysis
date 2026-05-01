"""Sentiment analysis: VADER, RoBERTa, and agreement metrics."""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from .config import ROBERTA_MODEL

VADER_THRESHOLD = 0.05

SENTIMENT_MAPS: Dict[str, Dict[str, int]] = {
    "p2neg":  {"Positive": 1, "SlightlyPositive": 1, "Neutral": 0,
               "SlightlyNegative": 0, "Negative": 0},
    "p2plus": {"Positive": 0, "SlightlyPositive": 0, "Neutral": 0,
               "SlightlyNegative": 1, "Negative": 1},
    "p3":     {"Positive": 2, "SlightlyPositive": 2, "Neutral": 0,
               "SlightlyNegative": 1, "Negative": 1},
}

COLLAPSE_5_TO_3 = {
    "Positive": "Positive", "SlightlyPositive": "Positive",
    "Neutral": "Neutral",
    "SlightlyNegative": "Negative", "Negative": "Negative",
}


def analyze_vader(texts: List[str]) -> pd.DataFrame:
    """Apply VADER to a list of texts.

    Returns DataFrame with columns: compound, vader_label.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    compounds = [sid.polarity_scores(t)["compound"] for t in texts]
    labels = [
        "Positive" if s >= VADER_THRESHOLD else ("Negative" if s <= -VADER_THRESHOLD else "Neutral")
        for s in compounds
    ]
    return pd.DataFrame({"compound": compounds, "vader_label": labels})


def analyze_roberta(
    texts: List[str],
    model_name: str = ROBERTA_MODEL,
    cache_path: Optional[Path] = None,
    device: int = -1,
) -> pd.DataFrame:
    """Apply RoBERTa sentiment model to texts, with optional CSV cache.

    Returns DataFrame with columns: roberta_label, roberta_score.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        return pd.read_csv(cache_path)

    from transformers import pipeline
    _label_map = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
    pipe = pipeline(
        "text-classification",
        model=model_name,
        truncation=True,
        max_length=512,
        device=device,
    )
    results = pipe(texts, batch_size=32)
    out = pd.DataFrame({
        "roberta_label": [_label_map.get(r["label"].lower(), r["label"]) for r in results],
        "roberta_score": [r["score"] for r in results],
    })
    if cache_path:
        out.to_csv(cache_path, index=False)
    return out


def agreement_metrics(y_true, y_pred, average="macro") -> Dict[str, float]:
    """Compute accuracy, Cohen's kappa, and macro-F1."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa":    cohen_kappa_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def agreement_with_predicted(
    df: pd.DataFrame,
    inferred_col: str,
    predicted_col: str = "predicted_sentiment",
) -> Dict[str, float]:
    """Compute agreement between an inferred sentiment col and MLLM predicted labels.

    Collapses the 5-class predicted scale to 3 classes before comparison.
    """
    sub = df[[inferred_col, predicted_col]].dropna()
    pred_3 = sub[predicted_col].map(COLLAPSE_5_TO_3)
    valid  = pred_3.notna()
    return agreement_metrics(pred_3[valid], sub[inferred_col][valid])


def agreement_with_human_gt(
    df: pd.DataFrame,
    approach_col: str,
    gt_df: pd.DataFrame,
    sigma_thresholds: List[int] = (3, 4, 5),
    problems: List[str] = ("p2neg", "p2plus", "p3"),
) -> pd.DataFrame:
    """Compute macro-F1 and accuracy vs. PerceptSent ground truth.

    Args:
        df:         per-image DataFrame with image_id and *approach_col* (predicted sentiment).
        approach_col: column name for the sentiment to evaluate.
        gt_df:      PerceptSent ground truth with image_id and sigma-specific columns.
        sigma_thresholds: list of consensus levels σ to evaluate.
        problems:   list of task formulations to evaluate.

    Returns:
        DataFrame with columns: sigma, problem, f1_macro, accuracy.
    """
    rows = []
    for sigma in sigma_thresholds:
        sigma_col = f"label_{sigma}"
        if sigma_col not in gt_df.columns:
            continue
        sub_gt = gt_df[gt_df[sigma_col].notna()][["image_id", sigma_col]]
        for problem in problems:
            smap = SENTIMENT_MAPS[problem]
            merged = sub_gt.merge(df[["image_id", approach_col]], on="image_id", how="inner")
            merged = merged.dropna(subset=[approach_col])
            y_true = merged[sigma_col].astype(int)
            y_pred = merged[approach_col].map(smap)
            valid  = y_pred.notna()
            if valid.sum() == 0:
                continue
            metrics = agreement_metrics(y_true[valid], y_pred[valid].astype(int))
            rows.append({"sigma": sigma, "problem": problem, **metrics})
    return pd.DataFrame(rows)

"""Between-group Jensen--Shannon divergence of justification-topic distributions.

For each image and persona dimension (gender, economic status, political
orientation, personality), we build each group's normalized distribution over
the justification topics and measure how far the groups diverge using the
(squared) Jensen--Shannon divergence, averaged over all group pairs and then
over images. A within-image label-permutation test provides a p-value: the
group labels are shuffled among the annotations of each image (preserving group
sizes), so the null hypothesis is that persona group membership carries no
information about which topics are emphasized.

The distribution is taken over the full set of assigned justification topics
(BERTopic noise, topic ``-1``, is excluded). The topic x persona heatmap
visualizes only the top labeled topics for readability, whereas the JSD uses
the complete topic distribution so every image contributes.
"""

from collections.abc import Iterable
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from .schema import DEMO_COLS


def _pairwise_jsd(dists: list[np.ndarray]) -> float:
    """Mean squared Jensen--Shannon divergence (base 2) over all group pairs.

    ``jensenshannon`` returns the JS *distance*; squaring yields the JS
    *divergence* in [0, 1]. Averaging over pairs keeps dimensions with
    different numbers of groups (personality has 3) comparable to the
    two-level dimensions.
    """
    vals = [jensenshannon(a, b, base=2) ** 2 for a, b in combinations(dists, 2)]
    return float(np.mean(vals)) if vals else np.nan


def _mean_jsd(per_image: list[tuple[np.ndarray, np.ndarray, int]], n_topics: int) -> float:
    """Mean per-image between-group JSD for one dimension.

    ``per_image`` is a list of (topic_idx, group_codes, n_groups) tuples.
    """
    jvals = []
    for topic_idx, codes, n_groups in per_image:
        dists = []
        ok = True
        for k in range(n_groups):
            counts = np.bincount(topic_idx[codes == k], minlength=n_topics).astype(float)
            total = counts.sum()
            if total == 0:
                ok = False
                break
            dists.append(counts / total)
        if ok:
            jvals.append(_pairwise_jsd(dists))
    return float(np.nanmean(jvals)) if jvals else np.nan


def compute_topic_jsd(
    df: pd.DataFrame,
    ann_topics: pd.DataFrame,
    *,
    topics: Iterable[int] | None = None,
    n_perm: int = 1000,
    seed: int = 42,
    noise_topic: int = -1,
) -> pd.DataFrame:
    """Compute per-dimension between-group topic JSD with a permutation p-value.

    Parameters
    ----------
    df : DataFrame with ``annotation_id`` and the four ``DEMO_COLS``.
    ann_topics : DataFrame with ``annotation_id``, ``image_id``, ``just_topic``.
    topics : iterable of topic ids to build the distribution over. If ``None``
        (default), all assigned topics in the data are used, excluding
        ``noise_topic``.
    n_perm : number of within-image label permutations for the null.
    seed : RNG seed for reproducibility.
    noise_topic : topic id treated as BERTopic noise and excluded (default -1).

    Returns:
    -------
    DataFrame with columns ``dimension``, ``jsd``, ``p_perm``, ``n_images``.
    """
    rng = np.random.default_rng(seed)

    demo = df[["annotation_id", *DEMO_COLS]].drop_duplicates("annotation_id")
    merged = ann_topics.merge(demo, on="annotation_id", how="inner")

    if topics is None:
        topics = sorted(t for t in merged["just_topic"].unique() if t != noise_topic)
    else:
        topics = [t for t in topics if t != noise_topic]
    topic_to_col = {t: i for i, t in enumerate(topics)}
    merged = merged[merged["just_topic"].isin(topics)].copy()
    merged["topic_idx"] = merged["just_topic"].map(topic_to_col).astype(int)
    n_topics = len(topics)

    results = []
    for dim in DEMO_COLS:
        per_image = []
        for _img_id, g in merged.groupby("image_id"):
            labels = g[dim].to_numpy()
            uniq = np.unique(labels)
            if len(uniq) < 2:
                continue
            code = {u: k for k, u in enumerate(uniq)}
            codes = np.array([code[x] for x in labels])
            per_image.append((g["topic_idx"].to_numpy(), codes, len(uniq)))

        observed = _mean_jsd(per_image, n_topics)

        count_ge = 0
        for _ in range(n_perm):
            permuted = []
            for topic_idx, codes, n_groups in per_image:
                pc = codes.copy()
                rng.shuffle(pc)
                permuted.append((topic_idx, pc, n_groups))
            if _mean_jsd(permuted, n_topics) >= observed:
                count_ge += 1
        p_perm = (count_ge + 1) / (n_perm + 1)

        results.append(
            {
                "dimension": dim,
                "jsd": observed,
                "p_perm": p_perm,
                "n_images": len(per_image),
            }
        )

    return pd.DataFrame(results)

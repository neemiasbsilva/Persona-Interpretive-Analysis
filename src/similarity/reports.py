"""Canonical within/cross and modality-contrast result tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import OUTPUTS
from ..schema import DEMO_COLS, MODALITIES
from .paired_stats import (
    EPS,
    N_BOOT,
    N_WILCOXON_RESAMPLES,
    benjamini_hochberg,
    paired_stats,
    significance_stars,
    stable_seed_sequence,
)


def validate_within_cross_frame(wc_df: pd.DataFrame) -> pd.DataFrame:
    """Validate the one-row-per-image/dimension/modality analysis frame."""
    required = {
        "image_id",
        "dimension",
        "modality",
        "within_mean",
        "cross_mean",
    }
    missing = sorted(required.difference(wc_df.columns))
    if missing:
        raise ValueError(f"within/cross frame is missing columns: {missing}")
    key = ["image_id", "dimension", "modality"]
    if wc_df[key].isna().any(axis=None):
        raise ValueError("image_id, dimension, and modality must not be missing")
    duplicated = wc_df.duplicated(key, keep=False)
    if duplicated.any():
        examples = wc_df.loc[duplicated, key].head(5).to_dict("records")
        raise ValueError(
            "within/cross frame must have unique image/dimension/modality keys; "
            f"duplicate examples: {examples}"
        )
    values = wc_df[["within_mean", "cross_mean"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("within_mean and cross_mean must contain only finite values")
    return wc_df


def ordered_cell(
    wc_df: pd.DataFrame,
    dimension: str,
    modality: str,
) -> pd.DataFrame:
    """Select a complete cell in a canonical, row-order-invariant image order."""
    cell = wc_df[(wc_df["dimension"] == dimension) & (wc_df["modality"] == modality)].copy()
    if cell.empty:
        raise ValueError(f"no observations for dimension={dimension!r}, modality={modality!r}")
    cell["_image_order"] = cell["image_id"].map(lambda value: (type(value).__name__, repr(value)))
    return cell.sort_values("_image_order", kind="stable").drop(columns="_image_order")


def run_within_cross_significance(
    wc_df: pd.DataFrame | None = None,
    *,
    source_path: Path | None = None,
    demo_cols: list = DEMO_COLS,
    modalities: tuple[str, ...] = MODALITIES,
    n_boot: int = N_BOOT,
    seed: int = 42,
    wilcoxon_resamples: int = N_WILCOXON_RESAMPLES,
    eps: float = EPS,
    out_path: Path | None = None,
    seed_namespace: str = "within_cross",
) -> pd.DataFrame:
    """Magnitude and significance of the within-minus-cross similarity gap.

    Consumes the per-image means produced by ``compute_within_cross_similarity``
    (one ``within_mean`` and one ``cross_mean`` per image), so the unit of
    analysis is the image: each (dimension, modality) cell reduces to two
    vectors of length n_images (50 for the baseline corpora).

    The two vectors describe the *same* images and are therefore paired. The
    test is a two-sided Wilcoxon signed-rank on the per-image gap.  It evaluates
    a signed-rank location null (with a symmetry assumption for the usual
    location interpretation), not a null about the arithmetic mean.  The exact
    binomial sign test is retained as a sensitivity analysis that does not
    require symmetry.

    ``eps`` is a post-hoc descriptive reference for caption effects, not a
    formal decision threshold.  Its output is left missing for other modalities.

    ``seed_namespace`` prefixes the per-cell seed key, so a second construction
    over the same (dimension, modality) grid draws independent resampling
    streams.  The default reproduces the published within/cross keys exactly.

    Reported per cell: arithmetic mean Delta with a 95% BCa bootstrap interval,
    median, sign/zero/tie counts, the Wilcoxon statistic and method, exact sign
    test p-value, matched-pairs rank-biserial correlation, and BH-adjusted
    p-values for both inferential tests.

    Unlike the other functions here, ``out_path`` is write-only rather than a
    read-through cache: these are the numbers quoted in the paper, and silently
    returning stale statistics after the underlying means changed would be worse
    than recomputing (the whole table takes a few seconds).

    Returns DataFrame with columns:
        dimension, modality, n_images, within_mean, cross_mean, delta,
        median_diff, ci_lo, ci_hi, n_pos, n_zero, n_abs_ties, w_stat,
        p_wilcoxon, p_sign, rank_biserial, wilcoxon_method,
        p_wilcoxon_bh, p_sign_bh, p_bh,
        ci_relation_to_exploratory_reference, stars
    """
    if wc_df is None:
        source_path = Path(source_path) if source_path else OUTPUTS / "within_cross_persona_sim.csv"
        wc_df = pd.read_csv(source_path)
    wc_df = validate_within_cross_frame(wc_df)

    rows = []
    for dim in demo_cols:
        for mod in modalities:
            sub = ordered_cell(wc_df, dim, mod)
            within = sub["within_mean"].to_numpy()
            cross = sub["cross_mean"].to_numpy()
            stats = paired_stats(
                within,
                cross,
                stable_seed_sequence(seed, seed_namespace, dim, mod),
                n_boot,
                wilcoxon_resamples,
            )
            rows.append(
                {
                    "dimension": dim,
                    "modality": mod,
                    "n_images": stats["n_images"],
                    "within_mean": float(within.mean()),
                    "cross_mean": float(cross.mean()),
                    "delta": stats["mean_diff"],
                    "median_diff": stats["median_diff"],
                    "ci_lo": stats["ci_lo"],
                    "ci_hi": stats["ci_hi"],
                    "n_pos": stats["n_pos"],
                    "n_zero": stats["n_zero"],
                    "n_abs_ties": stats["n_abs_ties"],
                    "w_stat": stats["w_stat"],
                    "p_wilcoxon": stats["p_wilcoxon"],
                    "p_sign": stats["p_sign"],
                    "rank_biserial": stats["rank_biserial"],
                    "wilcoxon_method": stats["wilcoxon_method"],
                }
            )

    result = pd.DataFrame(rows)
    result["p_wilcoxon_bh"] = benjamini_hochberg(result["p_wilcoxon"].to_numpy())
    result["p_sign_bh"] = benjamini_hochberg(result["p_sign"].to_numpy())
    result["p_bh"] = result["p_wilcoxon_bh"]
    relation = pd.Series(pd.NA, index=result.index, dtype="string")
    caption = result["modality"].eq("caption")
    relation.loc[caption] = np.select(
        [
            result.loc[caption, "ci_hi"] < eps,
            result.loc[caption, "ci_lo"] > eps,
        ],
        [
            "below",
            "above",
        ],
        default="overlaps",
    )
    result["ci_relation_to_exploratory_reference"] = relation
    result["stars"] = result["p_wilcoxon"].map(significance_stars)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
    return result


def run_modality_contrast(
    wc_df: pd.DataFrame | None = None,
    *,
    source_path: Path | None = None,
    demo_cols: list = DEMO_COLS,
    reference: str = "caption",
    targets: tuple[str, ...] = ("justification",),
    n_boot: int = N_BOOT,
    seed: int = 42,
    wilcoxon_resamples: int = N_WILCOXON_RESAMPLES,
    out_path: Path | None = None,
    seed_namespace: str = "modality_contrast",
) -> pd.DataFrame:
    """Does an attribute move interpretation more than it moves description?

    Tests the paper's central claim directly, as a positive hypothesis rather
    than as an absence of evidence. For each dimension we form the per-image
    within-minus-cross gap under the target modality and under the reference
    modality, and compare them paired by image -- so a positive contrast means
    the attribute reshapes the target modality more than it reshapes captions.

    The default comparison is justification cosine versus caption cosine.  A
    perception contrast is intentionally not a default because Jaccard and
    cosine effect sizes are not commensurate.  The inferential and resampling
    machinery is shared with ``run_within_cross_significance``.

    ``seed_namespace`` prefixes the per-cell seed key, so a second construction
    over the same (dimension, target, reference) grid draws independent
    resampling streams.  The default reproduces the published marginal
    justification-minus-caption keys exactly.

    Returns DataFrame with columns:
        dimension, target, reference, n_images, mean_contrast, median_contrast,
        ci_lo, ci_hi, n_pos, n_zero, n_abs_ties, w_stat, p_wilcoxon, p_sign,
        rank_biserial, wilcoxon_method, p_wilcoxon_bh, p_sign_bh, p_bh
    """
    if wc_df is None:
        source_path = Path(source_path) if source_path else OUTPUTS / "within_cross_persona_sim.csv"
        wc_df = pd.read_csv(source_path)
    wc_df = validate_within_cross_frame(wc_df)

    def _gap(dim: str, modality: str) -> pd.Series:
        sub = ordered_cell(wc_df, dim, modality).set_index("image_id")
        return sub["within_mean"] - sub["cross_mean"]

    rows = []
    for dim in demo_cols:
        ref_gap = _gap(dim, reference)
        for target in targets:
            tgt_gap = _gap(dim, target)
            if set(tgt_gap.index) != set(ref_gap.index):
                target_only = sorted(set(tgt_gap.index).difference(ref_gap.index), key=repr)[:5]
                reference_only = sorted(set(ref_gap.index).difference(tgt_gap.index), key=repr)[:5]
                raise ValueError(
                    f"image pairing mismatch for {dim!r}, {target!r} versus "
                    f"{reference!r}; target-only={target_only}, "
                    f"reference-only={reference_only}"
                )
            image_order = sorted(
                tgt_gap.index, key=lambda value: (type(value).__name__, repr(value))
            )

            stats = paired_stats(
                tgt_gap.loc[image_order].to_numpy(),
                ref_gap.loc[image_order].to_numpy(),
                stable_seed_sequence(seed, seed_namespace, dim, target, reference),
                n_boot,
                wilcoxon_resamples,
            )
            rows.append(
                {
                    "dimension": dim,
                    "target": target,
                    "reference": reference,
                    "n_images": stats["n_images"],
                    "mean_contrast": stats["mean_diff"],
                    "median_contrast": stats["median_diff"],
                    "ci_lo": stats["ci_lo"],
                    "ci_hi": stats["ci_hi"],
                    "n_pos": stats["n_pos"],
                    "n_zero": stats["n_zero"],
                    "n_abs_ties": stats["n_abs_ties"],
                    "w_stat": stats["w_stat"],
                    "p_wilcoxon": stats["p_wilcoxon"],
                    "p_sign": stats["p_sign"],
                    "rank_biserial": stats["rank_biserial"],
                    "wilcoxon_method": stats["wilcoxon_method"],
                }
            )

    result = pd.DataFrame(rows)
    result["p_wilcoxon_bh"] = benjamini_hochberg(result["p_wilcoxon"].to_numpy())
    result["p_sign_bh"] = benjamini_hochberg(result["p_sign"].to_numpy())
    result["p_bh"] = result["p_wilcoxon_bh"]

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
    return result

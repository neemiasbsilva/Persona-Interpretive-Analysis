"""Paired inference on image-level differences: Wilcoxon, BCa, sign test, BH."""

import hashlib

import numpy as np

EPS = 0.01

N_BOOT = 100_000
N_WILCOXON_RESAMPLES = 100_000


def significance_stars(p: float) -> str:
    """Significance markers.

    Retained for figure back-compatibility only. Under the paired tests used
    here every cell lands at ``***``, so the stars carry no information; the
    figures annotate Delta and the exploratory reference line instead.
    """
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini--Hochberg adjusted p-values via SciPy's maintained routine."""
    from scipy.stats import false_discovery_control

    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1:
        raise ValueError("p-values must be a one-dimensional array")
    if len(p) == 0:
        return p.copy()
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be finite and lie in [0, 1]")
    return np.asarray(false_discovery_control(p, method="bh"), dtype=float)


def matched_pairs_rank_biserial(diff: np.ndarray) -> float:
    """Wilcoxon matched-pairs rank-biserial: (W+ - W-) / (W+ + W-).

    The matched-pairs form. This repository never uses the independent-samples
    variant: within and cross means come from the same image. +1 means every
    non-zero per-image difference is positive.
    """
    nz = diff[diff != 0]
    if len(nz) == 0:
        return 0.0
    from scipy.stats import rankdata

    ranks = rankdata(np.abs(nz), method="average")
    w_pos = ranks[nz > 0].sum()
    w_neg = ranks[nz < 0].sum()
    total = w_pos + w_neg
    return float((w_pos - w_neg) / total) if total else 0.0


def stable_seed_sequence(
    seed: int,
    *cell_key: object,
) -> np.random.SeedSequence:
    """Return a reproducible seed that does not depend on table iteration order."""
    digest = hashlib.sha256("\x1f".join(map(str, cell_key)).encode("utf-8")).digest()
    words = np.frombuffer(digest[:16], dtype="<u4").astype(np.uint32).tolist()
    return np.random.SeedSequence([int(seed), *words])


def validate_paired_vectors(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize two vectors paired observation by observation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("paired samples must be one-dimensional")
    if a.shape != b.shape:
        raise ValueError(f"paired samples must have equal lengths; got {len(a)} and {len(b)}")
    if len(a) < 2:
        raise ValueError("paired comparisons require at least two observations")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired samples must contain only finite values")
    return a, b


def paired_stats(
    a: np.ndarray,
    b: np.ndarray,
    rng: int | np.random.SeedSequence | np.random.Generator = 42,
    n_boot: int = N_BOOT,
    wilcoxon_resamples: int = N_WILCOXON_RESAMPLES,
) -> dict:
    """Paired comparison of two per-image vectors.

    The already-computed difference is passed to a two-sided Wilcoxon test to
    avoid subtraction-induced rank changes inside SciPy.  Exact Wilcoxon
    inference is used only when its no-zero/no-tie assumptions are met; other
    non-degenerate cases use a deterministic permutation implementation.  An
    exact binomial sign test is reported as a symmetry-robust sensitivity check.
    The arithmetic mean receives a BCa bootstrap confidence interval.
    """
    from scipy.stats import PermutationMethod, binomtest, bootstrap, wilcoxon

    a, b = validate_paired_vectors(a, b)
    if n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if wilcoxon_resamples < 1:
        raise ValueError("wilcoxon_resamples must be a positive integer")

    if isinstance(rng, np.random.Generator):
        entropy = rng.integers(0, 2**32, size=4, dtype=np.uint32).tolist()
        seed_sequence = np.random.SeedSequence(entropy)
    elif isinstance(rng, np.random.SeedSequence):
        seed_sequence = rng
    else:
        seed_sequence = np.random.SeedSequence(int(rng))
    bootstrap_seed, permutation_seed = seed_sequence.spawn(2)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    permutation_rng = np.random.default_rng(permutation_seed)

    diff = np.asarray(a - b, dtype=float)
    n = len(diff)
    nonzero = diff[diff != 0]
    abs_nonzero = np.abs(nonzero)
    _, tie_counts = np.unique(abs_nonzero, return_counts=True)
    n_abs_ties = int(tie_counts[tie_counts > 1].sum())
    n_zero = int((diff == 0).sum())
    n_pos = int((diff > 0).sum())

    if np.all(diff == diff[0]):
        lo = hi = float(diff[0])
    else:
        ci = bootstrap(
            (diff,),
            np.mean,
            n_resamples=int(n_boot),
            confidence_level=0.95,
            alternative="two-sided",
            method="BCa",
            rng=bootstrap_rng,
        ).confidence_interval
        lo, hi = float(ci.low), float(ci.high)
        if not np.isfinite([lo, hi]).all():
            raise RuntimeError("BCa bootstrap produced a non-finite confidence interval")

    if len(nonzero) == 0:
        w_stat, p_wilcoxon = 0.0, 1.0
        wilcoxon_method = "all_zero"
    else:
        if n_zero == 0 and n_abs_ties == 0:
            method = "exact"
            wilcoxon_method = "exact"
        else:
            method = PermutationMethod(
                n_resamples=int(wilcoxon_resamples),
                rng=permutation_rng,
            )
            wilcoxon_method = "permutation"
        wilcoxon_result = wilcoxon(
            diff,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method=method,
        )
        w_stat = float(wilcoxon_result.statistic)
        p_wilcoxon = float(wilcoxon_result.pvalue)

    n_nonzero = len(nonzero)
    p_sign = (
        float(binomtest(n_pos, n_nonzero, p=0.5, alternative="two-sided").pvalue)
        if n_nonzero
        else 1.0
    )

    return {
        "n_images": n,
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_pos": n_pos,
        "n_zero": n_zero,
        "n_abs_ties": n_abs_ties,
        "w_stat": float(w_stat),
        "p_wilcoxon": float(p_wilcoxon),
        "p_sign": p_sign,
        "rank_biserial": matched_pairs_rank_biserial(diff),
        "wilcoxon_method": wilcoxon_method,
    }

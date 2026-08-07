"""Cross-model correlation of inter-profile similarity matrices.

Quantifies whether the persona-conditioned *pattern* of inter-profile similarity
reproduces across two annotating models (e.g. Qwen3-VL vs. Gemma4). For each
modality we correlate the two models' 24x24 image-conditioned inter-profile
similarity matrices cell-by-cell, over the upper triangle. By default the
diagonal (the 24 within-profile self-similarity cells) is *included*
(``np.triu_indices(n, k=0)`` -> n=300 distinct profile pairs); pass
``--exclude-diagonal`` to use only the strictly off-diagonal cross-profile cells
(``k=1`` -> n=276).

The two matrices are aligned by ``ic_profile_sim_labels.csv``; the script asserts
that the label files are identical before correlating, so a silent misalignment
cannot corrupt the numbers.

Run from project root:
    uv run python -m src.cross_model_correlation
    uv run python -m src.cross_model_correlation --model-a qwen-vl --model-b gemma-4-E4B-it_t01
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .config import PROJECT_ROOT
from .models import outputs_dir

MODALITY_FILES = {
    "caption": "ic_profile_sim_caption.npy",
    "justification": "ic_profile_sim_just.npy",
    "perception": "ic_profile_sim_jaccard.npy",
}
LABELS_FILE = "ic_profile_sim_labels.csv"


def _load_labels(model: str) -> str:
    """Return the raw label-file text for alignment checking."""
    path = outputs_dir(model) / LABELS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing label file for '{model}': {path}")
    return path.read_text()


def compute_cross_model_correlation(
    model_a: str,
    model_b: str,
    *,
    include_diagonal: bool = True,
) -> pd.DataFrame:
    """Correlate each modality's inter-profile matrix between two models.

    By default the correlation is taken over the upper triangle *including* the
    diagonal (the 24 within-profile self-similarity cells), i.e. the full set of
    distinct profile pairs. Set ``include_diagonal=False`` to restrict to the
    strictly off-diagonal (cross-profile) cells.

    Args:
        model_a: First model directory id.
        model_b: Second model directory id.
        include_diagonal: Keep the within-profile diagonal cells.

    Returns:
        Tidy DataFrame with one row per modality.
    """
    labels_a = _load_labels(model_a)
    labels_b = _load_labels(model_b)
    if labels_a != labels_b:
        raise ValueError(
            f"Profile label order differs between '{model_a}' and '{model_b}'. "
            f"The similarity matrices are not aligned; refusing to correlate."
        )

    tri_k = 0 if include_diagonal else 1
    rows = []
    n_profiles = None
    for modality, fname in MODALITY_FILES.items():
        path_a = outputs_dir(model_a) / fname
        path_b = outputs_dir(model_b) / fname
        if not path_a.exists() or not path_b.exists():
            raise FileNotFoundError(
                f"Missing '{modality}' matrix. Run generate_figures.py for both "
                f"models first.\n  {path_a} exists={path_a.exists()}\n"
                f"  {path_b} exists={path_b.exists()}"
            )
        mat_a = np.load(path_a)
        mat_b = np.load(path_b)
        if mat_a.shape != mat_b.shape:
            raise ValueError(f"Shape mismatch for '{modality}': {mat_a.shape} vs {mat_b.shape}")
        n_profiles = mat_a.shape[0]

        idx = np.triu_indices(mat_a.shape[0], k=tri_k)
        va, vb = mat_a[idx], mat_b[idx]
        pearson_r, pearson_p = pearsonr(va, vb)
        spearman_r, spearman_p = spearmanr(va, vb)

        rows.append(
            {
                "modality": modality,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "n": len(va),
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["model_a"] = model_a
    df.attrs["model_b"] = model_b
    df.attrs["n_profiles"] = n_profiles
    return df


def main() -> None:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", default="qwen-vl")
    parser.add_argument("--model-b", default="gemma-4-E4B-it_t01")
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "outputs" / "cross_model_correlation.csv"),
        help="Where to save the tidy correlation table (CSV).",
    )
    parser.add_argument(
        "--exclude-diagonal",
        action="store_true",
        help="Correlate only off-diagonal (cross-profile) cells (k=1, n=276). "
        "Default includes the within-profile diagonal (k=0, n=300).",
    )
    args = parser.parse_args()

    include_diagonal = not args.exclude_diagonal
    df = compute_cross_model_correlation(
        args.model_a,
        args.model_b,
        include_diagonal=include_diagonal,
    )

    tri_desc = (
        "upper triangle incl. diagonal" if include_diagonal else "off-diagonal upper triangle"
    )
    print(f"Cross-model correlation: {args.model_a} vs {args.model_b}")
    print(
        f"({tri_desc}, n={df['n'].iloc[0]} profile pairs from {df.attrs['n_profiles']} profiles)\n"
    )
    for _, r in df.iterrows():
        print(
            f"  {r['modality']:<14}  "
            f"Pearson r = {r['pearson_r']:.4f} (p = {r['pearson_p']:.3e})   "
            f"Spearman = {r['spearman_r']:.4f} (p = {r['spearman_p']:.3e})"
        )

    df.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

"""Record the inputs, seeds, and library versions behind each canonical run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from .config import MODEL, PROJECT_ROOT
from .hashing import sha256_file
from .report_schema import EXPECTED_AGENTS_PER_IMAGE, N_BOOT, SEED, WILCOXON_RESAMPLES
from .similarity import (
    EPS,
    MATCHED_FACTORIAL_CONTRAST_SEED_NAMESPACE,
    MATCHED_FACTORIAL_SEED_NAMESPACE,
    within_cross_cache_metadata_path,
)


def _write_method_manifest(
    path: Path,
    source_path: Path,
    source_metadata: dict[str, object],
    *,
    stats: pd.DataFrame,
    contrast: pd.DataFrame,
    complete_image_sensitivity: pd.DataFrame,
) -> None:
    """Record the choices needed to interpret and reproduce the CSVs."""
    manifest = {
        "model": MODEL,
        "source": {
            "path": str(source_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(source_path),
            "cache_metadata_path": str(
                within_cross_cache_metadata_path(source_path).relative_to(PROJECT_ROOT)
            ),
            "cache_schema_version": source_metadata["schema_version"],
            "empty_perception_policy": source_metadata["empty_perception_policy"],
            "required_upstream_policy": {
                "missing_embedding_id": (
                    "filter the complete annotation row before aligning embeddings"
                ),
                "empty_perception_response": (
                    "missing for perception pairs; retain valid caption and justification"
                ),
                "absent_generation": "not imputed",
            },
        },
        "analysis_unit": "image",
        "inference_scope": (
            "comparable new images, conditional on the fixed generated-agent panel"
        ),
        "within_cross_estimand": ("arithmetic mean over images of within_mean minus cross_mean"),
        "confidence_interval": {
            "estimand": "arithmetic mean paired difference",
            "method": "BCa bootstrap over images",
            "level": 0.95,
            "resamples": N_BOOT,
            "seed": SEED,
        },
        "wilcoxon": {
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "null": ("the paired-difference distribution is symmetric about zero"),
            "fallback_resamples": WILCOXON_RESAMPLES,
            "methods_used_within_cross": sorted(
                stats["wilcoxon_method"].astype(str).unique().tolist()
            ),
            "methods_used_contrast": sorted(
                contrast["wilcoxon_method"].astype(str).unique().tolist()
            ),
        },
        "sensitivity": {
            "method": "exact two-sided binomial sign test",
            "zeros": "excluded",
            "complete_image_panel": {
                "selection": "images with all intended generated agents",
                "expected_agents_per_image": EXPECTED_AGENTS_PER_IMAGE,
                "n_images": int(complete_image_sensitivity["n_images"].iloc[0]),
                "output": "within_cross_complete_image_sensitivity.csv",
            },
        },
        "multiplicity": {
            "method": "Benjamini-Hochberg",
            "within_cross_family": "12 tests within this model",
            "contrast_family": "4 justification-minus-caption tests within this model",
        },
        "contrast": {
            "target": "justification",
            "reference": "caption",
            "metric": "cosine similarity for both modalities",
            "excluded": (
                "perception-minus-caption because Jaccard and cosine effects "
                "are not on a common scale"
            ),
        },
        "exploratory_reference": {
            "epsilon": EPS,
            "status": "post hoc descriptive benchmark, not an equivalence margin",
        },
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _write_matched_factorial_manifest(
    path: Path,
    source_path: Path,
    *,
    source_metadata: dict[str, object],
    stats: pd.DataFrame,
    contrast: pd.DataFrame,
    matched_pairs_per_dimension: dict[str, int],
) -> None:
    """Record the choices needed to interpret and reproduce the matched CSVs.

    The exploratory epsilon reference is deliberately absent: it belongs to the
    marginal analysis only, and the paper describes matched effects by magnitude.

    Args:
        path: Destination JSON path.
        source_path: The per-image matched similarity cache the statistics came from.
        source_metadata: Provenance sidecar of that cache.
        stats: The canonical matched-factorial statistics frame.
        contrast: The canonical matched justification-minus-caption frame.
        matched_pairs_per_dimension: Profile-pair count behind each dimension.
    """
    manifest = {
        "model": MODEL,
        "source": {
            "path": str(source_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(source_path),
            "cache_metadata_path": str(
                within_cross_cache_metadata_path(source_path).relative_to(PROJECT_ROOT)
            ),
            "cache_schema_version": source_metadata["schema_version"],
            "empty_perception_policy": source_metadata["empty_perception_policy"],
            "construction": source_metadata["construction"],
        },
        "analysis_unit": "image",
        "status": "secondary; the marginal analysis in within_cross_significance.csv is primary",
        "matched_factorial_estimand": (
            "arithmetic mean over images of the matched profile-pair contrast "
            "(mean own-profile coherence of the two profiles minus their "
            "between-profile similarity), averaged over the profile pairs that "
            "differ only on the target dimension"
        ),
        "matched_pairs_per_dimension": matched_pairs_per_dimension,
        "unusable_pair_policy": (
            "a profile pair leaves both the within and cross terms together, so "
            "their difference equals the mean of the per-pair contrasts"
        ),
        "confidence_interval": {
            "estimand": "arithmetic mean paired difference",
            "method": "BCa bootstrap over images",
            "level": 0.95,
            "resamples": N_BOOT,
            "seed": SEED,
        },
        "wilcoxon": {
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "null": "the paired-difference distribution is symmetric about zero",
            "fallback_resamples": WILCOXON_RESAMPLES,
            "methods_used": sorted(stats["wilcoxon_method"].astype(str).unique().tolist()),
            "methods_used_contrast": sorted(
                contrast["wilcoxon_method"].astype(str).unique().tolist()
            ),
        },
        "sensitivity": {
            "method": "exact two-sided binomial sign test",
            "zeros": "excluded",
        },
        "multiplicity": {
            "method": "Benjamini-Hochberg",
            "matched_factorial_family": "12 tests within this model",
            "matched_factorial_contrast_family": (
                "4 justification-minus-caption tests within this model"
            ),
        },
        "contrast": {
            "estimand": (
                "arithmetic mean over images of the matched justification effect "
                "minus the matched caption effect"
            ),
            "target": "justification",
            "reference": "caption",
            "metric": "cosine similarity for both modalities",
            "excluded": (
                "perception-minus-caption because Jaccard and cosine effects "
                "are not on a common scale"
            ),
            "seed_namespace": MATCHED_FACTORIAL_CONTRAST_SEED_NAMESPACE,
            "output": "matched_factorial_modality_contrast.csv",
        },
        "seed_namespace": MATCHED_FACTORIAL_SEED_NAMESPACE,
        "ratio_reference": "within_cross_significance.csv",
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

"""Cosine and Jaccard similarity across persona agents.

The package is split by concern, but this module is the import surface every
caller and test uses. Moving a symbol between submodules is free; removing one
from here is a breaking change, because the paper-table generator and the
regression tests both import from ``src.similarity`` directly.
"""

from .cache_metadata import (
    EMPTY_TAG_POLICY,
    WITHIN_CROSS_CACHE_SCHEMA,
    validate_within_cross_cache_metadata,
    within_cross_cache_metadata_path,
    within_cross_input_metadata,
)
from .matched_factorial import (
    MATCHED_FACTORIAL_CONSTRUCTION,
    MATCHED_FACTORIAL_CONTRAST_SEED_NAMESPACE,
    MATCHED_FACTORIAL_FRAME_COLUMNS,
    MATCHED_FACTORIAL_SEED_NAMESPACE,
    compute_matched_factorial_similarity,
    matched_profile_pairs,
    persona_profiles,
)
from .paired_stats import (
    EPS,
    N_BOOT,
    N_WILCOXON_RESAMPLES,
    benjamini_hochberg,
    matched_pairs_rank_biserial,
    paired_stats,
    significance_stars,
    stable_seed_sequence,
    validate_paired_vectors,
)
from .pairwise import compute_per_image_similarity
from .perception import parse_perception_tags
from .profiles import compute_image_conditioned_profile_sim, compute_profile_coherence
from .reports import (
    ordered_cell,
    run_modality_contrast,
    run_within_cross_significance,
    validate_within_cross_frame,
)
from .within_cross import compute_within_cross_similarity, validate_similarity_inputs

__all__ = [
    "EMPTY_TAG_POLICY",
    "EPS",
    "MATCHED_FACTORIAL_CONSTRUCTION",
    "MATCHED_FACTORIAL_CONTRAST_SEED_NAMESPACE",
    "MATCHED_FACTORIAL_FRAME_COLUMNS",
    "MATCHED_FACTORIAL_SEED_NAMESPACE",
    "N_BOOT",
    "N_WILCOXON_RESAMPLES",
    "WITHIN_CROSS_CACHE_SCHEMA",
    "benjamini_hochberg",
    "compute_image_conditioned_profile_sim",
    "compute_matched_factorial_similarity",
    "compute_per_image_similarity",
    "compute_profile_coherence",
    "compute_within_cross_similarity",
    "matched_pairs_rank_biserial",
    "matched_profile_pairs",
    "ordered_cell",
    "paired_stats",
    "parse_perception_tags",
    "persona_profiles",
    "run_modality_contrast",
    "run_within_cross_significance",
    "significance_stars",
    "stable_seed_sequence",
    "validate_paired_vectors",
    "validate_similarity_inputs",
    "validate_within_cross_cache_metadata",
    "validate_within_cross_frame",
    "within_cross_cache_metadata_path",
    "within_cross_input_metadata",
]

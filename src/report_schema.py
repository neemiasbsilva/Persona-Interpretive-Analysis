"""Canonical column sets, inferential settings, and paper-table markers."""

from __future__ import annotations

from .schema import (
    DIM_DISPLAY_ORDER as DIM_ORDER,
)
from .schema import (
    DIM_LABELS,
    MODALITY_LABELS,
)
from .schema import (
    MODALITY_DISPLAY_ORDER as MODALITY_ORDER,
)

__all__ = [
    "COMBINED_TABLE_LABEL",
    "CONTRAST_COLUMNS",
    "DIM_LABELS",
    "DIM_ORDER",
    "EXPECTED_AGENTS_PER_IMAGE",
    "MATCHED_CONTRAST_COLUMNS",
    "MATCHED_CONTRAST_TABLE_BEGIN",
    "MATCHED_CONTRAST_TABLE_END",
    "MATCHED_CONTRAST_TABLE_LABEL",
    "MATCHED_FACTORIAL_COLUMNS",
    "MATCHED_TABLE_BEGIN",
    "MATCHED_TABLE_END",
    "MATCHED_TABLE_LABEL",
    "MODALITY_LABELS",
    "MODALITY_ORDER",
    "N_BOOT",
    "PAPER_MODELS",
    "RETIRED_TABLE_LABELS",
    "SEED",
    "SENSITIVITY_COLUMNS",
    "TABLE_BEGIN",
    "TABLE_END",
    "WILCOXON_RESAMPLES",
    "WITHIN_COLUMNS",
]


N_BOOT = 100_000
SEED = 42
WILCOXON_RESAMPLES = 100_000
EXPECTED_AGENTS_PER_IMAGE = 1_200
TABLE_BEGIN = "% BEGIN GENERATED PAIRED STATISTICS TABLE"
TABLE_END = "% END GENERATED PAIRED STATISTICS TABLE"
MATCHED_TABLE_BEGIN = "% BEGIN GENERATED MATCHED FACTORIAL TABLE"
MATCHED_TABLE_END = "% END GENERATED MATCHED FACTORIAL TABLE"
MATCHED_CONTRAST_TABLE_BEGIN = "% BEGIN GENERATED MATCHED FACTORIAL CONTRAST TABLE"
MATCHED_CONTRAST_TABLE_END = "% END GENERATED MATCHED FACTORIAL CONTRAST TABLE"
COMBINED_TABLE_LABEL = "tab:within_cross_qwenvl_gemma4"
MATCHED_TABLE_LABEL = "tab:matched_factorial"
MATCHED_CONTRAST_TABLE_LABEL = "tab:matched_factorial_contrast"
RETIRED_TABLE_LABELS = (
    "tab:within_cross_qwen",
    "tab:within_cross_gemma",
    "tab:modality_contrast",
    "tab:within_cross_combined",
)

PAPER_MODELS = {
    "qwen-vl": "Qwen3-VL",
    "gemma-4-E4B-it_t01": "Gemma4",
}

WITHIN_COLUMNS = [
    "dimension",
    "modality",
    "n_images",
    "within_mean",
    "cross_mean",
    "delta",
    "median_diff",
    "ci_lo",
    "ci_hi",
    "n_pos",
    "n_zero",
    "n_abs_ties",
    "w_stat",
    "p_wilcoxon",
    "p_wilcoxon_bh",
    "p_sign",
    "p_sign_bh",
    "rank_biserial",
    "wilcoxon_method",
    "ci_relation_to_exploratory_reference",
]

CONTRAST_COLUMNS = [
    "dimension",
    "target",
    "reference",
    "n_images",
    "mean_contrast",
    "median_contrast",
    "ci_lo",
    "ci_hi",
    "n_pos",
    "n_zero",
    "n_abs_ties",
    "w_stat",
    "p_wilcoxon",
    "p_wilcoxon_bh",
    "p_sign",
    "p_sign_bh",
    "rank_biserial",
    "wilcoxon_method",
]

SENSITIVITY_COLUMNS = [
    "selection",
    "expected_agents_per_image",
    *CONTRAST_COLUMNS,
]

MATCHED_FACTORIAL_COLUMNS = [
    "dimension",
    "modality",
    "n_images",
    "n_matched_pairs",
    "within_mean",
    "cross_mean",
    "delta",
    "median_diff",
    "ci_lo",
    "ci_hi",
    "n_pos",
    "n_zero",
    "n_abs_ties",
    "w_stat",
    "p_wilcoxon",
    "p_wilcoxon_bh",
    "p_sign",
    "p_sign_bh",
    "rank_biserial",
    "wilcoxon_method",
    "marginal_delta",
    "matched_to_marginal_ratio",
]

MATCHED_CONTRAST_COLUMNS = [
    "dimension",
    "target",
    "reference",
    "n_images",
    "n_matched_pairs",
    "mean_contrast",
    "median_contrast",
    "ci_lo",
    "ci_hi",
    "n_pos",
    "n_zero",
    "n_abs_ties",
    "w_stat",
    "p_wilcoxon",
    "p_wilcoxon_bh",
    "p_sign",
    "p_sign_bh",
    "rank_biserial",
    "wilcoxon_method",
]

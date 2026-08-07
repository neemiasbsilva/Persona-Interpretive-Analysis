#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
08_matched_factorial.sh — Matched-factorial within/cross contrast (Appendix F).

Secondary analysis. Where 02_paired_statistics.sh buckets annotation pairs on
one attribute while the other three vary freely, this contrasts pairs of full
persona profiles that agree on the other three attributes and differ only on
the target one, which isolates that attribute's contribution.

It also contrasts the two cosine modalities against each other inside the
matched design, Gamma = matched justification effect minus matched caption
effect, which is the direct test of the paper's descriptive-versus-interpretive
claim under a controlled factorial comparison.

The inference is unchanged: two-sided Wilcoxon signed-rank tests on the 50
image-level differences, mean effects with 95% BCa bootstrap intervals,
matched-pairs rank-biserial correlations, exact sign-test sensitivity checks,
and Benjamini-Hochberg correction within each model's 12 within/cross
comparisons and, separately, within its 4 justification-minus-caption contrasts.

Usage:
  ./scripts/08_matched_factorial.sh <model> [--check-paper]

  <model>         qwen-vl (default) or gemma-4-E4B-it_t01
  --check-paper   fail if either generated table and its marked block in
                  paper.tex disagree. Only meaningful once BOTH models
                  have been run.

Produces (under outputs/<model>/):
  matched_factorial_persona_sim.csv        per-image matched within/cross means
  matched_factorial_significance.csv       12 matched-factorial tests
  matched_factorial_modality_contrast.csv  4 matched justification-minus-caption tests
  matched_factorial_method.json            input digest + inferential settings
and, once both models exist:
  outputs/matched_factorial_table.tex           the Appendix F effect table
  outputs/matched_factorial_contrast_table.tex  the Appendix F contrast table

Requires 01_pipeline.sh for the embeddings and 02_paired_statistics.sh for the
marginal effects the reported ratio is taken against.
EOF
}


MODEL="${PAPER_MODELS[0]}"
CHECK=()
for arg in "$@"; do
    case "$arg" in
        -h|--help)     usage; exit 0 ;;
        --check-paper) CHECK=(--check-paper) ;;
        --*)           die "Unknown flag: $arg" ;;
        *)             MODEL="$arg" ;;
    esac
done

require_uv
init_log "$MODEL"
log "Matched-factorial contrast for $MODEL"

for cache in caption_embeddings.npy justification_embeddings.npy; do
    [[ -f "outputs/$MODEL/$cache" ]] \
        || die "outputs/$MODEL/$cache not found. Run ./scripts/01_pipeline.sh $MODEL first."
done
[[ -f "outputs/$MODEL/within_cross_significance.csv" ]] \
    || die "outputs/$MODEL/within_cross_significance.csv not found. Run ./scripts/02_paired_statistics.sh $MODEL first."

PERSONA_MODEL="$MODEL" run_step "src.matched_factorial_report" \
    env PERSONA_MODEL="$MODEL" uv run python -m src.matched_factorial_report "${CHECK[@]}"

list_artifacts \
    "outputs/$MODEL/matched_factorial_persona_sim.csv" \
    "outputs/$MODEL/matched_factorial_significance.csv" \
    "outputs/$MODEL/matched_factorial_modality_contrast.csv" \
    "outputs/$MODEL/matched_factorial_method.json" \
    "outputs/matched_factorial_table.tex" \
    "outputs/matched_factorial_contrast_table.tex"

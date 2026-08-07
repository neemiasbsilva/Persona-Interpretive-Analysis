#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
02_paired_statistics.sh — Canonical paired image-level statistics for the paper
(Section 3.3, Section 4.2, Appendix E).

Runs the two-sided Wilcoxon signed-rank tests on the 50 image-level differences
per dimension and modality, together with mean effects, 95% BCa bootstrap
intervals, matched-pairs rank-biserial correlations, exact sign-test sensitivity
checks, and Benjamini-Hochberg correction within each test family.

Usage:
  ./scripts/02_paired_statistics.sh <model> [--check-paper]

  <model>         qwen-vl (default) or gemma-4-E4B-it_t01
  --check-paper   fail if the generated table and the marked block in paper.tex
                  disagree. Only meaningful once BOTH models have been run.

Produces (under outputs/<model>/):
  within_cross_significance.csv               12 within-minus-cross tests
  within_cross_modality_contrast.csv          4 justification-minus-caption contrasts
  within_cross_complete_image_sensitivity.csv contrasts on fully-generated images only
  within_cross_method.json                    input digest + inferential settings
and, once both models exist:
  outputs/within_cross_combined_table.tex     the Appendix E table (Panels A and B)

Requires 01_pipeline.sh to have produced outputs/<model>/within_cross_persona_sim.csv.
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
log "Paired statistics for $MODEL"

[[ -f "outputs/$MODEL/within_cross_persona_sim.csv" ]] \
    || die "outputs/$MODEL/within_cross_persona_sim.csv not found. Run ./scripts/01_pipeline.sh $MODEL first."

PERSONA_MODEL="$MODEL" run_step "src.similarity_report" \
    env PERSONA_MODEL="$MODEL" uv run python -m src.similarity_report "${CHECK[@]}"

list_artifacts \
    "outputs/$MODEL/within_cross_significance.csv" \
    "outputs/$MODEL/within_cross_modality_contrast.csv" \
    "outputs/$MODEL/within_cross_complete_image_sensitivity.csv" \
    "outputs/$MODEL/within_cross_method.json" \
    "outputs/within_cross_combined_table.tex"

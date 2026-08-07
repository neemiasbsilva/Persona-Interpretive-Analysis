#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
05_summary_figures.sh — Cross-model summary figures and annotated backup matrices.

Figures A and B compare the two paper models side by side (persona-effect summary
and topic contrasts) and therefore need both models' outputs. Figure C is the
per-model annotated "reader's guide" version of the inter-profile matrices; it is
also produced by 01_pipeline.sh via --annotated-only.

Usage:
  ./scripts/05_summary_figures.sh [--model-a <model>] [--model-b <model>]
                                  [--annotated-only <model>]

  defaults: --model-a qwen-vl  --model-b gemma-4-E4B-it_t01

Produces:
  figures/fig_persona_effect_summary.{pdf,png}
  figures/fig_topic_contrasts.{pdf,png}
  figures/<model>/fig_*_annotated.{pdf,png}
EOF
}


MODEL_A="${PAPER_MODELS[0]}"
MODEL_B="${PAPER_MODELS[1]}"
ANNOTATED_ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)         usage; exit 0 ;;
        --model-a)         MODEL_A="$2"; shift 2 ;;
        --model-b)         MODEL_B="$2"; shift 2 ;;
        --annotated-only)  ANNOTATED_ONLY="$2"; shift 2 ;;
        *)                 die "Unknown argument: $1" ;;
    esac
done

require_uv

if [[ -n "$ANNOTATED_ONLY" ]]; then
    init_log "$ANNOTATED_ONLY"
    log "Annotated matrices for $ANNOTATED_ONLY"
    run_step "src.summary_figures --annotated-only" \
        uv run python -m src.summary_figures --annotated-only "$ANNOTATED_ONLY"
    list_artifacts "figures/$ANNOTATED_ONLY"
    exit 0
fi

init_log "$MODEL_A"
log "Summary figures: $MODEL_A vs $MODEL_B"

for model in "$MODEL_A" "$MODEL_B"; do
    [[ -f "outputs/$model/within_cross_significance.csv" ]] \
        || die "outputs/$model/within_cross_significance.csv not found. Run ./scripts/02_paired_statistics.sh $model first."
done

run_step "src.summary_figures" \
    uv run python -m src.summary_figures --model-a "$MODEL_A" --model-b "$MODEL_B"

list_artifacts \
    "figures/fig_persona_effect_summary.pdf" \
    "figures/fig_topic_contrasts.pdf"

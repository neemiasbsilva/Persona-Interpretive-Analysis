#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
04_cross_model_correlation.sh — Cross-model agreement between the inter-profile
similarity matrices of the two paper models (Table 2).

Correlates the upper triangles of the image-conditioned 24x24 matrices for
captions, justifications, and perception tags, reporting Pearson r and Spearman rho.

Usage:
  ./scripts/04_cross_model_correlation.sh [--model-a <model>] [--model-b <model>]
                                          [--exclude-diagonal]

  defaults: --model-a qwen-vl  --model-b gemma-4-E4B-it_t01

Produces:
  outputs/cross_model_correlation.csv   one row per modality (r, rho, p-values, n)

Requires 01_pipeline.sh for both models (it writes the ic_profile_sim_*.npy caches).
EOF
}


MODEL_A="${PAPER_MODELS[0]}"
MODEL_B="${PAPER_MODELS[1]}"
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)           usage; exit 0 ;;
        --model-a)           MODEL_A="$2"; shift 2 ;;
        --model-b)           MODEL_B="$2"; shift 2 ;;
        --exclude-diagonal)  EXTRA+=(--exclude-diagonal); shift ;;
        *)                   die "Unknown argument: $1" ;;
    esac
done

require_uv
init_log "$MODEL_A"
log "Cross-model correlation: $MODEL_A vs $MODEL_B"

for model in "$MODEL_A" "$MODEL_B"; do
    [[ -f "outputs/$model/ic_profile_sim_caption.npy" ]] \
        || die "outputs/$model/ic_profile_sim_caption.npy not found. Run ./scripts/01_pipeline.sh $model first."
done

run_step "src.cross_model_correlation" \
    uv run python -m src.cross_model_correlation \
        --model-a "$MODEL_A" --model-b "$MODEL_B" "${EXTRA[@]}"

list_artifacts "outputs/cross_model_correlation.csv"

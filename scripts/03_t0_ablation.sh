#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
03_t0_ablation.sh — Decoding-sensitivity ablation at temperature T=0 (Appendix D).

Varies economic status alone (other attributes fixed to Female, Conservative,
Analytical) and measures per-image cross-level similarity for captions and
justifications (cosine) and perception tags (Jaccard), under deterministic decoding.

Usage:
  ./scripts/03_t0_ablation.sh [--model-a <model>] [--model-b <model>]

  defaults: --model-a qwen-vl-t0  --model-b gemma4-t0

Produces:
  outputs/<t0-model>/t0_econ_summary.csv   per-modality within/cross means
  outputs/<t0-model>/t0_within_cross.csv   per-image cross-level similarities
  outputs/t0_ablation_table.tex            Appendix D table
  figures/fig_t0_ablation.{pdf,png}        Appendix D figure
EOF
}


MODEL_A="${T0_MODELS[0]}"
MODEL_B="${T0_MODELS[1]}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)  usage; exit 0 ;;
        --model-a)  MODEL_A="$2"; shift 2 ;;
        --model-b)  MODEL_B="$2"; shift 2 ;;
        *)          die "Unknown argument: $1" ;;
    esac
done

require_uv
init_log "$MODEL_A"
log "T=0 ablation: $MODEL_A and $MODEL_B"

for model in "$MODEL_A" "$MODEL_B"; do
    require_corpus "$model"
    run_step "src.t0_ablation ($model)" \
        env PERSONA_MODEL="$model" uv run python -m src.t0_ablation
done

run_step "src.t0_ablation --summary" \
    uv run python -m src.t0_ablation --summary --model-a "$MODEL_A" --model-b "$MODEL_B"

list_artifacts \
    "outputs/$MODEL_A/t0_econ_summary.csv" \
    "outputs/$MODEL_B/t0_econ_summary.csv" \
    "outputs/t0_ablation_table.tex" \
    "figures/fig_t0_ablation.pdf"

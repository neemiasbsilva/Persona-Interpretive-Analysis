#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
01_pipeline.sh — Per-model analysis pipeline: embeddings, per-image similarity,
BERTopic topics, and the per-model figures used in the paper.

Usage:
  ./scripts/01_pipeline.sh <model> [--force]

  <model>   corpus under data/<model>/ (default: qwen-vl).
            Paper models: qwen-vl, gemma-4-E4B-it_t01
  --force   clear the regenerable caches first and recompute from scratch.

Produces (under outputs/<model>/ and figures/<model>/):
  caption/justification embeddings, per-image + within/cross similarity caches,
  BERTopic topic assignments, inter-profile matrices, topic and no-persona figures.

This is a thin wrapper around ./run_experiments.sh so that every experiment in
the paper has one entry point under scripts/. Run it before 02_paired_statistics.sh.
EOF
}


MODEL="${PAPER_MODELS[0]}"
FORCE=()
for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --force)   FORCE=(--force) ;;
        --*)       die "Unknown flag: $arg" ;;
        *)         MODEL="$arg" ;;
    esac
done

require_uv
require_corpus "$MODEL"

./run_experiments.sh "$MODEL" "${FORCE[@]}"

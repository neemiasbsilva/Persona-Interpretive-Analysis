#!/usr/bin/env bash
usage() {
    cat <<'EOF'
run_experiments.sh - Generate every artifact the paper reports.

Usage:
  ./run_experiments.sh <model>              one model: caches, figures, statistics
  ./run_experiments.sh <model> --force      clear caches first and recompute
  ./run_experiments.sh                      defaults to the first paper model
  ./run_experiments.sh --all                every paper model plus the cross-model
                                            steps, the paper-table check, and tests
  ./run_experiments.sh --all --skip-pipeline
                                            reuse cached embeddings, similarities and
                                            topics; redo only statistics, figures,
                                            and checks

<model> selects the corpus under data/<model>/ and namespaces every cached
artifact (outputs/<model>/) and figure (figures/<model>/). It is exported as
PERSONA_MODEL so src/config.py resolves all paths for that model.

--all runs, in dependency order:
  1. this script for each paper model   embeddings, similarity, topics, figures,
                                        and the paired Wilcoxon statistics
  2. matched-factorial contrast         Appendix F, needs step 1 for each model
  3. cross-model correlation            Table 2
  4. T=0 ablation                       Appendix D, skipped if the corpora are absent
  5. cross-model summary figures        needs both models
  6. paper-table checks                 needs both models' reports
  7. regression tests

Outputs feeding paper.tex:
  outputs/within_cross_combined_table.tex   Appendix E table
  outputs/matched_factorial_table.tex       Appendix F effect table
  outputs/matched_factorial_contrast_table.tex  Appendix F contrast table
  outputs/cross_model_correlation.csv       Table 2
  outputs/<model>/within_cross_significance.csv
  outputs/<model>/matched_factorial_significance.csv
  outputs/t0_ablation_table.tex             Appendix D table
  figures/<model>/                          per-model figures

Requires: uv (https://github.com/astral-sh/uv)
Re-running without --force skips cached steps automatically.
EOF
}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

read -ra ALL_PAPER_MODELS < <(uv run python -m src.models paper)
read -ra ALL_T0_MODELS < <(uv run python -m src.models t0)

MODEL="${ALL_PAPER_MODELS[0]}"
FORCE=0
RUN_ALL=0
SKIP_PIPELINE=0
for arg in "$@"; do
    case "$arg" in
        -h|--help)       usage; exit 0 ;;
        --force)         FORCE=1 ;;
        --all)           RUN_ALL=1 ;;
        --skip-pipeline) SKIP_PIPELINE=1 ;;
        --*)             echo "Unknown flag: $arg" >&2; exit 1 ;;
        *)               MODEL="$arg" ;;
    esac
done

if [[ $RUN_ALL -eq 1 ]]; then
    stamp() { echo "[$(date '+%H:%M:%S')] $*"; }
    fail()  { echo "ERROR: $*" >&2; exit 1; }

    stamp "Full run for: ${ALL_PAPER_MODELS[*]}"

    if [[ $SKIP_PIPELINE -eq 1 ]]; then
        stamp "[1/7] Skipping per-model pipeline (--skip-pipeline)"
        for model in "${ALL_PAPER_MODELS[@]}"; do
            stamp "      Refreshing statistics only: $model"
            PERSONA_MODEL="$model" ./scripts/02_paired_statistics.sh "$model" \
                || fail "paired statistics failed for $model"
        done
    else
        for model in "${ALL_PAPER_MODELS[@]}"; do
            stamp "[1/7] Pipeline: $model"
            FORCE_ARGS=()
            [[ $FORCE -eq 1 ]] && FORCE_ARGS=(--force)
            "$0" "$model" "${FORCE_ARGS[@]}" || fail "pipeline failed for $model"
        done
    fi

    for model in "${ALL_PAPER_MODELS[@]}"; do
        stamp "[2/7] Matched-factorial contrast: $model"
        ./scripts/08_matched_factorial.sh "$model" \
            || fail "matched-factorial contrast failed for $model"
    done

    stamp "[3/7] Cross-model correlation"
    ./scripts/04_cross_model_correlation.sh || fail "cross-model correlation failed"

    stamp "[4/7] T=0 ablation"
    if [[ -f "data/${ALL_T0_MODELS[0]}/annotations_baseline.jsonl" ]]; then
        ./scripts/03_t0_ablation.sh || fail "T=0 ablation failed"
    else
        stamp "      SKIPPED: data/${ALL_T0_MODELS[0]}/annotations_baseline.jsonl not present (T=0 corpora are optional; ./scripts/09_hub.sh pull --groups corpora)"
    fi

    stamp "[5/7] Summary figures"
    ./scripts/05_summary_figures.sh || fail "summary figures failed"

    stamp "[6/7] Verifying the generated tables against paper.tex"
    ./scripts/02_paired_statistics.sh "${ALL_PAPER_MODELS[0]}" --check-paper \
        || fail "the marked Appendix E table in paper.tex does not match the canonical CSVs"
    ./scripts/08_matched_factorial.sh "${ALL_PAPER_MODELS[0]}" --check-paper \
        || fail "the marked Appendix F table in paper.tex does not match the canonical CSVs"

    stamp "[7/7] Regression tests"
    ./scripts/06_tests.sh -q || fail "tests failed"

    stamp ""
    stamp "Complete. Artifacts for paper.tex:"
    stamp "  outputs/within_cross_combined_table.tex   Appendix E table"
    stamp "  outputs/matched_factorial_table.tex       Appendix F effect table"
    stamp "  outputs/matched_factorial_contrast_table.tex  Appendix F contrast table"
    stamp "  outputs/cross_model_correlation.csv       Table 2"
    stamp "  outputs/*/within_cross_significance.csv   Section 4.2 effects"
    stamp "  outputs/t0_ablation_table.tex             Appendix D table"
    stamp "  figures/<model>/                          per-model figures"
    exit 0
fi

export PERSONA_MODEL="$MODEL"

NB_DIR="notebooks"
OUT_DIR="outputs/$MODEL"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR" "figures/$MODEL"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOGFILE="$LOG_DIR/run_${TIMESTAMP}.log"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGFILE"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

log "Model: $MODEL"
[[ -f "data/$MODEL/annotations_baseline.jsonl" ]] \
    || die "data/$MODEL/annotations_baseline.jsonl not found. Run ./scripts/09_hub.sh pull --groups corpora, or copy the corpus into data/$MODEL/."

if [[ $FORCE -eq 1 ]]; then
    log "WARNING: --force set. Removing embedding, similarity, and topic caches for $MODEL..."
    rm -f "$OUT_DIR"/caption_embeddings.npy "$OUT_DIR"/justification_embeddings.npy
    rm -f "$OUT_DIR"/caption_embeddings_ids.csv
    rm -f "$OUT_DIR"/np_caption_embeddings.npy "$OUT_DIR"/np_just_embeddings.npy
    rm -f "$OUT_DIR"/np_nothink_caption_embeddings.npy "$OUT_DIR"/np_nothink_just_embeddings.npy
    rm -f "$OUT_DIR"/per_image_cosine_similarity.csv "$OUT_DIR"/per_image_just_similarity.csv
    rm -f "$OUT_DIR"/per_persona_cosine_similarity.csv
    rm -f "$OUT_DIR"/within_cross_persona_sim.csv
    rm -f "$OUT_DIR"/within_cross_persona_sim.csv.meta.json
    rm -f "$OUT_DIR"/within_profile_coherence.csv
    rm -f "$OUT_DIR"/within_cross_significance.csv
    rm -f "$OUT_DIR"/within_cross_modality_contrast.csv
    rm -f "$OUT_DIR"/within_cross_complete_image_sensitivity.csv
    rm -f "$OUT_DIR"/within_cross_method.json
    rm -f "$OUT_DIR"/matched_factorial_persona_sim.csv
    rm -f "$OUT_DIR"/matched_factorial_persona_sim.csv.meta.json
    rm -f "$OUT_DIR"/matched_factorial_significance.csv
    rm -f "$OUT_DIR"/matched_factorial_modality_contrast.csv
    rm -f "$OUT_DIR"/matched_factorial_method.json
    rm -f "$OUT_DIR"/profile_sim_labels.csv
    rm -f "$OUT_DIR"/ic_profile_sim_caption.npy "$OUT_DIR"/ic_profile_sim_just.npy
    rm -f "$OUT_DIR"/ic_profile_sim_jaccard.npy "$OUT_DIR"/ic_profile_sim_labels.csv
    rm -f "$OUT_DIR"/bertopic_captions.pkl
    rm -f "$OUT_DIR"/annotations_with_topics.csv
    rm -f "$OUT_DIR"/caption_topic_info.csv "$OUT_DIR"/just_topic_info.csv
    rm -f "$OUT_DIR"/topic_labels_just.csv
    log "Caches cleared."
fi

log "Checking dependencies..."
command -v uv >/dev/null 2>&1 || die "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
uv run jupyter --version >/dev/null 2>&1 || die "jupyter not found in uv env. Install: uv pip install jupyter nbconvert"
uv run python -c "import src.config" 2>/dev/null || die "src/ package not importable. Run from project root."

NOTEBOOKS_PRE=(
    "01_eda_captions_justifications.ipynb"
    "03_text_similarity_captions.ipynb"
    "02_sentiment_analysis_justifications.ipynb"
    "04_topic_modeling_bertopic.ipynb"
)
NOTEBOOKS_POST=(
    "05_convergence_analysis.ipynb"
)

run_nb() {
    local nb="$1"
    log "Running $nb ..."
    uv run jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=3600 \
        --ExecutePreprocessor.kernel_name=python3 \
        "$NB_DIR/$nb" \
        >> "$LOGFILE" 2>&1 \
        && log "  DONE: $nb" \
        || { log "  FAILED: $nb — check $LOGFILE"; exit 1; }
    uv run jupyter nbconvert \
        --to notebook \
        --inplace \
        --ClearOutputPreprocessor.enabled=True \
        "$NB_DIR/$nb" \
        >> "$LOGFILE" 2>&1
}

for nb in "${NOTEBOOKS_PRE[@]}"; do run_nb "$nb"; done

log "Generating paired Wilcoxon statistics (src/similarity_report.py) ..."
uv run python -m src.similarity_report >> "$LOGFILE" 2>&1 \
    && log "  DONE: src/similarity_report.py" \
    || { log "  FAILED: src/similarity_report.py — check $LOGFILE"; exit 1; }

log "Generating standalone figures (src/generate_figures.py) ..."
uv run python -m src.generate_figures >> "$LOGFILE" 2>&1 \
    && log "  DONE: src/generate_figures.py" \
    || { log "  FAILED: src/generate_figures.py — check $LOGFILE"; exit 1; }

log "Generating annotated backup matrices (src/summary_figures.py) ..."
uv run python -m src.summary_figures --annotated-only "$MODEL" >> "$LOGFILE" 2>&1 \
    && log "  DONE: annotated matrices" \
    || { log "  FAILED: src/summary_figures.py — check $LOGFILE"; exit 1; }

for nb in "${NOTEBOOKS_POST[@]}"; do run_nb "$nb"; done

log ""
log "Pipeline complete for $MODEL. Figures in figures/$MODEL/:"
ls "figures/$MODEL"/fig_*.pdf 2>/dev/null | sed 's/^/  /' | tee -a "$LOGFILE" || true
log ""
log "Full log: $LOGFILE"

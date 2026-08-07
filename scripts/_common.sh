#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

die() { echo "ERROR: $*" >&2; exit 1; }

command -v uv >/dev/null 2>&1 \
    || die "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"

read -ra PAPER_MODELS < <(uv run python -m src.models paper) \
    || die "Could not read the model list from src.models"
read -ra T0_MODELS < <(uv run python -m src.models t0) \
    || die "Could not read the model list from src.models"

LOGFILE=""
init_log() {
    local scope="$1"
    local log_dir="outputs/$scope/logs"
    mkdir -p "$log_dir"
    LOGFILE="$log_dir/$(basename "$0" .sh)_$(date '+%Y%m%d_%H%M%S').log"
}

log() {
    if [[ -n "$LOGFILE" ]]; then
        echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGFILE"
    else
        echo "[$(date '+%H:%M:%S')] $*"
    fi
}

require_uv() {
    uv run python -c "import src.config" >/dev/null 2>&1 \
        || die "src/ package not importable. Run from the repository root."
}

require_corpus() {
    local model="$1"
    [[ -f "data/$model/annotations_baseline.jsonl" ]] \
        || die "data/$model/annotations_baseline.jsonl not found. See README (Downloading the datasets)."
}

run_step() {
    local desc="$1"; shift
    log "$desc"
    if [[ -n "$LOGFILE" ]]; then
        "$@" >>"$LOGFILE" 2>&1 || die "$desc failed — see $LOGFILE"
    else
        "$@" || die "$desc failed"
    fi
    log "  DONE: $desc"
}

list_artifacts() {
    log ""
    log "Artifacts:"
    local path
    for path in "$@"; do
        [[ -e "$path" ]] && log "  $path"
    done
    [[ -n "$LOGFILE" ]] && log "" && log "Full log: $LOGFILE"
    return 0
}

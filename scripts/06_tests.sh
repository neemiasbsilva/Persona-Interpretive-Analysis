#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
06_tests.sh — Regression tests over the statistical artifacts.

tests/unit/         — the paired Wilcoxon / BCa / BH machinery, the seed
                      derivation, the matched-factorial construction, and both
                      paper.tex table contracts against committed golden fixtures
tests/integration/  — the generated CSVs and both marked paper.tex table blocks
                      still agree with the code; auto-skipped without the corpora

Usage:
  ./scripts/06_tests.sh [pytest args...]

Example:
  ./scripts/06_tests.sh -k paper -vv
EOF
}


case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

require_uv
uv run pytest tests/ "$@"

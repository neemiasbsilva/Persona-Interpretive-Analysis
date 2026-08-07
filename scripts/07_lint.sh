#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
07_lint.sh - Run every static check over the whole repository.

Usage:
  ./scripts/07_lint.sh [--fix]

  --fix   Apply ruff's safe autofixes and reformat before reporting.

Runs, in order:
  ruff format   formatting
  ruff check    lint
  check_style   no comments, no file over 500 lines
  mypy          strict type checking

Exits non-zero if any check reports a finding. This is the full-tree entry point
for every static check, and is what an agent should invoke, since agents do not
run git commit here.
EOF
}

FIX=0
for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --fix)     FIX=1 ;;
        *)         die "Unknown argument: $arg" ;;
    esac
done

require_uv

STATUS=0
report() {
    local label="$1" code="$2"
    if [[ "$code" -eq 0 ]]; then
        echo "  PASS  $label"
    else
        echo "  FAIL  $label"
        STATUS=1
    fi
}

if [[ $FIX -eq 1 ]]; then
    echo "Applying ruff autofixes..."
    uv run ruff check --fix || true
    uv run ruff format || true
    echo ""
fi

echo "Static checks:"

uv run ruff format --check >/tmp/lint_fmt.$$ 2>&1
report "ruff format" $?

uv run ruff check >/tmp/lint_ruff.$$ 2>&1
report "ruff check" $?

uv run python tools/check_style.py --all >/tmp/lint_style.$$ 2>&1
report "check_style" $?

uv run mypy >/tmp/lint_mypy.$$ 2>&1
report "mypy" $?

echo ""
for name in fmt ruff style mypy; do
    file="/tmp/lint_${name}.$$"
    if [[ -s "$file" ]]; then
        echo "--- $name ---"
        tail -40 "$file"
        echo ""
    fi
    rm -f "$file"
done

if [[ $STATUS -ne 0 ]]; then
    echo "Lint failed. Re-run with --fix to apply the mechanical fixes."
fi
exit "$STATUS"

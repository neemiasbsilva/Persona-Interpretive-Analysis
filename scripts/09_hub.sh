#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
09_hub.sh — Publish the corpora, outputs and figures to the Hugging Face Hub, or pull
them back into the working copy.

Dataset: https://huggingface.co/datasets/MInDS-lab-UTFPR/UrbanPersona-120K-Interpretive

Usage:
  ./scripts/09_hub.sh status
  ./scripts/09_hub.sh build
  ./scripts/09_hub.sh push [--dry-run] [--force] [--prune]
  ./scripts/09_hub.sh pull [--groups corpora outputs embeddings figures card]
                           [--overwrite] [--revision <rev>]

  status   token, remote repository, staging tree and recorded uploads
  build    link data/, outputs/ and figures/ into hub/datasets/ and write the card
  push     build, then upload every changed chunk (one commit each) and the
           top-level files; uploads only add or replace files, --prune also
           deletes the remote files missing here after listing them, and
           --dry-run reports the plan (offline and without a token unless
           --prune has to list the remote)
  pull     download into hub/downloads/, then copy every file the working copy
           is missing; --overwrite also replaces files whose bytes differ

Options pass through to python -m src.hub.commands; put --help after an action
for its full list, e.g. ./scripts/09_hub.sh pull --help.

Produces:
  hub/datasets/            staging tree of file symlinks, plus the dataset card
  hub/upload_state.json    digest of every uploaded chunk and top-level file
  hub/downloads/           what pull downloaded, before it is copied into place

hub/ is gitignored. Publishing needs a write-scoped token (uv run hf auth login).
The dataset stays private until the paper is published, so a pull needs a token
with read access to MInDS-lab-UTFPR until then.
EOF
}


case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "")        usage; exit 1 ;;
esac

require_uv
uv run python -m src.hub.commands "$@"

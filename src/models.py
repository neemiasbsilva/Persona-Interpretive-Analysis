"""Canonical model identifiers and the per-model artifact layout.

Single source of truth for the model names used by the paper. The shell harness
reads these through ``python -m src.models`` so ``scripts/_common.sh`` never
hardcodes them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAPER_MODELS: tuple[str, ...] = ("qwen-vl", "gemma-4-E4B-it_t01")
T0_MODELS: tuple[str, ...] = ("qwen-vl-t0", "gemma4-t0")

DEFAULT_MODEL = PAPER_MODELS[0]

_DISPLAY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("qwen", "Qwen3-VL"),
    ("gemma", "Gemma4"),
)


def display_name(model: str) -> str:
    """Return the human-facing label for a directory-style model id.

    Args:
        model: Model identifier such as ``"gemma-4-E4B-it_t01"``.

    Returns:
        Paper-facing label, or the identifier unchanged when unrecognised.
    """
    lowered = model.lower()
    for prefix, label in _DISPLAY_PREFIXES:
        if lowered.startswith(prefix):
            return label
    return model


def outputs_dir(model: str, project_root: Path = PROJECT_ROOT) -> Path:
    """Return the cached-artifact directory for a model.

    Args:
        model: Model identifier.
        project_root: Repository root.

    Returns:
        Path to ``outputs/<model>``.
    """
    return project_root / "outputs" / model


def figures_dir(model: str, project_root: Path = PROJECT_ROOT) -> Path:
    """Return the figure directory for a model.

    Args:
        model: Model identifier.
        project_root: Repository root.

    Returns:
        Path to ``figures/<model>``.
    """
    return project_root / "figures" / model


def data_dir(model: str, project_root: Path = PROJECT_ROOT) -> Path:
    """Return the corpus directory for a model.

    Args:
        model: Model identifier.
        project_root: Repository root.

    Returns:
        Path to ``data/<model>``.
    """
    return project_root / "data" / model


def annotations_path(model: str, project_root: Path = PROJECT_ROOT) -> Path:
    """Return the baseline annotation corpus for a model.

    Args:
        model: Model identifier.
        project_root: Repository root.

    Returns:
        Path to ``data/<model>/annotations_baseline.jsonl``.
    """
    return data_dir(model, project_root) / "annotations_baseline.jsonl"


def main() -> None:
    """Print a model list as whitespace-separated names for the shell harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=("paper", "t0", "all"), nargs="?", default="paper")
    args = parser.parse_args()
    groups = {"paper": PAPER_MODELS, "t0": T0_MODELS, "all": PAPER_MODELS + T0_MODELS}
    print(" ".join(groups[args.group]))


if __name__ == "__main__":
    main()

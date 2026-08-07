"""Shared fixtures and collection rules for the test suite."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pytest

from src.models import PAPER_MODELS

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "golden"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root.

    Returns:
        Absolute path to the repository root.
    """
    return ROOT


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """Return the directory holding committed regression fixtures.

    Returns:
        Absolute path to ``tests/golden``.
    """
    return GOLDEN


@pytest.fixture(scope="session")
def frozen_seed_entropy() -> dict[str, list[int]]:
    """Return the recorded seed entropy for every statistical cell.

    Returns:
        Mapping of pipe-joined cell key to the expected entropy list.
    """
    payload = (GOLDEN / "frozen_seed_entropy.json").read_text(encoding="utf-8")
    return json.loads(payload)


def _missing_integration_inputs() -> str | None:
    """Report the first required artifact that is absent for integration tests.

    Returns:
        A human-readable reason to skip, or ``None`` when inputs are present.
    """
    required = [ROOT / "paper.tex"]
    for model in PAPER_MODELS:
        required.append(ROOT / "data" / model / "annotations_baseline.jsonl")
        required.append(ROOT / "outputs" / model / "within_cross_significance.csv")
        required.append(ROOT / "outputs" / model / "matched_factorial_significance.csv")
        required.append(ROOT / "outputs" / model / "matched_factorial_modality_contrast.csv")

    for path in required:
        if not path.exists():
            return f"missing {path.relative_to(ROOT)} (see README: Downloading the datasets)"
    return None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip integration tests when their corpora or generated outputs are absent.

    Keeps a bare clone green while still running the full suite on a machine
    that has the datasets.

    Args:
        items: Collected test items, modified in place.
    """
    reason: str | None = None
    checked = False
    for item in items:
        if item.get_closest_marker("integration") is None:
            continue
        if not checked:
            reason = _missing_integration_inputs()
            checked = True
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))

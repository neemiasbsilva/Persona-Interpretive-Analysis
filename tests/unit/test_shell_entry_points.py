"""Check that every shell entry point documents itself through --help.

Help text used to be produced by printing the script's own comment header with
``sed``. That broke silently when the comments were removed, so these tests
assert on the *content* of the output, not just the exit status.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.models import PAPER_MODELS, T0_MODELS

ROOT_SCRIPTS = ("run_experiments.sh",)


def _entry_points(repo_root: Path) -> list[Path]:
    scripts = sorted(p for p in (repo_root / "scripts").glob("*.sh") if p.name != "_common.sh")
    return scripts + [repo_root / name for name in ROOT_SCRIPTS]


def _help_text(script: Path) -> str:
    result = subprocess.run(
        ["/usr/bin/env", "bash", str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"{script.name} --help exited {result.returncode}"
    return result.stdout


def test_every_entry_point_is_discovered(repo_root):
    names = {p.name for p in _entry_points(repo_root)}
    assert "run_experiments.sh" in names
    assert "07_lint.sh" in names
    assert len(names) >= 8


def test_help_starts_with_the_script_name(repo_root):
    for script in _entry_points(repo_root):
        text = _help_text(script)
        first = text.splitlines()[0] if text.splitlines() else ""
        assert first.startswith(script.name), (
            f"{script.name} --help should describe itself, got: {first!r}"
        )


def test_help_never_leaks_source_code(repo_root):
    for script in _entry_points(repo_root):
        text = _help_text(script)
        for leak in ("usage() {", 'source "$(dirname', "sed -n", "#!/usr/bin/env"):
            assert leak not in text, f"{script.name} --help leaked source: {leak!r}"


def test_help_documents_usage_and_is_substantial(repo_root):
    for script in _entry_points(repo_root):
        text = _help_text(script)
        assert "Usage:" in text, f"{script.name} --help has no Usage section"
        assert len(text.splitlines()) >= 5


@pytest.mark.parametrize("group", ["paper", "t0", "all"])
def test_model_lists_are_readable_by_the_shell_harness(repo_root, group):
    result = subprocess.run(
        ["/usr/bin/env", "uv", "run", "python", "-m", "src.models", group],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=120,
        check=True,
    )
    names = result.stdout.split()
    expected = {
        "paper": list(PAPER_MODELS),
        "t0": list(T0_MODELS),
        "all": [*PAPER_MODELS, *T0_MODELS],
    }[group]
    assert names == expected


def test_reproduce_paper_script_is_gone(repo_root):
    assert not (repo_root / "reproduce_paper.sh").exists()

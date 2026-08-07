"""Enforce the two repository rules that ruff cannot express.

1. No comments. Python source carries meaning in names, types, and Google-style
   docstrings. Only three pragma forms survive, because strict typing and lint
   suppression have nowhere else to live: ``# type: ignore[...]``, ``# noqa:...``,
   and a first-line shebang.
2. No file over 500 lines.

Comment detection is token-based rather than textual: the codebase is full of hex
colour literals such as ``"#d73027"``, and any regex over raw lines reports them
as comments.
"""

from __future__ import annotations

import argparse
import re
import sys
import tokenize
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_LINES = 500
PYTHON_ROOTS = ("src", "tests", "tools")
SHELL_ROOTS = ("scripts",)
SKIP_DIRS = frozenset({"__pycache__", ".venv", ".ipynb_checkpoints", "notebooks"})

ALLOWED_PRAGMA = re.compile(
    r"^#\s*(type:\s*ignore(\[[^\]]*\])?|noqa(:\s*[A-Z]+[0-9]+(,\s*)?)*)\s*$"
)
SHEBANG = re.compile(r"^#!")


class Finding(tuple[str, int, str]):
    """A single style violation as ``(path, line, message)``."""

    __slots__ = ()


def _relative(path: Path) -> str:
    """Render a path relative to the repository root when possible.

    Args:
        path: Path to render.

    Returns:
        Repo-relative POSIX path, or the absolute path when outside the repo.
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def check_python_comments(path: Path) -> list[Finding]:
    """Report every comment token that is not an allowed pragma.

    Args:
        path: Python file to scan.

    Returns:
        One finding per disallowed comment.
    """
    findings: list[Finding] = []
    with path.open("rb") as stream:
        try:
            tokens = list(tokenize.tokenize(stream.readline))
        except (tokenize.TokenError, SyntaxError) as error:
            return [Finding((_relative(path), 0, f"could not tokenize: {error}"))]

    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        text = token.string.strip()
        line = token.start[0]
        if ALLOWED_PRAGMA.match(text):
            continue
        if line == 1 and SHEBANG.match(text):
            continue
        findings.append(Finding((_relative(path), line, f"comment not allowed: {text[:60]}")))
    return findings


HEREDOC_START = re.compile(r"<<-?\s*'?([A-Za-z_][A-Za-z0-9_]*)'?")


def check_shell_comments(path: Path) -> list[Finding]:
    """Report every comment line in a shell script.

    Heredoc bodies are skipped: ``usage()`` renders its help text from a
    heredoc, and a line starting with ``#`` inside one is help text, not a
    comment.

    Args:
        path: Shell script to scan.

    Returns:
        One finding per comment line, excluding a first-line shebang.
    """
    findings: list[Finding] = []
    terminator: str | None = None

    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()

        if terminator is not None:
            if stripped == terminator:
                terminator = None
            continue

        match = HEREDOC_START.search(raw)
        if match:
            terminator = match.group(1)
            continue

        if not stripped.startswith("#"):
            continue
        if number == 1 and SHEBANG.match(stripped):
            continue
        findings.append(Finding((_relative(path), number, f"comment not allowed: {stripped[:60]}")))
    return findings


def check_length(path: Path) -> list[Finding]:
    """Report the file when it exceeds the line budget.

    Args:
        path: File to measure.

    Returns:
        A single finding when the file is too long, otherwise an empty list.
    """
    count = len(path.read_text(encoding="utf-8").splitlines())
    if count <= MAX_LINES:
        return []
    return [Finding((_relative(path), count, f"{count} lines exceeds the {MAX_LINES}-line limit"))]


def _discover(roots: tuple[str, ...], suffix: str) -> list[Path]:
    """Collect files with a suffix under the given repository roots.

    Args:
        roots: Directory names relative to the repository root.
        suffix: File suffix to match, including the dot.

    Returns:
        Sorted list of matching paths.
    """
    found: list[Path] = []
    for name in roots:
        root = PROJECT_ROOT / name
        if not root.exists():
            continue
        found.extend(
            path for path in root.rglob(f"*{suffix}") if not SKIP_DIRS.intersection(path.parts)
        )
    return sorted(found)


def check_paths(paths: list[Path]) -> list[Finding]:
    """Run every applicable check against the given files.

    Args:
        paths: Files to check; non-existent and unsupported files are ignored.

    Returns:
        All findings, in discovery order.
    """
    findings: list[Finding] = []
    for path in paths:
        if not path.is_file():
            continue
        if path.suffix == ".py":
            findings.extend(check_python_comments(path))
            findings.extend(check_length(path))
        elif path.suffix == ".sh":
            findings.extend(check_shell_comments(path))
            findings.extend(check_length(path))
    return findings


def main() -> int:
    """Run the command-line interface.

    Returns:
        Process exit status: 0 when no findings were reported.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="files to check")
    parser.add_argument("--all", action="store_true", help="check every tracked source file")
    args = parser.parse_args()

    if args.all:
        targets = _discover(PYTHON_ROOTS, ".py") + _discover(SHELL_ROOTS, ".sh")
        targets.extend(
            path for path in PROJECT_ROOT.glob("*.sh") if not SKIP_DIRS.intersection(path.parts)
        )
    else:
        targets = args.paths

    if not targets:
        print("check_style: nothing to check")
        return 0

    findings = check_paths(sorted(set(targets)))
    for path, line, message in findings:
        print(f"{path}:{line}: {message}")

    if findings:
        print(f"\ncheck_style: {len(findings)} finding(s) in {len(targets)} file(s)")
        return 1
    print(f"check_style: {len(targets)} file(s) clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())

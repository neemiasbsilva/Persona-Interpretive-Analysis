"""Record and compare SHA-256 manifests of generated artifacts.

The manifest is the operational definition of "behaviour-preserving" during the
refactor: any phase that changes a byte under ``outputs/`` or ``figures/`` has
changed a published number or figure and must be investigated.

PDF files embed a creation timestamp, so they differ on every regeneration even
when their content is identical. They are hashed and reported in a separate
group that the differ does not count as a failure unless ``--strict-pdf`` is
given; compare the sibling PNG instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOTS = ("outputs", "figures")
SKIP_DIRS = frozenset({"logs", "__pycache__", ".ipynb_checkpoints"})
TIMESTAMPED_SUFFIXES = frozenset({".pdf"})


def _iter_files(root: Path) -> list[Path]:
    """Return every file under ``root``, skipping regenerated-noise directories.

    Args:
        root: Directory to walk.

    Returns:
        Sorted list of absolute file paths.
    """
    if not root.exists():
        return []
    found = [
        path
        for path in root.rglob("*")
        if path.is_file() and not SKIP_DIRS.intersection(path.relative_to(root).parts)
    ]
    return sorted(found)


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


FileEntry = dict[str, object]
FileTable = dict[str, FileEntry]


class Manifest(TypedDict):
    """A recorded snapshot of every generated artifact."""

    generated: str
    project_root: str
    files: FileTable


def build_file_table(project_root: Path = PROJECT_ROOT) -> FileTable:
    """Hash every artifact under the snapshot roots.

    Args:
        project_root: Repository root containing ``outputs/`` and ``figures/``.

    Returns:
        Mapping of repo-relative path to size and digest.
    """
    files: FileTable = {}
    for name in SNAPSHOT_ROOTS:
        for path in _iter_files(project_root / name):
            key = path.relative_to(project_root).as_posix()
            files[key] = {"sha256": _sha256(path), "size": path.stat().st_size}
    return files


def build_manifest(project_root: Path = PROJECT_ROOT) -> Manifest:
    """Build a complete manifest with provenance.

    Args:
        project_root: Repository root containing ``outputs/`` and ``figures/``.

    Returns:
        Manifest recording every artifact hash.
    """
    return Manifest(
        generated=datetime.now(UTC).isoformat(),
        project_root=str(project_root),
        files=build_file_table(project_root),
    )


def write_manifest(destination: Path, project_root: Path = PROJECT_ROOT) -> int:
    """Write a fresh manifest to disk.

    Args:
        destination: Path of the JSON manifest to create.
        project_root: Repository root to snapshot.

    Returns:
        Number of files recorded.
    """
    manifest = build_manifest(project_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return len(manifest["files"])


def diff_manifest(
    baseline_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Compare the working tree against a recorded manifest.

    Args:
        baseline_path: Manifest written by :func:`write_manifest`.
        project_root: Repository root to compare.

    Returns:
        Tuple of ``(changed, timestamped, added, removed)`` repo-relative paths,
        where ``timestamped`` holds differing files whose format embeds a
        creation time and therefore cannot be compared byte-for-byte.
    """
    baseline: FileTable = json.loads(baseline_path.read_text(encoding="utf-8"))["files"]
    current = build_file_table(project_root)

    changed: list[str] = []
    timestamped: list[str] = []
    for key, entry in sorted(baseline.items()):
        if key not in current:
            continue
        if current[key]["sha256"] == entry["sha256"]:
            continue
        if Path(key).suffix in TIMESTAMPED_SUFFIXES:
            timestamped.append(key)
        else:
            changed.append(key)

    added = sorted(set(current) - set(baseline))
    removed = sorted(set(baseline) - set(current))
    return changed, timestamped, added, removed


def _report(title: str, paths: list[str], limit: int = 40) -> None:
    """Print one group of differing paths.

    Args:
        title: Group heading.
        paths: Repo-relative paths in the group.
        limit: Maximum number of paths to print.
    """
    if not paths:
        return
    print(f"\n{title} ({len(paths)}):")
    for path in paths[:limit]:
        print(f"  {path}")
    if len(paths) > limit:
        print(f"  ... and {len(paths) - limit} more")


def main() -> int:
    """Run the command-line interface.

    Returns:
        Process exit status: 0 when the tree matches the baseline.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", type=Path, metavar="PATH", help="write a new manifest")
    group.add_argument("--diff", type=Path, metavar="PATH", help="compare against a manifest")
    parser.add_argument(
        "--strict-pdf",
        action="store_true",
        help="treat differing PDFs as failures instead of reporting them separately",
    )
    args = parser.parse_args()

    if args.write is not None:
        count = write_manifest(args.write)
        print(f"Wrote {count} artifact hashes to {args.write}")
        return 0

    changed, timestamped, added, removed = diff_manifest(args.diff)
    _report("CHANGED", changed)
    _report("CHANGED (timestamped format, compare the PNG instead)", timestamped)
    _report("ADDED", added)
    _report("REMOVED", removed)

    failed = bool(changed or removed) or (args.strict_pdf and bool(timestamped))
    if failed:
        print("\nSnapshot differs from baseline.")
        return 1
    print("\nSnapshot matches baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

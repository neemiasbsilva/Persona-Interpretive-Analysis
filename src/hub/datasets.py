"""What this repository publishes to the Hub, staged as a tree of file symlinks.

The dataset mirrors the working copy path for path, so ``hub pull`` puts every file
back where :mod:`src.config` reads it:

    data/<model>/            the corpus files under their pipeline-standard names
    outputs/<model>/         CSV, JSON, parquet, npy and txt artifacts, embeddings included
    outputs/heatmap_values/  the exported heatmap matrices
    outputs/                 the cross-model tables
    figures/<model>/         per-model figures, PDF and PNG
    figures/                 cross-model figures

:func:`is_published` is the single rule, built from :mod:`src.models`; staging and
pulling both apply it. Run logs, LaTeX tables, pickles and the PerceptSent example
image never match it. Zero-byte files are skipped, and so are the
``t0_economic_status_*`` copies of the T=0 corpora, which repeat the standard names
byte for byte.

Nothing is copied. ``upload_folder`` keeps the ``Path.glob("**/*")`` entries that pass
``is_file()``, which follows a file symlink, while Python 3.11's recursive glob never
enters a symlinked directory; directories are therefore real and only files are links.
"""

from __future__ import annotations

import filecmp
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..models import PAPER_MODELS, T0_MODELS
from . import megabytes

MODELS: tuple[str, ...] = (*PAPER_MODELS, *T0_MODELS)
CORPUS_FILES: dict[str, str] = {
    "baseline": "annotations_baseline.jsonl",
    "failures": "annotation_failures.jsonl",
    "no_persona_think": "annotations_no_persona_think.jsonl",
    "no_persona_no_think": "annotations_no_persona_no_think.jsonl",
}
LEGACY_PREFIX = "t0_economic_status_"
SHARED_OUTPUT_DIRS: tuple[str, ...] = ("heatmap_values",)
OUTPUT_SUFFIXES = frozenset({".csv", ".json", ".npy", ".parquet", ".txt"})
FIGURE_SUFFIXES = frozenset({".pdf", ".png"})
EXCLUDED_PARTS = frozenset({"logs", "__pycache__", ".ipynb_checkpoints"})
EMBEDDING_NAMES: tuple[str, ...] = ("caption_embeddings.npy", "justification_embeddings.npy")
PUBLISHED_ROOTS: tuple[str, ...] = ("data", "outputs", "figures")
CARD_NAME = "README.md"


@dataclass(frozen=True, slots=True)
class StagingReport:
    """Outcome of one staging build.

    Attributes:
        files: Published files linked in the staging tree.
        size_bytes: Their combined size.
        linked: Links created or repointed by this build.
        pruned: Staged entries removed because they are no longer published.
        unpublished: Legacy T=0 copies that differ from their standard-name corpus.
    """

    files: int
    size_bytes: int
    linked: int
    pruned: int
    unpublished: tuple[str, ...]


def _in_layout(parts: tuple[str, ...], directories: tuple[str, ...]) -> bool:
    return len(parts) == 2 or (len(parts) == 3 and parts[1] in directories)


def is_published(relative: PurePosixPath) -> bool:
    """Decide whether a repository-relative path belongs to the dataset.

    Args:
        relative: POSIX path relative to the repository root.

    Returns:
        ``True`` for the corpus, output and figure files the dataset carries.
    """
    parts = relative.parts
    if not parts or any(part.startswith(".") for part in parts):
        return False
    if EXCLUDED_PARTS.intersection(parts):
        return False
    if parts[0] == "data":
        return len(parts) == 3 and parts[1] in MODELS and parts[2] in CORPUS_FILES.values()
    if parts[0] == "outputs":
        return relative.suffix in OUTPUT_SUFFIXES and _in_layout(
            parts, (*MODELS, *SHARED_OUTPUT_DIRS)
        )
    if parts[0] == "figures":
        return relative.suffix in FIGURE_SUFFIXES and _in_layout(parts, MODELS)
    return False


def walk_files(directory: Path) -> Iterator[Path]:
    """Yield the files under a directory, skipping hidden and excluded directories.

    Symlinked files are yielded; symlinked directories are never entered.

    Args:
        directory: Root of the walk; a missing directory yields nothing.

    Yields:
        File paths, sorted within each directory.
    """
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = sorted(
            name for name in dirnames if not name.startswith(".") and name not in EXCLUDED_PARTS
        )
        for name in sorted(filenames):
            if not name.startswith("."):
                yield Path(dirpath) / name


def published_files(project_root: Path) -> dict[str, Path]:
    """Map every non-empty published file to its repository path.

    Args:
        project_root: Repository root to publish from.

    Returns:
        Repository-relative POSIX path to the source file.
    """
    published: dict[str, Path] = {}
    for root in PUBLISHED_ROOTS:
        for path in walk_files(project_root / root):
            relative = PurePosixPath(path.relative_to(project_root).as_posix())
            if is_published(relative) and path.is_file() and path.stat().st_size > 0:
                published[str(relative)] = path
    return published


def unpublished_variants(project_root: Path) -> tuple[str, ...]:
    """Name the legacy T=0 copies that publishing only the standard names would lose.

    Args:
        project_root: Repository root.

    Returns:
        Paths of non-empty ``t0_economic_status_*`` files whose standard-name sibling
        is missing or differs byte for byte.
    """
    lost: list[str] = []
    for model in MODELS:
        for name in CORPUS_FILES.values():
            legacy = project_root / "data" / model / f"{LEGACY_PREFIX}{name}"
            if not legacy.is_file() or legacy.stat().st_size == 0:
                continue
            standard = legacy.with_name(name)
            if not standard.is_file() or not filecmp.cmp(legacy, standard, shallow=False):
                lost.append(legacy.relative_to(project_root).as_posix())
    return tuple(lost)


def _link(source: Path, target: Path) -> bool:
    resolved = source.resolve()
    if target.is_symlink() and target.readlink() == resolved:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(resolved)
    return True


def prune_staging(staging: Path, keep: set[str]) -> int:
    """Remove the staged entries that are no longer published, then empty directories.

    The card is kept, since every build rewrites it. A symlinked directory is always
    removed, because only files are ever linked.

    Args:
        staging: Staging tree root.
        keep: Repository-relative paths that are still published.

    Returns:
        Number of entries removed.
    """
    if not staging.is_dir():
        return 0
    kept = {*keep, CARD_NAME}
    pruned = 0
    for dirpath, dirnames, filenames in os.walk(staging, topdown=False):
        directory = Path(dirpath)
        stale = [
            directory / name
            for name in filenames
            if (directory / name).relative_to(staging).as_posix() not in kept
        ]
        stale.extend(directory / name for name in dirnames if (directory / name).is_symlink())
        for entry in stale:
            entry.unlink()
        pruned += len(stale)
        if directory != staging and not any(directory.iterdir()):
            directory.rmdir()
    return pruned


def build_staging(
    project_root: Path, staging: Path, *, on_event: Callable[[str], None] = print
) -> StagingReport:
    """Link every published file into the staging tree and prune what left it.

    Args:
        project_root: Repository root to publish from.
        staging: Staging tree root, created when absent.
        on_event: Receives one progress line per event.

    Returns:
        Counts for the build, including any legacy T=0 copy that differs.
    """
    staging.mkdir(parents=True, exist_ok=True)
    published = published_files(project_root)
    unpublished = unpublished_variants(project_root)
    for path in unpublished:
        on_event(f"error     {path} differs from its standard-name corpus")
    pruned = prune_staging(staging, set(published))
    linked = sum(_link(source, staging / relative) for relative, source in published.items())
    size_bytes = sum(source.stat().st_size for source in published.values())
    on_event(
        f"staged    {len(published)} files, {megabytes(size_bytes)} "
        f"(linked {linked}, pruned {pruned})"
    )
    return StagingReport(len(published), size_bytes, linked, pruned, unpublished)


def dataset_chunks(staging: Path) -> list[Path]:
    """Return the directories uploaded as one commit each.

    A chunk is a real directory directly under a published root, such as
    ``data/qwen-vl`` or ``outputs/heatmap_values``; a published root itself never is, so
    no upload can delete a whole root on the Hub.

    Args:
        staging: Staging tree root.

    Returns:
        Chunk directories in sorted order.
    """
    chunks: list[Path] = []
    for root in PUBLISHED_ROOTS:
        directory = staging / root
        if not directory.is_dir():
            continue
        chunks.extend(
            entry
            for entry in directory.iterdir()
            if entry.is_dir() and not entry.is_symlink() and not entry.name.startswith(".")
        )
    return sorted(chunks)


def dataset_files(staging: Path) -> list[Path]:
    """Return the staged files outside every chunk, the card included.

    Args:
        staging: Staging tree root.

    Returns:
        Top-level files such as ``README.md`` and ``figures/fig_t0_ablation.pdf``.
    """
    chunks = dataset_chunks(staging)
    return sorted(
        path
        for path in walk_files(staging)
        if not any(path.is_relative_to(chunk) for chunk in chunks)
    )

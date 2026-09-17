"""Pulling the published dataset back into the working copy.

The revision is resolved on the Hub first, so an unreachable Hub, a private repository
without a token or a mistyped id fails the pull instead of falling back to whatever an
earlier pull left behind. The selected files of that exact commit then download into
``hub/downloads/``, where huggingface_hub also keeps its ``.cache/huggingface/``
bookkeeping, and only those files are considered: a stale download the commit no longer
lists is never placed.

Each file is copied to its own path in the working copy only when that path is missing:
an existing file is replaced only on request, and only when its bytes differ, so a local
regeneration is never silently discarded. Copies land through a ``.partial`` file and a
rename, and a failed copy removes its ``.partial``, so a pull never leaves a truncated
embedding cache behind.

Only paths :func:`src.hub.datasets.is_published` accepts are ever written, so a pull
fills exactly the layout :mod:`src.config` reads. The card is downloaded with the
``card`` group but never placed, since its path is the repository README.
"""

from __future__ import annotations

import filecmp
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Literal

from huggingface_hub import HfApi, snapshot_download

from . import REPO_TYPE
from .datasets import CARD_NAME, EMBEDDING_NAMES, is_published

Outcome = Literal["placed", "identical", "kept", "replaced"]

EMBEDDING_PATTERNS: tuple[str, ...] = tuple(f"outputs/*/{name}" for name in EMBEDDING_NAMES)
GROUPS: dict[str, tuple[str, ...]] = {
    "corpora": ("data/*/*.jsonl",),
    "outputs": ("outputs/*",),
    "embeddings": EMBEDDING_PATTERNS,
    "figures": ("figures/*",),
    "card": (CARD_NAME,),
}


@dataclass(frozen=True, slots=True)
class DownloadPlan:
    """Patterns selecting what a pull downloads and places.

    Attributes:
        allow: Repository path patterns to include.
        ignore: Repository path patterns to leave out.
    """

    allow: tuple[str, ...]
    ignore: tuple[str, ...]

    def selects(self, relative: str) -> bool:
        """Apply the patterns the way ``snapshot_download`` does, with ``fnmatch``.

        Args:
            relative: Repository path.

        Returns:
            ``True`` when an allow pattern matches and no ignore pattern does.
        """
        allowed = any(fnmatch(relative, pattern) for pattern in self.allow)
        return allowed and not any(fnmatch(relative, pattern) for pattern in self.ignore)


@dataclass(frozen=True, slots=True)
class PullReport:
    """What a pull did to each published file it downloaded.

    Attributes:
        revision: Commit the files came from.
        placed: Paths copied because the working copy did not have them.
        identical: Paths the working copy already held byte for byte.
        kept: Paths left alone because they differ and overwriting was not requested.
        replaced: Paths overwritten because they differed and overwriting was requested.
    """

    revision: str
    placed: tuple[str, ...]
    identical: tuple[str, ...]
    kept: tuple[str, ...]
    replaced: tuple[str, ...]


def download_plan(groups: Sequence[str] | None = None) -> DownloadPlan:
    """Translate group names into download patterns.

    The embedding caches are ignored unless their own group is selected, even when the
    ``outputs`` group would otherwise match them.

    Args:
        groups: Names from :data:`GROUPS`; ``None`` or empty selects every group.

    Returns:
        The allow and ignore patterns.

    Raises:
        ValueError: When a group name is unknown.
    """
    selected = list(groups) if groups else list(GROUPS)
    unknown = sorted(set(selected) - set(GROUPS))
    if unknown:
        raise ValueError(f"unknown groups: {', '.join(unknown)}; choose from {', '.join(GROUPS)}")
    allow = tuple(dict.fromkeys(pattern for name in selected for pattern in GROUPS[name]))
    ignore = () if "embeddings" in selected else EMBEDDING_PATTERNS
    return DownloadPlan(allow, ignore)


def remote_files(repo_id: str, *, revision: str | None = None) -> tuple[str, list[str]]:
    """Resolve a revision of the dataset and list the files it holds.

    Args:
        repo_id: Dataset repository id.
        revision: Branch, tag or commit; the default branch when ``None``.

    Returns:
        The commit hash and the sorted repository paths at that commit.
    """
    info = HfApi().dataset_info(repo_id, revision=revision)
    return info.sha or revision or "", sorted(sibling.rfilename for sibling in info.siblings or [])


def download_dataset(
    repo_id: str, destination: Path, plan: DownloadPlan, *, revision: str | None = None
) -> Path:
    """Download the planned files of the dataset into a local directory.

    Args:
        repo_id: Dataset repository id.
        destination: Directory mirroring the repository layout.
        plan: Patterns selecting the files.
        revision: Branch, tag or commit; the default branch when ``None``.

    Returns:
        The directory the files were downloaded into.
    """
    destination.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id,
        repo_type=REPO_TYPE,
        revision=revision,
        local_dir=destination,
        allow_patterns=list(plan.allow),
        ignore_patterns=list(plan.ignore) or None,
    )
    return Path(path)


def _place(source: Path, target: Path, *, overwrite: bool) -> Outcome:
    exists = target.is_file()
    if exists and filecmp.cmp(source, target, shallow=False):
        return "identical"
    if exists and not overwrite:
        return "kept"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    try:
        shutil.copy2(source, partial)
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return "replaced" if exists else "placed"


def materialise(
    downloads: Path,
    project_root: Path,
    paths: Sequence[str],
    *,
    revision: str = "",
    overwrite: bool = False,
) -> PullReport:
    """Copy the listed, published files from the downloads into the working copy.

    Args:
        downloads: Directory the dataset was downloaded into.
        project_root: Repository root to fill.
        paths: Repository paths this pull selected; nothing else is placed.
        revision: Commit the paths were listed at, for the report.
        overwrite: Replace working-copy files whose bytes differ.

    Returns:
        What happened to each file.

    Raises:
        FileNotFoundError: When a listed path is missing from the downloads.
    """
    selected = [path for path in paths if is_published(PurePosixPath(path))]
    missing = [path for path in selected if not (downloads / path).is_file()]
    if missing:
        raise FileNotFoundError(f"not downloaded: {', '.join(missing)}")
    outcomes: dict[Outcome, list[str]] = {
        "placed": [],
        "identical": [],
        "kept": [],
        "replaced": [],
    }
    for relative in selected:
        outcome = _place(downloads / relative, project_root / relative, overwrite=overwrite)
        outcomes[outcome].append(relative)
    return PullReport(
        revision,
        tuple(outcomes["placed"]),
        tuple(outcomes["identical"]),
        tuple(outcomes["kept"]),
        tuple(outcomes["replaced"]),
    )


def pull(
    repo_id: str,
    downloads: Path,
    project_root: Path,
    *,
    groups: Sequence[str] | None = None,
    revision: str | None = None,
    overwrite: bool = False,
) -> PullReport:
    """Download the selected groups and place them in the working copy.

    Args:
        repo_id: Dataset repository id.
        downloads: Directory to download into.
        project_root: Repository root to fill.
        groups: Names from :data:`GROUPS`; ``None`` selects every group.
        revision: Branch, tag or commit; the default branch when ``None``.
        overwrite: Replace working-copy files whose bytes differ.

    Returns:
        What happened to each file.
    """
    plan = download_plan(groups)
    commit, files = remote_files(repo_id, revision=revision)
    selected = [path for path in files if plan.selects(path)]
    download_dataset(repo_id, downloads, plan, revision=commit or None)
    return materialise(downloads, project_root, selected, revision=commit, overwrite=overwrite)

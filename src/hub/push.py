"""Chunked, resumable uploads to the Hub.

``upload_folder`` already streams through Xet and skips files the Hub already holds, but
it does not remember across runs which chunks are current, and hashing a 200 MB chunk
remotely costs a round trip. A digest of every chunk's paths, sizes and contents is
therefore recorded in ``hub/upload_state.json`` and compared before each upload.

Each chunk is one commit, and the top-level files, the card among them, share one commit
sent after the chunks, so every path the card declares exists before the card. Uploads
only ever add or replace files. A file missing from the staging tree is never deleted
from the Hub by an upload, because a checkout that pulled only some groups is missing
files on purpose; :func:`delete_paths` removes remote-only files, and only when a push
asks for it with ``--prune``, after listing them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from ..hashing import sha256_file
from . import REPO_TYPE, megabytes

IGNORE_PATTERNS: tuple[str, ...] = ("*.partial", "*.tmp", ".DS_Store")


class UploadState:
    """Digests of what was last uploaded, persisted as JSON between runs."""

    def __init__(self, path: Path) -> None:
        """Load the recorded digests, starting empty when the file is absent or unreadable.

        Args:
            path: JSON file holding the digests.
        """
        self.path = path
        self.entries: dict[str, str] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                self.entries = {str(key): str(value) for key, value in loaded.items()}

    def is_current(self, key: str, digest: str) -> bool:
        """Report whether a key was last uploaded with exactly this digest.

        Args:
            key: State key from :func:`state_key`.
            digest: Digest of the local contents.

        Returns:
            ``True`` when the recorded digest matches.
        """
        return self.entries.get(key) == digest

    def record(self, key: str, digest: str) -> None:
        """Record a successful upload and persist the state atomically.

        Args:
            key: State key from :func:`state_key`.
            digest: Digest of the uploaded contents.
        """
        self.entries[key] = digest
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(self.path.name + ".tmp")
        payload = json.dumps(self.entries, indent=2, sort_keys=True) + "\n"
        partial.write_text(payload, encoding="utf-8")
        partial.replace(self.path)

    def forget(self, keys: Sequence[str]) -> None:
        """Drop recorded digests, so the next push uploads those paths again.

        Args:
            keys: State keys from :func:`state_key`; unknown keys are ignored.
        """
        dropped = [key for key in keys if self.entries.pop(key, None) is not None]
        if dropped:
            self._write()

    def count(self, repo_id: str) -> int:
        """Count the uploads recorded for one repository.

        Args:
            repo_id: Dataset repository id.

        Returns:
            Number of recorded chunks and files.
        """
        prefix = state_key(repo_id, "")
        return sum(1 for key in self.entries if key.startswith(prefix))


@dataclass(frozen=True, slots=True)
class UploadSummary:
    """Outcome of one upload pass.

    Attributes:
        uploaded: Chunks or files sent, or that a dry run would send.
        skipped: Chunks or files whose recorded digest was current.
        total: Chunks or files considered.
    """

    uploaded: int
    skipped: int
    total: int


def state_key(repo_id: str, relative: str) -> str:
    """Build the state key of a chunk or file, scoped to its repository.

    Args:
        repo_id: Dataset repository id.
        relative: Path in the repository.

    Returns:
        ``"dataset:<repo_id>:<relative>"``.
    """
    return f"{REPO_TYPE}:{repo_id}:{relative}"


def _ignored(path: Path) -> bool:
    return any(fnmatch(path.name, pattern) for pattern in IGNORE_PATTERNS)


def uploaded_files(folder: Path) -> list[Path]:
    """List the files an upload of a folder would send, following file symlinks.

    Args:
        folder: Chunk directory.

    Returns:
        Sorted files, without those matching :data:`IGNORE_PATTERNS`.
    """
    return sorted(path for path in folder.rglob("*") if path.is_file() and not _ignored(path))


def folder_digest(folder: Path) -> str:
    """Digest a chunk's relative paths, sizes and contents.

    Args:
        folder: Chunk directory.

    Returns:
        Lowercase hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    for path in uploaded_files(folder):
        relative = path.relative_to(folder).as_posix()
        line = f"{relative}:{path.stat().st_size}:{sha256_file(path)}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _require(api: HfApi | None) -> HfApi:
    if api is None:
        raise ValueError("an authenticated client is required unless dry_run is set")
    return api


def upload_chunks(
    api: HfApi | None,
    repo_id: str,
    staging: Path,
    chunks: Sequence[Path],
    state: UploadState,
    *,
    dry_run: bool = False,
    force: bool = False,
    on_event: Callable[[str], None] = print,
) -> UploadSummary:
    """Upload every chunk whose contents changed since its recorded upload.

    Args:
        api: Authenticated client; may be ``None`` for a dry run.
        repo_id: Dataset repository id.
        staging: Staging tree root.
        chunks: Chunk directories inside ``staging``.
        state: Recorded digests, updated after each successful upload.
        dry_run: Report what would be uploaded and upload nothing.
        force: Upload every chunk regardless of the recorded digests.
        on_event: Receives one progress line per chunk.

    Returns:
        Counts of uploaded and skipped chunks.
    """
    uploaded = skipped = 0
    for chunk in chunks:
        relative = chunk.relative_to(staging).as_posix()
        key = state_key(repo_id, relative)
        digest = folder_digest(chunk)
        size = megabytes(sum(path.stat().st_size for path in uploaded_files(chunk)))
        if not force and state.is_current(key, digest):
            skipped += 1
            on_event(f"skip      {relative}  ({size})")
            continue
        uploaded += 1
        on_event(f"upload    {relative}  ({size})")
        if dry_run:
            continue
        _require(api).upload_folder(
            repo_id=repo_id,
            folder_path=chunk,
            path_in_repo=relative,
            repo_type=REPO_TYPE,
            commit_message=f"Upload {relative}",
            ignore_patterns=list(IGNORE_PATTERNS),
        )
        state.record(key, digest)
    return UploadSummary(uploaded, skipped, len(chunks))


def upload_files(
    api: HfApi | None,
    repo_id: str,
    staging: Path,
    files: Sequence[Path],
    state: UploadState,
    *,
    dry_run: bool = False,
    force: bool = False,
    on_event: Callable[[str], None] = print,
) -> UploadSummary:
    """Upload the changed top-level files together, in a single commit.

    Args:
        api: Authenticated client; may be ``None`` for a dry run.
        repo_id: Dataset repository id.
        staging: Staging tree root.
        files: Files inside ``staging`` that belong to no chunk.
        state: Recorded digests, updated once the commit succeeds.
        dry_run: Report what would be uploaded and upload nothing.
        force: Upload every file regardless of the recorded digests.
        on_event: Receives one progress line per file.

    Returns:
        Counts of uploaded and skipped files.
    """
    pending: list[tuple[str, Path, str]] = []
    for path in files:
        relative = path.relative_to(staging).as_posix()
        digest = sha256_file(path)
        if not force and state.is_current(state_key(repo_id, relative), digest):
            on_event(f"skip      {relative}")
            continue
        on_event(f"upload    {relative}")
        pending.append((relative, path, digest))
    if pending and not dry_run:
        _require(api).create_commit(
            repo_id,
            [
                CommitOperationAdd(path_in_repo=relative, path_or_fileobj=path)
                for relative, path, _ in pending
            ],
            commit_message=f"Upload {len(pending)} top-level files",
            repo_type=REPO_TYPE,
        )
        for relative, _, digest in pending:
            state.record(state_key(repo_id, relative), digest)
    return UploadSummary(len(pending), len(files) - len(pending), len(files))


def remote_only_paths(remote: Sequence[str], staging: Path) -> list[str]:
    """List the remote files that the staging tree no longer has.

    Args:
        remote: Repository paths currently on the Hub.
        staging: Staging tree root.

    Returns:
        Sorted paths present on the Hub but absent from staging, ``.gitattributes`` aside.
    """
    return sorted(
        path for path in remote if path != ".gitattributes" and not (staging / path).is_file()
    )


def delete_paths(
    api: HfApi | None,
    repo_id: str,
    paths: Sequence[str],
    state: UploadState,
    *,
    dry_run: bool = False,
    on_event: Callable[[str], None] = print,
) -> int:
    """Delete remote files in a single commit, forgetting any record of them.

    Args:
        api: Authenticated client; may be ``None`` for a dry run.
        repo_id: Dataset repository id.
        paths: Repository paths to delete.
        state: Recorded digests; entries for deleted top-level files are dropped.
        dry_run: Report what would be deleted and delete nothing.
        on_event: Receives one progress line per path.

    Returns:
        Number of paths deleted, or that a dry run would delete.
    """
    for path in paths:
        on_event(f"delete    {path}")
    if paths and not dry_run:
        _require(api).create_commit(
            repo_id,
            [CommitOperationDelete(path_in_repo=path, is_folder=False) for path in paths],
            commit_message=f"Delete {len(paths)} files no longer published",
            repo_type=REPO_TYPE,
        )
        state.forget([state_key(repo_id, path) for path in paths])
    return len(paths)

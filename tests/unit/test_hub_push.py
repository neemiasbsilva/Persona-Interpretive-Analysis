"""Cover publishing: content digests, the upload record, deletions and the push command.

A fake client stands in for ``HfApi``, so every rule a real push depends on is pinned
without the network: uploads only add, a failed upload records nothing, remote files
missing locally are deleted only with ``--prune``, and a dry run needs no token.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from huggingface_hub import HfApi

from src.hub import auth, commands, push


def _write(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class FakeApi:
    """Stand-in for ``HfApi`` that records every call and can be told to fail."""

    def __init__(self, remote: list[str] | None = None, *, fail_uploads: bool = False) -> None:
        """Start with a remote file list and no recorded calls.

        Args:
            remote: Paths ``list_repo_files`` reports.
            fail_uploads: Make ``upload_folder`` and ``create_commit`` raise.
        """
        self.remote = remote or []
        self.fail_uploads = fail_uploads
        self.repos: list[dict[str, object]] = []
        self.listed: list[str] = []
        self.folders: list[dict[str, object]] = []
        self.commits: list[tuple[str, list[Any], dict[str, object]]] = []

    def create_repo(self, repo_id: str, **kwargs: object) -> None:
        """Record a repository creation.

        Args:
            repo_id: Dataset repository id.
            **kwargs: Remaining keyword arguments.
        """
        self.repos.append({"repo_id": repo_id, **kwargs})

    def upload_folder(self, **kwargs: object) -> None:
        """Record one folder upload.

        Args:
            **kwargs: The keyword arguments ``HfApi.upload_folder`` received.

        Raises:
            OSError: When the fake was built to fail uploads.
        """
        if self.fail_uploads:
            raise OSError("connection reset")
        self.folders.append(kwargs)

    def create_commit(self, repo_id: str, operations: list[Any], **kwargs: object) -> None:
        """Record one commit.

        Args:
            repo_id: Dataset repository id.
            operations: Commit operations.
            **kwargs: Remaining keyword arguments.

        Raises:
            OSError: When the fake was built to fail uploads.
        """
        if self.fail_uploads:
            raise OSError("connection reset")
        self.commits.append((repo_id, operations, kwargs))

    def list_repo_files(self, repo_id: str, **_: object) -> list[str]:
        """Record the listing and return the configured remote paths.

        Args:
            repo_id: Dataset repository id.

        Returns:
            The remote file list.
        """
        self.listed.append(repo_id)
        return list(self.remote)


def _staging(root: Path) -> Path:
    staging = root / "staging"
    _write(staging / "data" / "qwen-vl" / "annotations_baseline.jsonl", "{}\n")
    _write(staging / "outputs" / "qwen-vl" / "within_cross_significance.csv", "cell,p\n")
    _write(staging / "outputs" / "cross_model_correlation.csv", "modality,r\n")
    _write(staging / "README.md", "---\nlicense: cc-by-4.0\n---\n")
    return staging


def _chunks(staging: Path) -> list[Path]:
    return [staging / "data" / "qwen-vl", staging / "outputs" / "qwen-vl"]


def _files(staging: Path) -> list[Path]:
    return [staging / "README.md", staging / "outputs" / "cross_model_correlation.csv"]


def _quiet(_: str) -> None:
    return None


def test_folder_digest_reads_contents_not_just_sizes(tmp_path: Path) -> None:
    chunk = tmp_path / "chunk"
    table = _write(chunk / "within_cross_significance.csv", "cell,0.512\n")
    before = push.folder_digest(chunk)

    table.write_text("cell,0.498\n", encoding="utf-8")

    assert push.folder_digest(chunk) != before


def test_folder_digest_ignores_partial_and_tmp_files(tmp_path: Path) -> None:
    chunk = tmp_path / "chunk"
    _write(chunk / "kept.csv", "a\n")
    before = push.folder_digest(chunk)

    _write(chunk / "caption_embeddings.npy.partial", "half a file")
    _write(chunk / "scratch.tmp", "temporary")

    assert push.folder_digest(chunk) == before


def test_upload_state_is_keyed_per_repository(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    chunk = staging / "data" / "qwen-vl"
    state = push.UploadState(tmp_path / "upload_state.json")
    state.record(push.state_key("lab/repo", "data/qwen-vl"), push.folder_digest(chunk))

    ours = push.upload_chunks(None, "lab/repo", staging, [chunk], state, dry_run=True)
    fork = push.upload_chunks(None, "fork/repo", staging, [chunk], state, dry_run=True)

    assert (ours.uploaded, ours.skipped) == (0, 1)
    assert (fork.uploaded, fork.skipped) == (1, 0)
    assert push.UploadState(tmp_path / "upload_state.json").count("lab/repo") == 1


def test_a_dry_run_uploads_nothing_and_records_nothing(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    state_path = tmp_path / "upload_state.json"
    state = push.UploadState(state_path)

    chunks = push.upload_chunks(
        None, "lab/repo", staging, _chunks(staging), state, dry_run=True, on_event=_quiet
    )
    files = push.upload_files(
        None, "lab/repo", staging, _files(staging), state, dry_run=True, on_event=_quiet
    )

    assert (chunks.uploaded, files.uploaded) == (2, 2)
    assert not state_path.exists()


def test_uploads_only_add_and_skip_what_is_already_current(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    fake = FakeApi()
    api = cast("HfApi", fake)
    state = push.UploadState(tmp_path / "upload_state.json")

    push.upload_chunks(api, "lab/repo", staging, _chunks(staging), state, on_event=_quiet)
    push.upload_files(api, "lab/repo", staging, _files(staging), state, on_event=_quiet)
    chunks = push.upload_chunks(api, "lab/repo", staging, _chunks(staging), state, on_event=_quiet)
    files = push.upload_files(api, "lab/repo", staging, _files(staging), state, on_event=_quiet)

    assert [call["path_in_repo"] for call in fake.folders] == ["data/qwen-vl", "outputs/qwen-vl"]
    assert all(call.get("delete_patterns") is None for call in fake.folders)
    assert all(call["repo_type"] == "dataset" for call in fake.folders)
    assert len(fake.commits) == 1
    repo, operations, options = fake.commits[0]
    assert repo == "lab/repo"
    assert [operation.path_in_repo for operation in operations] == [
        "README.md",
        "outputs/cross_model_correlation.csv",
    ]
    assert options["repo_type"] == "dataset"
    assert (chunks.skipped, files.skipped) == (2, 2)


def test_a_failed_upload_records_nothing(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    api = cast("HfApi", FakeApi(fail_uploads=True))
    state_path = tmp_path / "upload_state.json"

    with pytest.raises(OSError, match="connection reset"):
        push.upload_chunks(api, "lab/repo", staging, _chunks(staging), push.UploadState(state_path))
    with pytest.raises(OSError, match="connection reset"):
        push.upload_files(api, "lab/repo", staging, _files(staging), push.UploadState(state_path))

    assert push.UploadState(state_path).count("lab/repo") == 0


def test_force_uploads_what_the_record_calls_current(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    fake = FakeApi()
    api = cast("HfApi", fake)
    state = push.UploadState(tmp_path / "upload_state.json")
    push.upload_chunks(api, "lab/repo", staging, _chunks(staging), state, on_event=_quiet)

    forced = push.upload_chunks(
        api, "lab/repo", staging, _chunks(staging), state, force=True, on_event=_quiet
    )

    assert forced.uploaded == 2
    assert len(fake.folders) == 4


def test_remote_only_paths_leave_the_gitattributes_alone(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    remote = [
        ".gitattributes",
        "README.md",
        "data/qwen-vl/annotations_baseline.jsonl",
        "outputs/qwen-vl/caption_embeddings.npy",
        "figures/fig_old.pdf",
    ]

    assert push.remote_only_paths(remote, staging) == [
        "figures/fig_old.pdf",
        "outputs/qwen-vl/caption_embeddings.npy",
    ]


def test_deleting_paths_commits_once_and_forgets_their_record(tmp_path: Path) -> None:
    fake = FakeApi()
    state = push.UploadState(tmp_path / "upload_state.json")
    state.record(push.state_key("lab/repo", "figures/fig_old.pdf"), "digest")
    paths = ["figures/fig_old.pdf", "outputs/qwen-vl/caption_embeddings.npy"]

    deleted = push.delete_paths(cast("HfApi", fake), "lab/repo", paths, state, on_event=_quiet)

    assert deleted == 2
    assert [operation.path_in_repo for operation in fake.commits[0][1]] == paths
    assert push.UploadState(tmp_path / "upload_state.json").count("lab/repo") == 0


class _Identity:
    """Stand-in client whose token has a read-only role."""

    def __init__(self, token: str) -> None:
        """Keep the token.

        Args:
            token: The resolved token.
        """
        self.token = token

    def whoami(self, **_: object) -> dict[str, object]:
        """Report a read-scoped token.

        Returns:
            A ``whoami`` payload.
        """
        return {"name": "reader", "auth": {"accessToken": {"role": "read"}}}


def test_a_read_scoped_token_is_refused_for_publishing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "read_token", lambda: "hf_read")
    monkeypatch.setattr(auth, "HfApi", _Identity)

    assert auth.resolve_api() is not None
    with pytest.raises(SystemExit, match="write-scoped token is required"):
        auth.resolve_api(write=True)


def _project(root: Path) -> Path:
    project = root / "repo"
    record = json.dumps({"timestamp_utc": "2026-07-21T06:00:00Z"})
    _write(project / "data" / "qwen-vl" / "annotations_baseline.jsonl", record + "\n")
    _write(project / "outputs" / "cross_model_correlation.csv", "modality,r\n")
    return project


def _push(root: Path, project: Path, *extra: str) -> int:
    return commands.main(
        [
            *("push", "--repo", "lab/repo", "--root", str(project)),
            *("--staging", str(root / "hub" / "datasets")),
            *("--state", str(root / "hub" / "upload_state.json")),
            *extra,
        ]
    )


def test_push_dry_run_needs_no_token_and_writes_no_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(**_: object) -> None:
        raise AssertionError("a dry run must not resolve a token")

    monkeypatch.setattr(commands, "resolve_api", refuse)

    status = _push(tmp_path, _project(tmp_path), "--dry-run")

    output = capsys.readouterr().out
    assert status == 0
    assert "would upload 1 of 1 chunks and 2 of 2 top-level files" in output
    assert "dry run: nothing was uploaded or deleted" in output
    assert (tmp_path / "hub" / "datasets" / "README.md").is_file()
    assert not (tmp_path / "hub" / "upload_state.json").exists()


def test_push_refuses_to_publish_when_a_t0_copy_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(**_: object) -> None:
        raise AssertionError("a failed build must not reach the Hub")

    monkeypatch.setattr(commands, "resolve_api", refuse)
    project = _project(tmp_path)
    _write(project / "data" / "qwen-vl-t0" / "annotations_baseline.jsonl", "{}\n")
    _write(
        project / "data" / "qwen-vl-t0" / "t0_economic_status_annotations_baseline.jsonl", "[]\n"
    )

    assert _push(tmp_path, project) == 1


def test_push_keeps_remote_only_files_unless_pruning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    remote = [".gitattributes", "outputs/qwen-vl/caption_embeddings.npy"]
    fake = FakeApi(remote)
    monkeypatch.setattr(commands, "resolve_api", lambda **_: fake)
    project = _project(tmp_path)

    kept = _push(tmp_path, project)
    kept_output = capsys.readouterr().out
    pruned = _push(tmp_path, project, "--prune")

    assert (kept, pruned) == (0, 0)
    assert fake.repos[0]["repo_type"] == "dataset"
    assert "kept      1 remote files missing locally" in kept_output
    deletions = [
        operation.path_in_repo
        for _, operations, _ in fake.commits
        for operation in operations
        if type(operation).__name__ == "CommitOperationDelete"
    ]
    assert deletions == ["outputs/qwen-vl/caption_embeddings.npy"]
    assert "deleted 1 remote files" in capsys.readouterr().out

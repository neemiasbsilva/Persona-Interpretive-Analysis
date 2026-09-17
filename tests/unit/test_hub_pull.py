"""Cover pulling the dataset back into a working copy, and the status report.

``snapshot_download`` and the revision lookup are replaced by fakes, so the placement
rules are pinned without the network: a pull places only the files the resolved commit
lists, fills missing files, never replaces a differing file unless asked, never writes
the card over the README, and fails instead of reusing old downloads.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from src.hub import commands, fetch

SNAPSHOT = {
    "README.md": "---\nlicense: cc-by-4.0\n---\n",
    "data/qwen-vl/annotations_baseline.jsonl": "{}\n",
    "data/qwen-vl/annotation_failures.jsonl": "{}\n",
    "outputs/qwen-vl/within_cross_significance.csv": "cell,p\n",
    "outputs/qwen-vl/caption_embeddings.npy": "embedding bytes",
    "outputs/cross_model_correlation.csv": "modality,r\n",
    "figures/qwen-vl/fig_within_cross.pdf": "%PDF",
}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_snapshot(root: Path) -> Path:
    """Write a downloaded snapshot, bookkeeping included.

    Args:
        root: Download directory.

    Returns:
        The download directory.
    """
    for relative, text in SNAPSHOT.items():
        _write(root / relative, text)
    _write(root / ".cache" / "huggingface" / "download" / "README.md.metadata", "etag\n")
    return root


def _not_found(message: str) -> RepositoryNotFoundError:
    request = httpx.Request("GET", "https://huggingface.co/api/datasets/lab/repo")
    return RepositoryNotFoundError(message, response=httpx.Response(401, request=request))


def _selected(plan: fetch.DownloadPlan) -> list[str]:
    return [path for path in SNAPSHOT if plan.selects(path)]


def test_every_group_is_downloaded_by_default() -> None:
    plan = fetch.download_plan()

    assert plan.selects("data/gemma4-t0/annotations_baseline.jsonl")
    assert plan.selects("figures/fig_t0_ablation.pdf")
    assert plan.selects("outputs/qwen-vl/caption_embeddings.npy")
    assert plan.ignore == ()


def test_embeddings_are_ignored_unless_their_group_is_selected() -> None:
    outputs = fetch.download_plan(["outputs"])
    embeddings = fetch.download_plan(["outputs", "embeddings"])

    assert outputs.selects("outputs/qwen-vl/within_cross_significance.csv")
    assert outputs.selects("outputs/qwen-vl-t0/t0_caption_embeddings.npy")
    assert not outputs.selects("outputs/qwen-vl/caption_embeddings.npy")
    assert embeddings.selects("outputs/qwen-vl/justification_embeddings.npy")


def test_an_unknown_group_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown groups: raw"):
        fetch.download_plan(["corpora", "raw"])


def test_pull_places_missing_files_and_keeps_existing_ones(tmp_path: Path) -> None:
    downloads = make_snapshot(tmp_path / "downloads")
    project = tmp_path / "repo"
    local = _write(project / "data" / "qwen-vl" / "annotations_baseline.jsonl", "local\n")

    report = fetch.materialise(downloads, project, _selected(fetch.download_plan()))

    assert report.kept == ("data/qwen-vl/annotations_baseline.jsonl",)
    assert local.read_text(encoding="utf-8") == "local\n"
    assert "outputs/qwen-vl/caption_embeddings.npy" in report.placed
    assert (project / "figures" / "qwen-vl" / "fig_within_cross.pdf").is_file()


def test_overwrite_replaces_only_files_whose_bytes_differ(tmp_path: Path) -> None:
    downloads = make_snapshot(tmp_path / "downloads")
    project = tmp_path / "repo"
    corpus = _write(project / "data" / "qwen-vl" / "annotations_baseline.jsonl", "local\n")
    _write(project / "outputs" / "cross_model_correlation.csv", "modality,r\n")

    report = fetch.materialise(downloads, project, _selected(fetch.download_plan()), overwrite=True)

    assert report.replaced == ("data/qwen-vl/annotations_baseline.jsonl",)
    assert report.identical == ("outputs/cross_model_correlation.csv",)
    assert corpus.read_text(encoding="utf-8") == "{}\n"


def test_pull_never_writes_the_card_over_the_repository_readme(tmp_path: Path) -> None:
    downloads = make_snapshot(tmp_path / "downloads")
    project = tmp_path / "repo"
    readme = _write(project / "README.md", "# Analysis repository\n")

    report = fetch.materialise(downloads, project, _selected(fetch.download_plan()), overwrite=True)

    assert readme.read_text(encoding="utf-8") == "# Analysis repository\n"
    assert not [path for path in (*report.placed, *report.replaced) if "README" in path]
    assert not (project / ".cache").exists()


def test_a_stale_download_the_commit_no_longer_lists_is_never_placed(tmp_path: Path) -> None:
    downloads = make_snapshot(tmp_path / "downloads")
    _write(downloads / "outputs" / "qwen-vl" / "removed_on_the_hub.csv", "old\n")
    project = tmp_path / "repo"

    report = fetch.materialise(downloads, project, _selected(fetch.download_plan(["outputs"])))

    assert "outputs/qwen-vl/removed_on_the_hub.csv" not in report.placed
    assert not (project / "outputs" / "qwen-vl" / "removed_on_the_hub.csv").exists()


def test_a_listed_file_missing_from_the_downloads_fails_the_pull(tmp_path: Path) -> None:
    downloads = make_snapshot(tmp_path / "downloads")

    with pytest.raises(FileNotFoundError, match=r"figures/fig_t0_ablation\.pdf"):
        fetch.materialise(downloads, tmp_path / "repo", ["figures/fig_t0_ablation.pdf"])


def test_a_failed_copy_leaves_neither_a_partial_file_nor_a_changed_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads = make_snapshot(tmp_path / "downloads")
    project = tmp_path / "repo"
    corpus = _write(project / "data" / "qwen-vl" / "annotations_baseline.jsonl", "local\n")

    def interrupted(source: Path, target: Path) -> None:
        shutil.copyfile(source, target)
        raise OSError("No space left on device")

    monkeypatch.setattr(shutil, "copy2", interrupted)

    with pytest.raises(OSError, match="No space left"):
        fetch.materialise(
            downloads, project, ["data/qwen-vl/annotations_baseline.jsonl"], overwrite=True
        )

    assert corpus.read_text(encoding="utf-8") == "local\n"
    assert not list(project.rglob("*.partial"))


def test_pull_command_resolves_the_commit_then_places_only_its_selected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    downloads = make_snapshot(tmp_path / "downloads")
    _write(downloads / "data" / "qwen-vl" / "annotations_no_persona_think.jsonl", "stale\n")
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
        calls.append({"repo_id": repo_id, **kwargs})
        return str(kwargs["local_dir"])

    monkeypatch.setattr(fetch, "remote_files", lambda *_, **__: ("abc123", sorted(SNAPSHOT)))
    monkeypatch.setattr(fetch, "snapshot_download", fake_snapshot_download)
    project = tmp_path / "repo"
    corpus = _write(project / "data" / "qwen-vl" / "annotations_baseline.jsonl", "local\n")

    status = commands.main(
        [
            *("pull", "--repo", "lab/repo", "--root", str(project)),
            *("--downloads", str(downloads), "--groups", "corpora", "outputs"),
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["revision"] == "abc123"
    assert calls[0]["local_dir"] == downloads
    assert calls[0]["ignore_patterns"] == list(fetch.EMBEDDING_PATTERNS)
    assert corpus.read_text(encoding="utf-8") == "local\n"
    assert "kept      data/qwen-vl/annotations_baseline.jsonl" in output
    assert "revision  abc123" in output
    assert "3 placed, 0 replaced, 1 kept" in output
    assert not (project / "data" / "qwen-vl" / "annotations_no_persona_think.jsonl").exists()
    assert not (project / "outputs" / "qwen-vl" / "caption_embeddings.npy").exists()


def test_pull_fails_instead_of_reusing_old_downloads_when_the_hub_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    downloads = make_snapshot(tmp_path / "downloads")

    def private(*_: object, **__: object) -> tuple[str, list[str]]:
        raise _not_found("401 Client Error: Repository Not Found")

    monkeypatch.setattr(fetch, "remote_files", private)
    project = tmp_path / "repo"

    status = commands.main(
        ["pull", "--repo", "lab/repo", "--root", str(project), "--downloads", str(downloads)]
    )

    assert status == 1
    assert "pull failed" in capsys.readouterr().err
    assert not project.exists()


def test_status_without_token_or_staging_stays_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(commands, "read_token", lambda: None)

    status = commands.main(
        [
            *("status", "--repo", "lab/repo", "--staging", str(tmp_path / "absent")),
            *("--state", str(tmp_path / "upload_state.json")),
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert "token     absent" in output
    assert "staging   not built" in output
    assert "recorded  0 uploads" in output


class _Sibling:
    """A remote file entry."""

    def __init__(self, rfilename: str) -> None:
        """Keep the repository path.

        Args:
            rfilename: Repository path.
        """
        self.rfilename = rfilename


class _Info:
    """A dataset info payload for a private repository."""

    sha = "abc123"
    private = True
    siblings = (_Sibling(".gitattributes"), _Sibling("figures/fig_old.pdf"))


class _StatusApi:
    """Stand-in client for the status report."""

    def __init__(self, *, reachable: bool) -> None:
        """Choose whether the repository answers.

        Args:
            reachable: Return info when ``True``, raise a not-found error otherwise.
        """
        self.reachable = reachable

    def whoami(self, **_: object) -> dict[str, object]:
        """Name the account.

        Returns:
            A ``whoami`` payload.
        """
        return {"name": "maintainer", "orgs": [{"name": "lab"}]}

    def dataset_info(self, repo_id: str) -> _Info:
        """Describe the repository, or refuse.

        Args:
            repo_id: Dataset repository id.

        Returns:
            The info payload.

        Raises:
            RepositoryNotFoundError: When the fake is unreachable.
        """
        if not self.reachable:
            raise _not_found(f"{repo_id} not found")
        return _Info()


@pytest.mark.parametrize(
    ("fake", "expected"),
    [
        (_StatusApi(reachable=True), "remote    2 files at abc123 (private)"),
        (_StatusApi(reachable=False), "remote    unavailable (RepositoryNotFoundError)"),
    ],
)
def test_status_reports_visibility_and_remote_only_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake: _StatusApi,
    expected: str,
) -> None:
    monkeypatch.setattr(commands, "read_token", lambda: "hf_token")
    monkeypatch.setattr(commands, "resolve_api", lambda **_: fake)
    staging = tmp_path / "staging"
    staging.mkdir()

    status = commands.main(
        [
            *("status", "--repo", "lab/repo", "--staging", str(staging)),
            *("--state", str(tmp_path / "upload_state.json")),
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert "token     maintainer (orgs: lab)" in output
    assert expected in output
    if fake.reachable:
        assert "figures/fig_old.pdf" in output

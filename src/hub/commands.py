"""Publish the corpora, outputs and figures to the Hugging Face Hub, or pull them back.

Run from the project root, usually through ``scripts/09_hub.sh``:
    uv run python -m src.hub.commands status
    uv run python -m src.hub.commands build
    uv run python -m src.hub.commands push --dry-run
    uv run python -m src.hub.commands pull --groups corpora
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx

from ..models import PROJECT_ROOT
from . import DATASET_REPO_ID, REPO_TYPE, dataset_url
from .auth import LOGIN_HINT, describe_identity, read_token, resolve_api
from .cards import write_dataset_card
from .datasets import build_staging, dataset_chunks, dataset_files
from .fetch import GROUPS, pull
from .push import UploadState, delete_paths, remote_only_paths, upload_chunks, upload_files

HUB_ROOT = PROJECT_ROOT / "hub"
DEFAULT_STAGING = HUB_ROOT / "datasets"
DEFAULT_DOWNLOADS = HUB_ROOT / "downloads"
DEFAULT_STATE = HUB_ROOT / "upload_state.json"
REMOTE_ONLY_SHOWN = 10


def _report_remote(arguments: argparse.Namespace) -> None:
    try:
        api = resolve_api()
    except SystemExit as error:
        print(f"token     rejected ({error})")
        return
    print(f"token     {describe_identity(api)}")
    try:
        info = api.dataset_info(arguments.repo)
    except (httpx.HTTPError, OSError) as error:
        print(f"remote    unavailable ({type(error).__name__})")
        return
    remote = [sibling.rfilename for sibling in info.siblings or []]
    visibility = "private" if info.private else "public"
    print(f"remote    {len(remote)} files at {info.sha} ({visibility})")
    if arguments.staging.is_dir():
        stale = remote_only_paths(remote, arguments.staging)
        print(f"          {len(stale)} not in the staging tree (push --prune deletes them)")
        for path in stale[:REMOTE_ONLY_SHOWN]:
            print(f"            {path}")


def command_status(arguments: argparse.Namespace) -> int:
    """Report the token, the remote repository, the staging tree and recorded uploads.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        Process exit status.
    """
    print(f"repo      {arguments.repo}")
    print(f"url       {dataset_url(arguments.repo)}")
    if read_token() is None:
        print("token     absent (uv run hf auth login)")
    else:
        _report_remote(arguments)
    if arguments.staging.is_dir():
        chunks = dataset_chunks(arguments.staging)
        files = dataset_files(arguments.staging)
        print(f"staging   {arguments.staging}: {len(chunks)} chunks, {len(files)} top-level files")
    else:
        print("staging   not built (./scripts/09_hub.sh build)")
    state = UploadState(arguments.state)
    print(f"recorded  {state.count(arguments.repo)} uploads in {arguments.state}")
    return 0


def command_build(arguments: argparse.Namespace) -> int:
    """Stage the published files and write the dataset card.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        Process exit status: 1 when a legacy T=0 copy differs from its standard corpus.
    """
    report = build_staging(arguments.root, arguments.staging)
    card = write_dataset_card(arguments.staging, arguments.repo)
    chunks = dataset_chunks(arguments.staging)
    files = dataset_files(arguments.staging)
    print(f"card      {card}")
    print(f"layout    {len(chunks)} chunks, {len(files)} top-level files")
    return 1 if report.unpublished else 0


def command_push(arguments: argparse.Namespace) -> int:
    """Build, then upload every changed chunk and top-level file.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        Process exit status.
    """
    status = command_build(arguments)
    if status:
        return status
    dry_run: bool = arguments.dry_run
    prune: bool = arguments.prune
    staging: Path = arguments.staging
    api = None
    if not dry_run:
        api = resolve_api(write=True)
        api.create_repo(arguments.repo, repo_type=REPO_TYPE, exist_ok=True)
    elif prune:
        api = resolve_api()
    state = UploadState(arguments.state)
    force: bool = arguments.force
    chunks = upload_chunks(
        api, arguments.repo, staging, dataset_chunks(staging), state, dry_run=dry_run, force=force
    )
    files = upload_files(
        api, arguments.repo, staging, dataset_files(staging), state, dry_run=dry_run, force=force
    )
    deleted = 0
    if api is not None:
        remote = api.list_repo_files(arguments.repo, repo_type=REPO_TYPE)
        stale = remote_only_paths(remote, staging)
        if prune:
            deleted = delete_paths(api, arguments.repo, stale, state, dry_run=dry_run)
        elif stale:
            print(f"kept      {len(stale)} remote files missing locally (--prune deletes them)")
    verb = "would upload" if dry_run else "uploaded"
    print(
        f"\n{verb} {chunks.uploaded} of {chunks.total} chunks and "
        f"{files.uploaded} of {files.total} top-level files"
    )
    if prune:
        print(f"{'would delete' if dry_run else 'deleted'} {deleted} remote files")
    if dry_run:
        print("dry run: nothing was uploaded or deleted")
    print(dataset_url(arguments.repo))
    return 0


def command_pull(arguments: argparse.Namespace) -> int:
    """Download the selected groups and fill the working copy's missing files.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        Process exit status.
    """
    try:
        report = pull(
            arguments.repo,
            arguments.downloads,
            arguments.root,
            groups=arguments.groups,
            revision=arguments.revision,
            overwrite=arguments.overwrite,
        )
    except (httpx.HTTPError, OSError) as error:
        print(f"pull failed: {error}", file=sys.stderr)
        print(f"A private dataset needs a token with read access.\n{LOGIN_HINT}", file=sys.stderr)
        return 1
    print(f"revision  {report.revision}")
    for label, paths in (("placed", report.placed), ("replaced", report.replaced)):
        for path in paths:
            print(f"{label:<9} {path}")
    for path in report.kept:
        print(f"kept      {path}  (differs; --overwrite replaces it)")
    print(
        f"\n{len(report.placed)} placed, {len(report.replaced)} replaced, "
        f"{len(report.kept)} kept, {len(report.identical)} already identical"
    )
    print(f"downloads {arguments.downloads} (remove it to reclaim the space)")
    return 0


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", default=DATASET_REPO_ID, help="dataset repository id")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="repository root")
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING, help="staging tree")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Parser with the ``status``, ``build``, ``push`` and ``pull`` actions.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    actions = parser.add_subparsers(dest="action", required=True)
    handlers: dict[str, tuple[str, Callable[[argparse.Namespace], int]]] = {
        "status": ("token, remote repository, staging tree and recorded uploads", command_status),
        "build": (
            "link the published files into the staging tree and write the card",
            command_build,
        ),
        "push": ("build, then upload every changed chunk and top-level file", command_push),
        "pull": ("download published files and fill the missing ones", command_pull),
    }
    for name, (summary, handler) in handlers.items():
        action = actions.add_parser(name, help=summary, description=summary)
        _common_arguments(action)
        action.set_defaults(handler=handler)
        if name in {"status", "push"}:
            action.add_argument("--state", type=Path, default=DEFAULT_STATE, help="upload record")
        if name == "push":
            action.add_argument("--dry-run", action="store_true", help="upload nothing")
            action.add_argument("--force", action="store_true", help="ignore the upload record")
            action.add_argument(
                "--prune",
                action="store_true",
                help="delete remote files the staging tree does not have, after listing them",
            )
        if name == "pull":
            action.add_argument("--downloads", type=Path, default=DEFAULT_DOWNLOADS)
            action.add_argument("--groups", nargs="+", choices=list(GROUPS), help="default: all")
            action.add_argument("--revision", help="branch, tag or commit to download")
            action.add_argument(
                "--overwrite", action="store_true", help="replace local files that differ"
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit status.
    """
    arguments = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = arguments.handler
    return handler(arguments)


if __name__ == "__main__":
    sys.exit(main())

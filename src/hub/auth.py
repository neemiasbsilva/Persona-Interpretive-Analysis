"""Hugging Face token resolution, checked once and reported plainly."""

from __future__ import annotations

import httpx
from huggingface_hub import HfApi, get_token

LOGIN_HINT = (
    "Authenticate with the Hugging Face Hub first:\n"
    "  uv run hf auth login    stores the token in the huggingface_hub token file\n"
    "  HF_TOKEN                in CI, set from a secret; push needs write scope"
)


def read_token() -> str | None:
    """Return the token from ``HF_TOKEN`` or from a stored ``hf auth login``.

    Returns:
        The token, or ``None`` when neither source has one.
    """
    return get_token()


def resolve_api(*, write: bool = False) -> HfApi:
    """Build an authenticated client after the Hub has confirmed its token.

    Args:
        write: Refuse a token whose scope is read-only.

    Returns:
        A client carrying the resolved token.

    Raises:
        SystemExit: When no token is found, the Hub rejects it, or ``write`` is
            requested with a read-scoped token.
    """
    token = read_token()
    if token is None:
        raise SystemExit(LOGIN_HINT)
    api = HfApi(token=token)
    try:
        identity = api.whoami(cache=True)
    except (httpx.HTTPError, OSError) as error:
        raise SystemExit(f"Hugging Face token rejected: {error}") from error
    access = (identity.get("auth") or {}).get("accessToken") or {}
    if write and access.get("role") == "read":
        raise SystemExit("Token has 'read' scope; a write-scoped token is required.")
    return api


def describe_identity(api: HfApi) -> str:
    """Name the account behind a client, with its organisations.

    Args:
        api: Authenticated client.

    Returns:
        ``"user (orgs: a, b)"``, or the bare user name when it has no organisations.
    """
    identity = api.whoami(cache=True)
    orgs = [entry["name"] for entry in identity.get("orgs", [])]
    suffix = f" (orgs: {', '.join(orgs)})" if orgs else ""
    return f"{identity.get('name', 'unknown')}{suffix}"

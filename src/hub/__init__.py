"""The UrbanPersona-120K-Interpretive dataset on the Hugging Face Hub, in both directions.

Publishing follows MLLMs-persona-evaluation, which publishes UrbanPersona-60K the same
way, with the content digests and per-chunk remote deletions of machine-bias-reproduction:

- :mod:`src.hub.datasets`    what is published, and the symlink staging tree
- :mod:`src.hub.provenance`  where each corpus came from, and the card's fixed prose
- :mod:`src.hub.cards`       the dataset card, generated from what was staged
- :mod:`src.hub.push`        chunked, resumable uploads
- :mod:`src.hub.fetch`       pulling published files back into the working copy
- :mod:`src.hub.auth`        token resolution
- :mod:`src.hub.commands`    status, build, push and pull, wrapped by scripts/09_hub.sh
"""

DATASET_REPO_ID = "MInDS-lab-UTFPR/UrbanPersona-120K-Interpretive"
REPO_TYPE = "dataset"
BYTES_PER_MB = 1_000_000


def megabytes(size_bytes: int) -> str:
    """Format a size in decimal megabytes, the unit the Hub displays.

    Args:
        size_bytes: Size in bytes.

    Returns:
        A label such as ``"91.7 MB"``.
    """
    return f"{size_bytes / BYTES_PER_MB:,.1f} MB"


def dataset_url(repo_id: str = DATASET_REPO_ID) -> str:
    """Return the browser URL of a dataset repository.

    Args:
        repo_id: Hub dataset id such as ``"org/name"``.

    Returns:
        The ``https://huggingface.co/datasets/...`` URL.
    """
    return f"https://huggingface.co/datasets/{repo_id}"

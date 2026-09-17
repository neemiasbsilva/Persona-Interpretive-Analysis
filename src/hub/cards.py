r"""The dataset card, generated from what was staged.

The card is rewritten on every build. Its configs reference only staged corpus files,
and every record count, date span and inventory size is read from the staging tree, so
the published card cannot drift from the published files.

Configs are conditions and splits are models. A condition fixes the record schema:
failure records add ``error`` and no-persona runs carry an empty ``raw_demographics``,
so one config per model would mix three schemas across its splits. Split names are the
model directories with non-word characters mapped to underscores, because the Hub
requires them to match ``\w+(\.\w+)*``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..models import PAPER_MODELS, T0_MODELS
from . import DATASET_REPO_ID, megabytes
from .datasets import CARD_NAME, CORPUS_FILES, MODELS, PUBLISHED_ROOTS, walk_files
from .provenance import (
    ACKNOWLEDGMENTS_SECTION,
    ATTEMPTS_LEAD,
    CITATION_SECTION,
    IMAGES_SECTION,
    INTRO,
    LICENSE_SECTION,
    OUTPUTS_SECTION,
    SCHEMA_NOTES,
    SCHEMA_ROWS,
    SPAN_SEPARATOR,
    SUBSTITUTIONS,
    USAGE_SECTION,
    VARIATION_NOTES,
    VARIATIONS,
    VARIATIONS_LEAD,
)

DEFAULT_CONFIG = "baseline"
SIZE_CATEGORIES: tuple[tuple[int, str], ...] = (
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
    (10_000_000, "1M<n<10M"),
)
TASK_CATEGORIES: tuple[str, ...] = ("image-to-text", "image-classification", "text-classification")
TAGS: tuple[str, ...] = (
    "urban-perception",
    "persona",
    "llm-agents",
    "multimodal",
    "vision-language-models",
    "sentiment-analysis",
    "perceptsent",
    "synthetic",
)


@dataclass(frozen=True, slots=True)
class Split:
    """One model's corpus file within a config.

    Attributes:
        name: Hub split name derived from the model directory.
        model: Model directory under ``data/``.
        path: Repository path of the JSONL file.
        records: Non-blank lines in the file.
        first: Earliest ``timestamp_utc``, or ``""`` when no record carries one.
        last: Latest ``timestamp_utc``, or ``""`` when no record carries one.
    """

    name: str
    model: str
    path: str
    records: int
    first: str
    last: str


@dataclass(frozen=True, slots=True)
class Config:
    """One condition, with a split per model that has it staged.

    Attributes:
        name: Hub config name.
        splits: Its splits, in model order.
    """

    name: str
    splits: tuple[Split, ...]


def split_name(model: str) -> str:
    """Map a model directory to a valid Hub split name.

    Args:
        model: Model directory such as ``"gemma-4-E4B-it_t01"``.

    Returns:
        The directory with every non-word character replaced by an underscore.
    """
    return re.sub(r"\W", "_", model)


def config_name(model: str, condition: str) -> str:
    """Name the config a model's corpus belongs to.

    Args:
        model: Model directory.
        condition: Key of :data:`src.hub.datasets.CORPUS_FILES`.

    Returns:
        The condition, prefixed with ``t0_`` for the T=0 ablation models.
    """
    return f"t0_{condition}" if model in T0_MODELS else condition


def read_split(staging: Path, model: str, condition: str) -> Split:
    """Count the records of a staged corpus file and the span of its timestamps.

    Args:
        staging: Staging tree root.
        model: Model directory.
        condition: Key of :data:`src.hub.datasets.CORPUS_FILES`.

    Returns:
        The split describing that file.
    """
    path = f"data/{model}/{CORPUS_FILES[condition]}"
    records = 0
    stamps: list[str] = []
    with (staging / path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            records += 1
            stamp = json.loads(line).get("timestamp_utc")
            if isinstance(stamp, str) and stamp:
                stamps.append(stamp)
    first, last = (min(stamps), max(stamps)) if stamps else ("", "")
    return Split(split_name(model), model, path, records, first, last)


def dataset_configs(staging: Path) -> tuple[Config, ...]:
    """Build the configs from the corpus files present in the staging tree.

    Args:
        staging: Staging tree root.

    Returns:
        Paper-model configs first, then the T=0 ones, each in condition order.
    """
    grouped: dict[str, list[Split]] = {}
    for models in (PAPER_MODELS, T0_MODELS):
        for condition, filename in CORPUS_FILES.items():
            for model in models:
                if (staging / "data" / model / filename).is_file():
                    split = read_split(staging, model, condition)
                    grouped.setdefault(config_name(model, condition), []).append(split)
    return tuple(Config(name, tuple(splits)) for name, splits in grouped.items())


def size_category(records: int) -> str:
    """Return the Hub size category for a record count.

    Args:
        records: Total records across every config.

    Returns:
        A ``size_categories`` value such as ``"100K<n<1M"``.
    """
    for bound, label in SIZE_CATEGORIES:
        if records < bound:
            return label
    return "n>10M"


def _total_records(configs: Sequence[Config]) -> int:
    return sum(split.records for config in configs for split in config.splits)


def front_matter(configs: Sequence[Config], repo_id: str = DATASET_REPO_ID) -> str:
    """Render the card's YAML metadata block.

    Args:
        configs: Configs to declare.
        repo_id: Dataset repository, whose name becomes ``pretty_name``.

    Returns:
        The block, fences included.
    """
    names = [config.name for config in configs]
    default = DEFAULT_CONFIG if DEFAULT_CONFIG in names else (names[0] if names else "")
    lines = [
        "---",
        f"pretty_name: {repo_id.rsplit('/', maxsplit=1)[-1]}",
        "license: cc-by-4.0",
        "language:",
        "  - en",
        "task_categories:",
        *(f"  - {category}" for category in TASK_CATEGORIES),
        "tags:",
        *(f"  - {tag}" for tag in TAGS),
        "size_categories:",
        f"  - {size_category(_total_records(configs))}",
        "configs:",
    ]
    for config in configs:
        lines.append(f"  - config_name: {config.name}")
        if config.name == default:
            lines.append("    default: true")
        lines.append("    data_files:")
        for split in config.splits:
            lines.extend((f"      - split: {split.name}", f"        path: {split.path}"))
    lines.append("---")
    return "\n".join(lines)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _span(first: str, last: str) -> str:
    start, end = first[:10], last[:10]
    if not start:
        return "—"
    return start if start == end else f"{start}{SPAN_SEPARATOR}{end}"


def _variations_table(configs: Sequence[Config]) -> str:
    splits = [split for config in configs for split in config.splits]
    rows: list[tuple[str, ...]] = []
    for model in MODELS:
        own = [split for split in splits if split.model == model and split.first]
        if not any(split.model == model for split in splits):
            continue
        variation = VARIATIONS[model]
        first = min((split.first for split in own), default="")
        last = max((split.last for split in own), default="")
        rows.append(
            (
                f"`{split_name(model)}`",
                variation.model_id,
                variation.backend,
                variation.decoding,
                variation.design,
                _span(first, last),
                variation.source,
            )
        )
    header = ("split", "model", "backend", "decoding", "design", "generated (UTC)", "source")
    return _table(header, rows)


def _configs_table(configs: Sequence[Config]) -> str:
    rows: list[tuple[str, ...]] = []
    for config in configs:
        label = f"`{config.name}`" + (" *(default)*" if config.name == DEFAULT_CONFIG else "")
        rows.extend(
            (
                label,
                f"`{split.name}`",
                f"{split.records:,}",
                f"`{split.path}`",
                _span(split.first, split.last),
            )
            for split in config.splits
        )
    return _table(("config", "split", "records", "file", "generated (UTC)"), rows)


def _attempts_sentence(configs: Sequence[Config]) -> str:
    counts = {
        (config.name, split.model): split.records for config in configs for split in config.splits
    }
    parts: list[str] = []
    total = 0
    for model in PAPER_MODELS:
        annotated = counts.get(("baseline", model))
        failed = counts.get(("failures", model), 0)
        if annotated is None:
            continue
        total += annotated + failed
        parts.append(f"`{split_name(model)}` {annotated:,} + {failed:,} = {annotated + failed:,}")
    if not parts:
        return ""
    return (
        f"{ATTEMPTS_LEAD}: {'; '.join(parts)}; {total:,} in all, "
        f"over {_total_records(configs):,} records across every config."
    )


def _inventory_table(staging: Path) -> str:
    rows: list[tuple[str, ...]] = []
    for root in PUBLISHED_ROOTS:
        files = list(walk_files(staging / root))
        size = megabytes(sum(path.stat().st_size for path in files))
        rows.append((f"`{root}/`", f"{len(files):,}", size))
    return _table(("folder", "files", "size"), rows)


def dataset_card(staging: Path, repo_id: str = DATASET_REPO_ID) -> str:
    """Render the dataset card for a staging tree.

    Args:
        staging: Staging tree root.
        repo_id: Dataset repository the card describes.

    Returns:
        The card, YAML metadata included.
    """
    configs = dataset_configs(staging)
    schema = _table(("field", "type", "description"), SCHEMA_ROWS)
    sections = [
        front_matter(configs, repo_id),
        INTRO,
        VARIATIONS_LEAD,
        _variations_table(configs),
        VARIATION_NOTES,
        "## Configs and splits",
        _configs_table(configs),
        _attempts_sentence(configs),
        "## Record schema\n\nEvery JSONL line is one annotation attempt:",
        schema,
        SCHEMA_NOTES,
        OUTPUTS_SECTION,
        "## Inventory",
        _inventory_table(staging),
        IMAGES_SECTION,
        USAGE_SECTION,
        CITATION_SECTION,
        ACKNOWLEDGMENTS_SECTION,
        LICENSE_SECTION,
    ]
    card = "\n\n".join(section for section in sections if section) + "\n"
    substitutions = {
        **SUBSTITUTIONS,
        "<<NAME>>": repo_id.rsplit("/", maxsplit=1)[-1],
        "<<REPO_ID>>": repo_id,
    }
    for placeholder, value in substitutions.items():
        card = card.replace(placeholder, value)
    return card


def write_dataset_card(staging: Path, repo_id: str = DATASET_REPO_ID) -> Path:
    """Write the dataset card to the root of the staging tree.

    Args:
        staging: Staging tree root, created when absent.
        repo_id: Dataset repository the card describes.

    Returns:
        Path of the written card.
    """
    staging.mkdir(parents=True, exist_ok=True)
    path = staging / CARD_NAME
    path.write_text(dataset_card(staging, repo_id), encoding="utf-8")
    return path

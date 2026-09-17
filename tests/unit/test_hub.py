"""Cover what reaches the Hub: the publish rules, the staging tree and the dataset card.

Every test builds a small checkout under ``tmp_path`` from the real model lists and runs
without the network, so the rules pinned here are the ones a real push applies.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub.repocard import metadata_load

from src.hub import cards, datasets, push
from src.hub.provenance import VARIATIONS
from src.models import PAPER_MODELS, T0_MODELS


def _record(stamp: str, condition: str = "baseline") -> dict[str, object]:
    return {
        "annotation_id": f"p00000000_img_x_{condition}",
        "persona_id": "p",
        "image_id": "x",
        "condition": condition,
        "timestamp_utc": stamp,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _write(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_checkout(root: Path) -> Path:
    """Lay out a miniature repository with every kind of path the publish rule sees.

    Args:
        root: Directory to build the checkout in.

    Returns:
        The checkout root.
    """
    for model in PAPER_MODELS:
        data = root / "data" / model
        _write_jsonl(
            data / "annotations_baseline.jsonl",
            [_record("2026-07-21T06:00:00Z"), _record("2026-07-22T10:00:00Z")],
        )
        _write_jsonl(data / "annotation_failures.jsonl", [_record("2026-07-21T07:00:00Z")])
        _write_jsonl(
            data / "annotations_no_persona_think.jsonl",
            [_record("2026-07-22T01:00:00Z", "no_persona_think")],
        )
        _write_jsonl(
            data / "annotations_no_persona_no_think.jsonl",
            [_record("2026-07-22T02:00:00Z", "no_persona_no_think")],
        )
        outputs = root / "outputs" / model
        _write(outputs / "within_cross_significance.csv", "cell,p\n1,0.01\n")
        _write(outputs / "caption_embeddings.npy", "embedding bytes")
        _write(outputs / "logs" / "run_1.log", "log line\n")
        _write(outputs / "bertopic_captions.pkl", "pickle")
        _write(root / "figures" / model / "fig_within_cross.pdf", "%PDF")
        _write(root / "figures" / model / "fig_within_cross.png", "PNG")
    for model in T0_MODELS:
        data = root / "data" / model
        baseline = [_record("2026-07-24T00:36:00Z")]
        _write_jsonl(data / "annotations_baseline.jsonl", baseline)
        _write_jsonl(data / "t0_economic_status_annotations_baseline.jsonl", baseline)
        _write(data / "t0_economic_status_annotation_failures.jsonl", "")
        _write_jsonl(
            data / "annotations_no_persona_no_think.jsonl",
            [_record("2026-07-24T19:00:00Z", "no_persona_no_think")],
        )
        _write(root / "outputs" / model / "t0_econ_summary.csv", "level,sim\n")
    _write(root / "data" / "perceptsent-raw" / ".gitkeep", "")
    _write(root / "outputs" / "heatmap_values" / "manifest.csv", "model,csv\n")
    _write(root / "outputs" / "cross_model_correlation.csv", "modality,r\n")
    _write(root / "outputs" / "t0_ablation_table.tex", "\\begin{tabular}")
    _write(root / "outputs" / "scratch" / "notes.csv", "a\n")
    _write(root / "figures" / "fig_t0_ablation.pdf", "%PDF")
    _write(root / "figures" / "example_scene.jpg", "JPEG")
    return root


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    return make_checkout(tmp_path / "repo")


def _staged(checkout: Path, staging: Path) -> set[str]:
    datasets.build_staging(checkout, staging, on_event=lambda _: None)
    return {path.relative_to(staging).as_posix() for path in datasets.walk_files(staging)}


def test_each_corpus_is_staged_once_under_its_standard_name(checkout: Path, tmp_path: Path) -> None:
    staged = _staged(checkout, tmp_path / "staging")

    assert "data/gemma4-t0/annotations_baseline.jsonl" in staged
    assert not [path for path in staged if datasets.LEGACY_PREFIX in path]
    assert sum(path.startswith("data/") for path in staged) == 12


def test_zero_byte_files_are_never_staged(checkout: Path, tmp_path: Path) -> None:
    _write(checkout / "data" / "qwen-vl" / "annotation_failures.jsonl", "")

    staged = _staged(checkout, tmp_path / "staging")

    assert "data/qwen-vl/annotation_failures.jsonl" not in staged
    assert "data/qwen-vl/annotations_baseline.jsonl" in staged


def test_logs_latex_pickles_and_images_stay_out_while_embeddings_go_in(
    checkout: Path, tmp_path: Path
) -> None:
    staged = _staged(checkout, tmp_path / "staging")

    assert "outputs/qwen-vl/caption_embeddings.npy" in staged
    assert "outputs/cross_model_correlation.csv" in staged
    assert "outputs/heatmap_values/manifest.csv" in staged
    assert not [path for path in staged if "/logs/" in path]
    assert not [path for path in staged if path.endswith((".tex", ".pkl", ".jpg"))]


def test_directories_outside_the_known_models_are_not_published(
    checkout: Path, tmp_path: Path
) -> None:
    staged = _staged(checkout, tmp_path / "staging")

    assert not [path for path in staged if "scratch" in path or "perceptsent" in path]


def test_staged_entries_are_file_symlinks_inside_real_directories(
    checkout: Path, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    datasets.build_staging(checkout, staging, on_event=lambda _: None)

    corpus = staging / "data" / "qwen-vl" / "annotations_baseline.jsonl"
    assert corpus.is_symlink()
    assert corpus.resolve() == (checkout / "data" / "qwen-vl" / "annotations_baseline.jsonl")
    assert not corpus.parent.is_symlink()


def test_a_rebuild_relinks_nothing_and_prunes_what_is_no_longer_published(
    checkout: Path, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    first = datasets.build_staging(checkout, staging, on_event=lambda _: None)
    (checkout / "outputs" / "heatmap_values" / "manifest.csv").unlink()

    second = datasets.build_staging(checkout, staging, on_event=lambda _: None)

    assert first.linked == first.files
    assert second.linked == 0
    assert second.pruned == 1
    assert not (staging / "outputs" / "heatmap_values").exists()


def test_a_t0_copy_that_differs_from_its_standard_corpus_is_reported(
    checkout: Path, tmp_path: Path
) -> None:
    legacy = checkout / "data" / "qwen-vl-t0" / "t0_economic_status_annotations_baseline.jsonl"
    _write_jsonl(legacy, [_record("2026-07-25T00:00:00Z")])

    report = datasets.build_staging(checkout, tmp_path / "staging", on_event=lambda _: None)

    assert report.unpublished == ("data/qwen-vl-t0/t0_economic_status_annotations_baseline.jsonl",)


def test_chunks_are_model_directories_and_shallow_files_stay_top_level(
    checkout: Path, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    datasets.build_staging(checkout, staging, on_event=lambda _: None)
    cards.write_dataset_card(staging)

    chunks = [chunk.relative_to(staging).as_posix() for chunk in datasets.dataset_chunks(staging)]
    loose = [path.relative_to(staging).as_posix() for path in datasets.dataset_files(staging)]

    assert "data/qwen-vl" in chunks
    assert "outputs/heatmap_values" in chunks
    assert "figures/gemma-4-E4B-it_t01" in chunks
    assert not [chunk for chunk in chunks if chunk in datasets.PUBLISHED_ROOTS]
    assert loose == [
        "README.md",
        "figures/fig_t0_ablation.pdf",
        "outputs/cross_model_correlation.csv",
    ]


def test_download_bookkeeping_is_never_offered_for_upload(checkout: Path, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    datasets.build_staging(checkout, staging, on_event=lambda _: None)
    _write(staging / ".cache" / "huggingface" / "download" / "README.md.metadata", "etag\n")

    loose = [path.relative_to(staging).as_posix() for path in datasets.dataset_files(staging)]

    assert not [path for path in loose if ".cache" in path]


def test_every_known_model_has_a_valid_unique_split_name() -> None:
    names = [cards.split_name(model) for model in (*PAPER_MODELS, *T0_MODELS)]

    assert all(re.fullmatch(r"\w+(\.\w+)*", name) for name in names)
    assert len(set(names)) == len(names)


def test_every_known_model_has_a_provenance_entry() -> None:
    assert set(VARIATIONS) == {*PAPER_MODELS, *T0_MODELS}


def test_card_front_matter_parses_and_references_only_staged_files(
    checkout: Path, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    datasets.build_staging(checkout, staging, on_event=lambda _: None)

    metadata = metadata_load(cards.write_dataset_card(staging, "lab/repo"))

    assert metadata is not None
    assert metadata["license"] == "cc-by-4.0"
    assert metadata["pretty_name"] == "repo"
    configs: list[dict[str, Any]] = metadata["configs"]
    assert [config["config_name"] for config in configs if config.get("default")] == ["baseline"]
    paths = [entry["path"] for config in configs for entry in config["data_files"]]
    assert len(paths) == 12
    assert all((staging / path).is_file() for path in paths)


def test_card_counts_records_and_dates_from_the_staged_corpora(
    checkout: Path, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    datasets.build_staging(checkout, staging, on_event=lambda _: None)

    card = cards.dataset_card(staging, "lab/repo")

    assert "| `baseline` *(default)* | `qwen_vl` | 2 |" in card
    assert "2026-07-21 – 2026-07-22" in card  # noqa: RUF001
    assert "`qwen_vl` 2 + 1 = 3" in card
    assert "https://huggingface.co/datasets/Neemias/UrbanPersona-60K" in card
    assert "@inproceedings{silva2026persona," in card
    assert "{\\&} NLP: Diversity-aware, Sociotechnical, Responsible Alignment (PANDORA)}" in card
    assert "INCT TILD-IAR (proc. 408490/2024-1)" in card
    assert "<<" not in card


def test_card_omits_a_config_whose_corpus_is_not_staged(checkout: Path, tmp_path: Path) -> None:
    for model in PAPER_MODELS:
        (checkout / "data" / model / "annotation_failures.jsonl").unlink()
    staging = tmp_path / "staging"
    datasets.build_staging(checkout, staging, on_event=lambda _: None)

    names = [config.name for config in cards.dataset_configs(staging)]

    assert "failures" not in names
    assert names[0] == "baseline"


def test_folder_digest_follows_the_staging_symlinks(checkout: Path, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    datasets.build_staging(checkout, staging, on_event=lambda _: None)
    chunk = staging / "outputs" / "qwen-vl"
    before = push.folder_digest(chunk)

    _write(checkout / "outputs" / "qwen-vl" / "within_cross_significance.csv", "cell,p\n1,0.02\n")

    assert push.folder_digest(chunk) != before


def test_conditions_are_configs_and_the_t0_models_get_their_own(
    checkout: Path, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    datasets.build_staging(checkout, staging, on_event=lambda _: None)

    configs = {
        config.name: [split.name for split in config.splits]
        for config in cards.dataset_configs(staging)
    }

    assert configs == {
        "baseline": ["qwen_vl", "gemma_4_E4B_it_t01"],
        "failures": ["qwen_vl", "gemma_4_E4B_it_t01"],
        "no_persona_think": ["qwen_vl", "gemma_4_E4B_it_t01"],
        "no_persona_no_think": ["qwen_vl", "gemma_4_E4B_it_t01"],
        "t0_baseline": ["qwen_vl_t0", "gemma4_t0"],
        "t0_no_persona_no_think": ["qwen_vl_t0", "gemma4_t0"],
    }
    assert next(iter(configs)) == cards.DEFAULT_CONFIG


def test_card_sizes_are_decimal_megabytes(checkout: Path, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    _write(checkout / "outputs" / "qwen-vl" / "caption_embeddings.npy", "x" * 1_500_000)
    datasets.build_staging(checkout, staging, on_event=lambda _: None)

    card = cards.dataset_card(staging, "lab/repo")

    assert "| `outputs/` |" in card
    assert "1.5 MB |" in card

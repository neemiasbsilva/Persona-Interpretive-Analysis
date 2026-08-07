"""Cover model identity and the per-model filesystem layout."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import PERSONA_MODEL_ENV, Settings, active_settings, settings_for
from src.models import (
    DEFAULT_MODEL,
    PAPER_MODELS,
    T0_MODELS,
    annotations_path,
    data_dir,
    display_name,
    figures_dir,
    outputs_dir,
)


def test_paper_and_t0_model_sets_are_distinct():
    assert len(PAPER_MODELS) == 2
    assert len(T0_MODELS) == 2
    assert not set(PAPER_MODELS) & set(T0_MODELS)
    assert DEFAULT_MODEL in PAPER_MODELS


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("qwen-vl", "Qwen3-VL"),
        ("qwen-vl-t0", "Qwen3-VL"),
        ("gemma-4-E4B-it_t01", "Gemma4"),
        ("gemma4-t0", "Gemma4"),
        ("QWEN-VL", "Qwen3-VL"),
        ("mystery-model", "mystery-model"),
    ],
)
def test_display_name_maps_directory_ids_to_paper_labels(model, expected):
    assert display_name(model) == expected


def test_every_known_model_has_a_display_name():
    for model in (*PAPER_MODELS, *T0_MODELS):
        assert display_name(model) in {"Qwen3-VL", "Gemma4"}


def test_path_helpers_namespace_by_model(tmp_path):
    assert outputs_dir("m", tmp_path) == tmp_path / "outputs" / "m"
    assert figures_dir("m", tmp_path) == tmp_path / "figures" / "m"
    assert data_dir("m", tmp_path) == tmp_path / "data" / "m"
    assert annotations_path("m", tmp_path) == tmp_path / "data" / "m" / "annotations_baseline.jsonl"


def test_settings_derive_every_path_from_model_and_root(tmp_path):
    settings = settings_for("fixture-model", tmp_path)
    assert settings.data_dir == tmp_path / "data" / "fixture-model"
    assert settings.data_path.name == "annotations_baseline.jsonl"
    assert settings.np_think_path.name == "annotations_no_persona_think.jsonl"
    assert settings.np_nothink_path.name == "annotations_no_persona_no_think.jsonl"
    assert settings.outputs == tmp_path / "outputs" / "fixture-model"
    assert settings.figures == tmp_path / "figures" / "fixture-model"


def test_settings_agree_with_the_model_path_helpers(tmp_path):
    settings = settings_for("m", tmp_path)
    assert settings.outputs == outputs_dir("m", tmp_path)
    assert settings.figures == figures_dir("m", tmp_path)
    assert settings.data_path == annotations_path("m", tmp_path)


def test_settings_are_frozen(tmp_path):
    settings = settings_for("m", tmp_path)
    with pytest.raises(AttributeError):
        settings.model = "other"


def test_ensure_dirs_creates_only_output_and_figure_trees(tmp_path):
    settings = settings_for("m", tmp_path)
    assert not settings.outputs.exists()
    settings.ensure_dirs()
    assert settings.outputs.is_dir()
    assert settings.figures.is_dir()
    assert not settings.data_dir.exists()


def test_settings_do_not_touch_the_filesystem_on_construction(tmp_path):
    settings_for("never-created", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_active_settings_is_resolved_once_per_process():
    first = active_settings()
    original = os.environ.get(PERSONA_MODEL_ENV)
    os.environ[PERSONA_MODEL_ENV] = "changed-midway"
    try:
        assert active_settings() is first
    finally:
        if original is None:
            os.environ.pop(PERSONA_MODEL_ENV, None)
        else:
            os.environ[PERSONA_MODEL_ENV] = original


def test_active_settings_reads_the_environment_after_a_cache_clear():
    original = os.environ.get(PERSONA_MODEL_ENV)
    active_settings.cache_clear()
    try:
        os.environ[PERSONA_MODEL_ENV] = "explicit-model"
        assert active_settings().model == "explicit-model"

        active_settings.cache_clear()
        os.environ.pop(PERSONA_MODEL_ENV, None)
        assert active_settings().model == DEFAULT_MODEL
    finally:
        active_settings.cache_clear()
        if original is None:
            os.environ.pop(PERSONA_MODEL_ENV, None)
        else:
            os.environ[PERSONA_MODEL_ENV] = original
        active_settings()


def test_settings_default_to_the_real_repository_root():
    assert isinstance(settings_for("m"), Settings)
    assert (settings_for("m").project_root / "pyproject.toml").exists()
    assert isinstance(settings_for("m").outputs, Path)

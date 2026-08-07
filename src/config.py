"""Path resolution, model identifiers, and analysis hyperparameters.

The active corpus is chosen by the ``PERSONA_MODEL`` environment variable and
resolved exactly once per process, so a run cannot change corpus midway:

    PERSONA_MODEL=gemma-4-E4B-it_t01 uv run python -m src.generate_figures

The module-level constants below are that single resolution, kept for the many
call sites that import them directly. Code that needs a *different* model in the
same process, including tests, calls :func:`settings_for` instead of mutating the
environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .models import DEFAULT_MODEL
from .models import PROJECT_ROOT as _REPO_ROOT

PERSONA_MODEL_ENV = "PERSONA_MODEL"


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved filesystem layout for one annotating model.

    Attributes:
        model: Directory-style model identifier.
        project_root: Repository root all paths are resolved against.
    """

    model: str
    project_root: Path

    @property
    def data_dir(self) -> Path:
        """Return the corpus directory for this model."""
        return self.project_root / "data" / self.model

    @property
    def data_path(self) -> Path:
        """Return the baseline annotation corpus."""
        return self.data_dir / "annotations_baseline.jsonl"

    @property
    def np_think_path(self) -> Path:
        """Return the no-persona corpus generated with reasoning enabled."""
        return self.data_dir / "annotations_no_persona_think.jsonl"

    @property
    def np_nothink_path(self) -> Path:
        """Return the no-persona corpus generated with reasoning disabled."""
        return self.data_dir / "annotations_no_persona_no_think.jsonl"

    @property
    def outputs(self) -> Path:
        """Return the cached-artifact directory for this model."""
        return self.project_root / "outputs" / self.model

    @property
    def figures(self) -> Path:
        """Return the figure directory for this model."""
        return self.project_root / "figures" / self.model

    def ensure_dirs(self) -> None:
        """Create the output and figure directories if they do not exist."""
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)


def settings_for(model: str, project_root: Path | None = None) -> Settings:
    """Build a settings object for an arbitrary model.

    Args:
        model: Directory-style model identifier.
        project_root: Repository root; defaults to the real repository.

    Returns:
        A frozen settings object; no directories are created.
    """
    return Settings(model=model, project_root=project_root or _REPO_ROOT)


@lru_cache(maxsize=1)
def active_settings() -> Settings:
    """Return the settings selected by the environment, resolved once per process.

    Returns:
        Settings for ``$PERSONA_MODEL``, or for the default model when unset.
    """
    return settings_for(os.environ.get(PERSONA_MODEL_ENV, DEFAULT_MODEL))


_ACTIVE = active_settings()

MODEL = _ACTIVE.model
PROJECT_ROOT = _ACTIVE.project_root
DATA_DIR = _ACTIVE.data_dir
DATA_PATH = _ACTIVE.data_path
NP_THINK_PATH = _ACTIVE.np_think_path
NP_NOTHINK_PATH = _ACTIVE.np_nothink_path
FIGURES = _ACTIVE.figures
OUTPUTS = _ACTIVE.outputs

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

FONT_SCALE = 1.8

SENT_COLORS = {
    "Negative": "#d73027",
    "SlightlyNegative": "#fc8d59",
    "Neutral": "#aaaaaa",
    "SlightlyPositive": "#91cf60",
    "Positive": "#1a9850",
}
SENT_COLORS_3 = {k: SENT_COLORS[k] for k in ("Negative", "Neutral", "Positive")}

BERTOPIC_MIN_CLUSTER_SIZE = 80
BERTOPIC_N_COMPONENTS = 5
BERTOPIC_N_NEIGHBORS = 15
BERTOPIC_TOP_N_WORDS = 10
BERTOPIC_MIN_DF = 10

EMBED_BATCH_SIZE = 256

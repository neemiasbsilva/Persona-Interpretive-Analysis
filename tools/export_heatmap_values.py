"""Export the exact numeric matrices behind every heatmap in the paper.

Run from the repository root:

    uv run python tools/export_heatmap_values.py

The exporter reads the same cached arrays and topic artifacts as the figure
generator. It does not infer values from rendered PDF or PNG files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loading import create_profiles, load_annotations, parse_demographics
from src.models import PAPER_MODELS, PROJECT_ROOT, display_name
from src.topic_modeling import load_topic_labels

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "heatmap_values"
FLOAT_FORMAT = "%.17g"


@dataclass(frozen=True, slots=True)
class Heatmap:
    """One numeric matrix and its paper-facing metadata."""

    model: str
    name: str
    figure: str
    metric: str
    matrix: pd.DataFrame

    @property
    def filename(self) -> str:
        """Return the model-scoped CSV filename."""
        return f"{self.model}__{self.name}.csv"


def _topic_persona_matrix(model: str) -> pd.DataFrame:
    """Reconstruct the topic-by-persona matrix used by generate_figures.py."""
    model_outputs = PROJECT_ROOT / "outputs" / model
    model_data = PROJECT_ROOT / "data" / model
    topic_labels = load_topic_labels(model_outputs / "topic_labels_just.csv")

    annotations = create_profiles(
        parse_demographics(load_annotations(model_data / "annotations_baseline.jsonl"))
    )
    topic_assignments = pd.read_csv(model_outputs / "annotations_with_topics.csv")
    merged = topic_assignments.merge(
        annotations[["annotation_id", "profile_abbr"]],
        on="annotation_id",
        how="left",
    ).dropna(subset=["profile_abbr"])

    labeled_topics = list(topic_labels)
    counts = (
        merged[merged["just_topic"].isin(labeled_topics)]
        .groupby(["just_topic", "profile_abbr"])
        .size()
        .reset_index(name="count")
    )
    matrix = counts.pivot_table(
        index="just_topic",
        columns="profile_abbr",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).fillna(0)
    matrix = matrix.div(matrix.sum(axis=0), axis=1)
    matrix.index = pd.Index(
        [topic_labels[int(topic)] for topic in matrix.index],
        name="topic",
    )
    matrix.columns.name = "persona"
    return matrix


def _profile_labels(model: str) -> tuple[list[str], pd.DataFrame]:
    """Load the matrix order and return abbreviated labels plus a label key."""
    path = PROJECT_ROOT / "outputs" / model / "ic_profile_sim_labels.csv"
    full_labels = pd.read_csv(path)["profile"].astype(str)

    def abbreviate(profile: str) -> str:
        parts = [part.strip() for part in profile.split("/")]
        return f"{parts[0][0]}/{parts[1][:4]}/{parts[2][:4]}/{parts[3][:4]}"

    short_labels = [abbreviate(profile) for profile in full_labels]
    key = pd.DataFrame(
        {
            "model": model,
            "display_model": display_name(model),
            "matrix_position": range(len(full_labels)),
            "abbreviation": short_labels,
            "full_profile": full_labels,
        }
    )
    return short_labels, key


def _similarity_matrix(model: str, filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one cached inter-profile matrix with its plotted labels."""
    short_labels, key = _profile_labels(model)
    values = np.load(PROJECT_ROOT / "outputs" / model / filename)
    expected_shape = (len(short_labels), len(short_labels))
    if values.shape != expected_shape:
        raise ValueError(f"{model}/{filename}: expected {expected_shape}, found {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{model}/{filename}: matrix contains non-finite values")
    matrix = pd.DataFrame(
        values,
        index=pd.Index(short_labels, name="row_persona"),
        columns=pd.Index(short_labels, name="column_persona"),
    )
    return matrix, key


def _collect() -> tuple[list[Heatmap], pd.DataFrame]:
    """Collect all eight heatmap panels and their profile label keys."""
    heatmaps: list[Heatmap] = []
    label_keys: list[pd.DataFrame] = []
    matrix_specs = (
        (
            "caption_similarity",
            "ic_profile_sim_caption.npy",
            "Appendix Figure A1",
            "mean image-conditioned cosine similarity",
        ),
        (
            "justification_similarity",
            "ic_profile_sim_just.npy",
            "Appendix Figure A2",
            "mean image-conditioned cosine similarity",
        ),
        (
            "perception_jaccard",
            "ic_profile_sim_jaccard.npy",
            "Appendix Figure A3",
            "mean image-conditioned Jaccard similarity",
        ),
    )

    for model in PAPER_MODELS:
        topic_matrix = _topic_persona_matrix(model)
        heatmaps.append(
            Heatmap(
                model=model,
                name="topic_persona_proportion",
                figure="Main-paper topic-persona heatmap",
                metric="within-profile topic proportion, renormalized over displayed topics",
                matrix=topic_matrix,
            )
        )
        for name, filename, figure, metric in matrix_specs:
            matrix, key = _similarity_matrix(model, filename)
            heatmaps.append(
                Heatmap(
                    model=model,
                    name=name,
                    figure=figure,
                    metric=metric,
                    matrix=matrix,
                )
            )
            label_keys.append(key)

    labels = pd.concat(label_keys, ignore_index=True).drop_duplicates()
    return heatmaps, labels


def _long_form(heatmaps: list[Heatmap]) -> pd.DataFrame:
    """Convert all matrices to one tidy table without losing precision."""
    frames: list[pd.DataFrame] = []
    for heatmap in heatmaps:
        wide = heatmap.matrix.rename_axis(index="row_label", columns="column_label").reset_index()
        frame = wide.melt(
            id_vars="row_label",
            var_name="column_label",
            value_name="value",
        )
        frame.insert(0, "metric", heatmap.metric)
        frame.insert(0, "figure", heatmap.figure)
        frame.insert(0, "heatmap", heatmap.name)
        frame.insert(0, "display_model", display_name(heatmap.model))
        frame.insert(0, "model", heatmap.model)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _text_report(heatmaps: list[Heatmap]) -> str:
    """Render a readable report containing every matrix value."""
    sections = [
        "Exact values behind all heatmap panels in paper.tex",
        "=====================================================",
        "",
        "CSV files retain up to 17 significant digits. Values below are shown",
        "to 8 decimal places for readability.",
        "",
        "The topic-persona panels are normalized exactly as in src/generate_figures.py:",
        "each persona column sums to 1 over the ten displayed/labeled topics.",
    ]
    for heatmap in heatmaps:
        values = heatmap.matrix.to_numpy(dtype=float)
        sections.extend(
            [
                "",
                f"{display_name(heatmap.model)} — {heatmap.figure}",
                f"artifact: {heatmap.filename}",
                f"metric: {heatmap.metric}",
                f"shape: {heatmap.matrix.shape[0]} x {heatmap.matrix.shape[1]}",
                f"range: {values.min():.17g} to {values.max():.17g}",
                "",
                heatmap.matrix.to_string(float_format=lambda value: f"{value:.8f}"),
            ]
        )
    return "\n".join(sections) + "\n"


def export(output_dir: Path) -> list[Heatmap]:
    """Write all matrix, metadata, and human-readable output artifacts."""
    heatmaps, labels = _collect()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    for heatmap in heatmaps:
        heatmap.matrix.to_csv(output_dir / heatmap.filename, float_format=FLOAT_FORMAT)
        values = heatmap.matrix.to_numpy(dtype=float)
        manifest_rows.append(
            {
                "model": heatmap.model,
                "display_model": display_name(heatmap.model),
                "figure": heatmap.figure,
                "heatmap": heatmap.name,
                "metric": heatmap.metric,
                "rows": heatmap.matrix.shape[0],
                "columns": heatmap.matrix.shape[1],
                "minimum": values.min(),
                "maximum": values.max(),
                "csv": heatmap.filename,
            }
        )

    pd.DataFrame(manifest_rows).to_csv(
        output_dir / "manifest.csv",
        index=False,
        float_format=FLOAT_FORMAT,
    )
    labels.to_csv(output_dir / "profile_labels.csv", index=False)
    _long_form(heatmaps).to_csv(
        output_dir / "all_heatmap_values_long.csv",
        index=False,
        float_format=FLOAT_FORMAT,
    )
    (output_dir / "all_heatmap_values.txt").write_text(
        _text_report(heatmaps),
        encoding="utf-8",
    )
    return heatmaps


def main() -> int:
    """Run the command-line exporter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    heatmaps = export(args.output_dir)
    print(f"Exported {len(heatmaps)} heatmap panels to {args.output_dir}")
    for heatmap in heatmaps:
        values = heatmap.matrix.to_numpy(dtype=float)
        print(
            f"  {display_name(heatmap.model):9s} "
            f"{heatmap.name:28s} "
            f"{heatmap.matrix.shape[0]:2d}x{heatmap.matrix.shape[1]:2d} "
            f"[{values.min():.8f}, {values.max():.8f}]"
        )
    print(f"Full readable output: {args.output_dir / 'all_heatmap_values.txt'}")
    print(f"Combined tidy CSV:    {args.output_dir / 'all_heatmap_values_long.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

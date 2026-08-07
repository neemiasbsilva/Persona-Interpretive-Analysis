"""Cross-model summary figures for the paper.

Figures A and B need both paper models' outputs; Figure C is per model and is
produced by ``--annotated-only <model>`` from the per-model pipeline.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import seaborn as sns

from .config import FONT_SCALE, PROJECT_ROOT
from .figures.annotated_matrices import figure_annotated_matrices
from .figures.effect_summary import figure_persona_effect_summary, figure_topic_contrasts
from .models import PAPER_MODELS

__all__ = [
    "figure_annotated_matrices",
    "figure_persona_effect_summary",
    "figure_topic_contrasts",
    "main",
]


def main() -> None:
    """Run the summary-figure command-line interface."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model-a", default=PAPER_MODELS[0])
    ap.add_argument("--model-b", default=PAPER_MODELS[1])
    ap.add_argument(
        "--annotated-only",
        metavar="MODEL",
        default=None,
        help="Generate only the per-model annotated backup matrices (Figure C) "
        "for MODEL and skip the cross-model Figures A/B. Used by the "
        "per-model run_experiments.sh, since A/B need both models' outputs.",
    )
    args = ap.parse_args()

    sns.set_theme(style="whitegrid", font_scale=FONT_SCALE * 0.7)

    if args.annotated_only:
        print(f"[C] Annotated backup matrices ({args.annotated_only}) ...")
        figure_annotated_matrices(args.annotated_only)
        print("\nAnnotated matrices complete.")
        return

    root_figs = PROJECT_ROOT / "figures"
    root_figs.mkdir(exist_ok=True)

    print("[A] Persona-effect summary ...")
    figure_persona_effect_summary(args.model_a, args.model_b, root_figs)

    print("[B] Topic contrasts ...")
    figure_topic_contrasts(args.model_a, args.model_b, root_figs)

    print("[C] Annotated backup matrices ...")
    for model in (args.model_a, args.model_b):
        figure_annotated_matrices(model)

    print("\nSummary figures complete.")


if __name__ == "__main__":
    main()

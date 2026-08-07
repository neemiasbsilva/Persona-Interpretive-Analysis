"""Persona dimensions, modalities, and their orderings.

Three orderings coexist and must not be collapsed into one another:

``DEMO_COLS`` and ``MODALITIES`` fix the order in which cells are *computed*, and
therefore the row order of ``within_cross_significance.csv`` and the image order
fed to the bootstrap. Changing them changes published numbers.

``DIM_DISPLAY_ORDER`` fixes the order rows are *rendered* in the Appendix E table
and the summary figures (strongest to weakest effect, matching the Section 4.2
prose). It is presentation only.

``MODALITY_FIGURE_ORDER`` differs from ``MODALITIES`` because figures plot from
coarse to fine linguistic abstraction, while computation follows the corpus
column order.
"""

from __future__ import annotations

DEMO_COLS: list[str] = [
    "gender",
    "economic_status",
    "political_spectrum",
    "personality",
]

MODALITIES: tuple[str, ...] = ("caption", "justification", "perception")

DIM_DISPLAY_ORDER: list[str] = [
    "economic_status",
    "political_spectrum",
    "personality",
    "gender",
]

MODALITY_DISPLAY_ORDER: list[str] = ["caption", "justification", "perception"]

MODALITY_FIGURE_ORDER: list[str] = ["caption", "perception", "justification"]

DIM_LABELS: dict[str, str] = {
    "economic_status": "Economic status",
    "political_spectrum": "Political orientation",
    "personality": "Personality",
    "gender": "Gender",
}

MODALITY_LABELS: dict[str, str] = {
    "caption": "Caption",
    "justification": "Justification",
    "perception": "Perception",
}

MODALITY_FIGURE_LABELS: dict[str, str] = {
    "caption": "Caption (descriptive)",
    "perception": "Perception (tags)",
    "justification": "Justification (interpretive)",
}

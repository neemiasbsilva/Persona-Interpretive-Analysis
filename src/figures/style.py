"""Shared plotting conventions for the cross-model summary figures.

The dimension order runs strongest to weakest effect, matching the Section 4.2
prose. The modality order runs coarse to fine linguistic abstraction, which is
deliberately different from the order the statistics are computed in. Both come
from :mod:`src.schema`, so a figure cannot drift from the reported tables.

The modality colours are colour-blind safe and shared by every figure here.
"""

from __future__ import annotations

from ..schema import DIM_DISPLAY_ORDER as DIMS
from ..schema import DIM_LABELS
from ..schema import MODALITY_FIGURE_LABELS as MOD_LABELS
from ..schema import MODALITY_FIGURE_ORDER as MODALITIES

MOD_COLORS = {
    "caption": "#4477AA",
    "perception": "#CCBB44",
    "justification": "#EE6677",
}

TOPIC_CONTRASTS = [
    ("economic_status", "Low income", "High income"),
    ("political_spectrum", "Progressive", "Conservative"),
]

__all__ = [
    "DIMS",
    "DIM_LABELS",
    "MODALITIES",
    "MOD_COLORS",
    "MOD_LABELS",
    "TOPIC_CONTRASTS",
]

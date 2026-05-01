"""Central configuration: paths, colors, model IDs, hyperparameters."""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
DATA_PATH     = PROJECT_ROOT / "data" / "annotations_baseline.jsonl"
NP_THINK_PATH = PROJECT_ROOT / "data" / "annotations_no_persona_think.jsonl"
NP_NOTHINK_PATH = PROJECT_ROOT / "data" / "annotations_no_persona_no_think.jsonl"
FIGURES       = PROJECT_ROOT / "figures"
OUTPUTS       = PROJECT_ROOT / "outputs"

# ── Model IDs ─────────────────────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# ── Plotting ──────────────────────────────────────────────────────────────────
FONT_SCALE = 1.4

SENT_COLORS = {
    "Negative":         "#d73027",
    "SlightlyNegative": "#fc8d59",
    "Neutral":          "#aaaaaa",
    "SlightlyPositive": "#91cf60",
    "Positive":         "#1a9850",
}
SENT_COLORS_3 = {k: SENT_COLORS[k] for k in ["Negative", "Neutral", "Positive"]}

# ── BERTopic hyperparameters ──────────────────────────────────────────────────
BERTOPIC_MIN_CLUSTER_SIZE = 80
BERTOPIC_N_COMPONENTS     = 5
BERTOPIC_N_NEIGHBORS      = 15
BERTOPIC_TOP_N_WORDS      = 10
BERTOPIC_MIN_DF           = 10

# ── Embedding hyperparameters ─────────────────────────────────────────────────
EMBED_BATCH_SIZE = 256

# ── Similarity hyperparameters ────────────────────────────────────────────────
PROFILE_SIM_N_SAMPLE = 150

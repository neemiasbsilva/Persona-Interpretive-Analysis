"""Where each annotating model's corpus came from, and the fixed prose of the dataset card.

:data:`VARIATIONS` is keyed by the model directories of :mod:`src.models`, and the test
suite fails when a model has no entry, so no corpus reaches the Hub undocumented. The
section templates use ``<<NAME>>`` placeholders rather than ``str.format`` fields,
because the code samples they carry contain braces.
"""

from __future__ import annotations

from dataclasses import dataclass

VENUE = "EMNLP 2026 Workshop Pandora"
PAPER_TITLE = (
    "Persona Prompting in Multimodal Urban Perception: "
    "Descriptive Convergence and Interpretive Variation"
)
PROJECT_URL = "https://neemiasbsilva.github.io/Persona-Interpretive-Analysis-Portfolio/"
CODE_URL = "https://github.com/neemiasbsilva/Persona-Interpretive-Analysis"
PIPELINE_URL = "https://github.com/neemiasbsilva/MLLMs-persona-evaluation"
UPSTREAM_DATASET_ID = "Neemias/UrbanPersona-60K"
UPSTREAM_DATASET_URL = f"https://huggingface.co/datasets/{UPSTREAM_DATASET_ID}"
PERCEPTSENT_URL = "https://github.com/ceslop84/perceptsent"
REPUBLISHED = f"[UrbanPersona-60K]({UPSTREAM_DATASET_URL}), republished"
GENERATED = "generated for this work"


@dataclass(frozen=True, slots=True)
class Variation:
    """How one model directory's corpora were produced.

    Attributes:
        model_id: Model as it was served.
        backend: Serving backend and weight precision.
        decoding: Decoding settings.
        design: Personas, images and no-persona runs.
        source: Where the corpus was produced or first published.
    """

    model_id: str
    backend: str
    decoding: str
    design: str
    source: str


VARIATIONS: dict[str, Variation] = {
    "qwen-vl": Variation(
        model_id="Qwen3-VL-8B (`qwen3-vl:8b`)",
        backend="Ollama",
        decoding="T = 0.1",
        design="1,200 personas × 50 images; no-persona 5 runs × 50 images, `think` and `no_think`",  # noqa: RUF001
        source=REPUBLISHED,
    ),
    "gemma-4-E4B-it_t01": Variation(
        model_id="`google/gemma-4-E4B-it`",
        backend="Hugging Face transformers, bf16",
        decoding="T = 0.1",
        design=(
            "the same 1,200 personas × 50 images; "  # noqa: RUF001
            "no-persona 5 runs × 50 images, thinking on and off"  # noqa: RUF001
        ),
        source=GENERATED,
    ),
    "qwen-vl-t0": Variation(
        model_id="`Qwen/Qwen3-VL-8B-Instruct`",
        backend="Hugging Face transformers, bf16",
        decoding="greedy, T = 0.0",
        design="100 economic-status personas × 50 images; 1 no-persona run × 50 images",  # noqa: RUF001
        source=GENERATED,
    ),
    "gemma4-t0": Variation(
        model_id="`google/gemma-4-E4B-it`",
        backend="Hugging Face transformers, bf16",
        decoding="greedy, T = 0.0",
        design="100 economic-status personas × 50 images; 1 no-persona run × 50 images",  # noqa: RUF001
        source=GENERATED,
    ),
}

INTRO = """# <<NAME>>

Annotation corpora and analysis outputs for **"<<PAPER_TITLE>>"** (<<VENUE>>). Two
multimodal LLMs, Qwen3-VL-8B and Gemma4 E4B, annotate the same 50 PerceptSent urban scenes as the
same 1,200 demographic personas at T = 0.1, 60,000 persona × image attempts per model and 120,000
in all, next to their no-persona ablations, a greedy T = 0 decoding ablation for both models, and
the tables, embedding caches and figures the paper's analysis derives from them.

- **Project page:** <<PROJECT_URL>>
- **Analysis code:** <<CODE_URL>>
- **Annotation pipeline:** <<PIPELINE_URL>>
- **Upstream dataset, source of the `qwen_vl` splits:** <<UPSTREAM_DATASET_URL>>"""  # noqa: RUF001

VARIATIONS_LEAD = """## Variations

The annotating model is the variation. Every corpus follows the UrbanPersona-60K protocol and
record schema (the same personas, images, prompt and perception vocabulary) and differs in the
model, its serving backend and its decoding. Each model is one split name across the configs."""

VARIATION_NOTES = """**`qwen_vl` is republished, not regenerated.** Its persona corpus, its 292
parse failures and its two no-persona runs are byte-identical to the `annotations/` files of
UrbanPersona-60K, produced by [MLLMs-persona-evaluation](<<PIPELINE_URL>>) with `qwen3-vl:8b`
served locally through Ollama. Cite that release for the corpus itself.

**The other three splits were generated for this work**, in a private extension of that pipeline
that adds a Hugging Face transformers backend with bf16 weights. The Gemma4 run replays the
published protocol: the same 1,200 `persona_id`s over the same 50 `image_id`s, so `qwen_vl` and
`gemma_4_E4B_it_t01` pair one to one, and its two no-persona runs switch Gemma4's thinking mode on
and off through the chat template's `enable_thinking` flag. The T = 0 ablation isolates decoding
noise: 100 of the same personas, 50 high-income and 50 low-income with gender, political spectrum
and personality fixed to Female, Conservative and Analytical, annotated by both models with greedy
decoding, plus one no-persona run each."""

SCHEMA_ROWS: tuple[tuple[str, str, str], ...] = (
    ("`annotation_id`", "string", "`p{persona_prefix}_img_{image_id}_{condition}`, unique per row"),
    ("`persona_id`", "string", "persona UUID, or `np_r01`…`np_r05` for the no-persona runs"),
    ("`image_id`", "string", "PerceptSent image identifier; the file is `{image_id}.jpg`"),
    ("`condition`", "string", "`baseline`, `no_persona_think` or `no_persona_no_think`"),
    (
        "`raw_demographics`",
        "object",
        "`gender`, `economic_status`, `political_spectrum`, `personality`",
    ),
    (
        "`predicted_sentiment`",
        "string",
        "`Negative`, `SlightlyNegative`, `Neutral`, `SlightlyPositive`, `Positive`",
    ),
    (
        "`predicted_perceptions`",
        "list[string]",
        "perception labels, asked for as 1–5 drawn from PerceptSent's 593-label vocabulary",  # noqa: RUF001
    ),
    ("`caption`", "string", "objective, persona-agnostic description of the scene"),
    ("`justification`", "string", "one sentence in the persona's own voice"),
    ("`parse_retries`", "int", "JSON parse attempts needed, 0–3"),  # noqa: RUF001
    ("`timestamp_utc`", "string", "`YYYY-MM-DDTHH:MM:SSZ`; `datasets` loads it as `timestamp[s]`"),
    ("`error`", "string", "`failures` only: always `parse_retries_exhausted`"),
)

SCHEMA_NOTES = """Before grouping or concatenating:

- `raw_demographics` is an **empty object** in the no-persona configs, which is the ablation
  itself, and a struct of four strings everywhere else; cast it before concatenating configs.
- In `failures`, `predicted_sentiment`, `caption` and `justification` are empty strings,
  `predicted_perceptions` is `[]` and `parse_retries` is 3.
- `predicted_perceptions` is published as the models emitted it. The vocabulary is not closed in
  practice: every split carries labels outside the 593, mostly shortened or paraphrased entries,
  and a few lists outside `failures` are empty or longer than five labels. The vocabulary is
  `vocabulary/unique_perceptions.json` of UrbanPersona-60K; normalise before counting labels.
- `t0_baseline` keeps `condition: baseline`; its `persona_id`s are a subset of the 1,200, and the
  T = 0 no-persona runs are `np_r01` alone."""

OUTPUTS_SECTION = """## Analysis outputs and figures

`outputs/<model>/` holds the cached artifacts the analysis code wrote for each corpus, exactly as
the analysis repository tracks them:

- `within_cross_*` and `matched_factorial_*`: the paired-statistics and matched-factorial tables
  behind the paper's effects, with `*_method.json` (tests, resampling, library versions) and
  `*.meta.json` (SHA-256 of the inputs each cache was built from).
- `caption_embeddings.npy` and `justification_embeddings.npy` (paper models, about 92 MB each):
  float32, one 384-dimensional `sentence-transformers/all-MiniLM-L6-v2` row per persona
  annotation, in the row order of `caption_embeddings_ids.csv`. `np_*_embeddings.npy` are the
  no-persona caches of the notebooks, and `t0_*_embeddings.npy` with `t0_embeddings_ids.csv`
  embed the T = 0 corpora.
- Per-image and 24 × 24 inter-profile similarity matrices, BERTopic topic tables with curated
  labels, convergence tests and descriptive statistics.

`outputs/heatmap_values/` holds the exact numbers behind every published heatmap, indexed by
`manifest.csv`, and `outputs/cross_model_correlation.csv` the cross-model agreement of the
inter-profile matrices. `figures/<model>/` and `figures/` carry the per-model and cross-model
figures as PDF and PNG. Run logs and LaTeX tables are not published."""  # noqa: RUF001

IMAGES_SECTION = """## Source images

The PerceptSent images are not redistributed. Every record addresses its scene by `image_id`, the
file stem of `{image_id}.jpg` in the original release: <<PERCEPTSENT_URL>>"""

USAGE_SECTION = """## Usage

```python
from datasets import load_dataset

REPO = "<<REPO_ID>>"
qwen = load_dataset(REPO, "baseline", split="qwen_vl")
gemma = load_dataset(REPO, "baseline", split="gemma_4_E4B_it_t01")
failures = load_dataset(REPO, "failures")
greedy = load_dataset(REPO, "t0_baseline", split="gemma4_t0")
```

Everything outside the configs is a plain file download:

```python
from huggingface_hub import hf_hub_download

REPO = "<<REPO_ID>>"
table = hf_hub_download(REPO, "outputs/cross_model_correlation.csv", repo_type="dataset")
```

In a clone of the analysis repository, `./scripts/09_hub.sh pull` puts every published file back
at its own path under `data/`, `outputs/` and `figures/`."""

AUTHORS = (
    "Neemias Buceli da Silva and Matt Ratto and Myriam Delgado and Rodrigo Minetto and "
    "Daniel Silver and Thiago H. Silva"
)
BOOKTITLE = (
    "EMNLP26 Workshop on Pluralistic AI {\\&} NLP: Diversity-aware, Sociotechnical, "
    "Responsible Alignment (PANDORA)"
)

CITATION_SECTION = """## Citation

```bibtex
@inproceedings{silva2026persona,
  title={<<PAPER_TITLE>>},
  author={<<AUTHORS>>},
  booktitle={<<BOOKTITLE>>},
  year={2026},
}
```

The `qwen_vl` splits are the published UrbanPersona-60K corpus; if you use them, please also cite:

```bibtex
@inproceedings{urbcom26-neemias,
  title     = {Stable Behavior, Limited Variation: Persona Validity in LLM Agents for Urban
               Sentiment Perception},
  author    = {Neemias B da Silva and Rodrigo Minetto and Daniel Silver and Thiago H Silva},
  year      = {2026},
  booktitle = {Proc. of IEEE DCOSS-IoT-UrbCom},
  address   = {Reykjavik, Iceland}
}
```"""

ACKNOWLEDGMENTS_SECTION = """## Acknowledgments

Supported by the CIFAR project *Towards socially grounded AI safety: Integrating causal and
institutional reasoning in language models*, the National Council for Scientific and
Technological Development - CNPq (processes 314603/2023-9, 441444/2023-7 and 444724/2024-9), and
INCT TILD-IAR (proc. 408490/2024-1)."""

LICENSE_SECTION = """## License

CC BY 4.0, like UrbanPersona-60K. The PerceptSent images and annotations remain under the terms of
their original release."""

SPAN_SEPARATOR = " – "  # noqa: RUF001
ATTEMPTS_LEAD = "Persona × image attempts per T = 0.1 model, annotations plus parse failures"  # noqa: RUF001

SUBSTITUTIONS: dict[str, str] = {
    "<<PAPER_TITLE>>": PAPER_TITLE,
    "<<VENUE>>": VENUE,
    "<<AUTHORS>>": AUTHORS,
    "<<BOOKTITLE>>": BOOKTITLE,
    "<<PROJECT_URL>>": PROJECT_URL,
    "<<CODE_URL>>": CODE_URL,
    "<<PIPELINE_URL>>": PIPELINE_URL,
    "<<UPSTREAM_DATASET_URL>>": UPSTREAM_DATASET_URL,
    "<<PERCEPTSENT_URL>>": PERCEPTSENT_URL,
}

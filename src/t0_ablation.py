"""Decoding-sensitivity ablation at temperature T=0 (economic-status only).

The T=0 corpora (data/{qwen-vl-t0,gemma4-t0}/) hold gender, political orientation,
and personality fixed and vary *only* economic status (50 Low + 50 High personas),
over all 50 images, decoded deterministically (T=0). Because decoding is
deterministic and the persona UUID does not enter the prompt, every persona within
an income level produces byte-identical output, so within-level similarity is
trivially 1.0. The informative quantity is therefore the *cross-level* (High vs Low)
similarity per image, which isolates the effect of flipping economic status alone.

Two modes:

  Per-model (default, model chosen by PERSONA_MODEL):
      PERSONA_MODEL=qwen-vl-t0 uv run python -m src.t0_ablation
    -> caches embeddings, writes outputs/<model>/t0_within_cross.csv and
       outputs/<model>/t0_econ_summary.csv.

  Summary (both models, model-agnostic figure + LaTeX table):
      uv run python -m src.t0_ablation --summary \
          --model-a qwen-vl-t0 --model-b gemma4-t0
    -> writes figures/fig_t0_ablation.{pdf,png} and outputs/t0_ablation_table.tex.
"""

import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

from .config import FONT_SCALE, OUTPUTS, PROJECT_ROOT
from .data_loading import create_profiles, load_annotations, parse_demographics
from .embeddings import load_or_encode_with_ids
from .models import display_name, outputs_dir
from .similarity import compute_within_cross_similarity

DIMENSION = "economic_status"
MODALITIES = [
    ("caption", "Caption\n(cosine)"),
    ("justification", "Justification\n(cosine)"),
    ("perception", "Perception\n(Jaccard)"),
]


def _text_identity_rate(df: pd.DataFrame, col: str) -> float:
    """Fraction of images whose High-income and Low-income outputs are identical.

    Identity is compared as a *set* of texts. At T=0 each level is a single
    deterministic output, so this reduces to 'is the High text == the Low text'
    for that image.

    Args:
        df: T=0 annotations for one model.
        col: Text column to compare.

    Returns:
        The identity rate, or NaN when no image has both levels.
    """
    hits, n = 0, 0
    for _, grp in df.groupby("image_id"):
        levels = grp.groupby(DIMENSION)[col].apply(frozenset)
        if len(levels) < 2:
            continue
        n += 1
        vals = list(levels)
        if all(v == vals[0] for v in vals):
            hits += 1
    return hits / n if n else float("nan")


def run_model() -> pd.DataFrame:
    """Compute the T=0 cross-level economic-status summary for the active model."""
    out = OUTPUTS
    out.mkdir(parents=True, exist_ok=True)

    print("Loading annotations...")
    df = load_annotations()
    df = parse_demographics(df)
    df = create_profiles(df)
    n_levels = df[DIMENSION].nunique()
    print(
        f"  {len(df):,} records | images={df['image_id'].nunique()} | "
        f"{DIMENSION} levels={n_levels} ({sorted(df[DIMENSION].unique())})"
    )
    if n_levels < 2:
        raise SystemExit(f"Need >=2 {DIMENSION} levels; found {n_levels}.")

    print("Embedding captions and justifications...")
    id_cache = out / "t0_embeddings_ids.csv"
    cap_embs, id_index = load_or_encode_with_ids(
        df, "caption", out / "t0_caption_embeddings.npy", id_cache
    )
    just_embs, _ = load_or_encode_with_ids(
        df, "justification", out / "t0_justification_embeddings.npy", id_cache
    )

    print("Computing within/cross similarity (economic_status)...")
    wc = compute_within_cross_similarity(
        df,
        cap_embs,
        just_embs,
        id_index,
        cache_path=out / "t0_within_cross.csv",
    )
    wc = wc[wc["dimension"] == DIMENSION].copy()

    id_rates = {
        "caption": _text_identity_rate(df, "caption"),
        "justification": _text_identity_rate(df, "justification"),
        "perception": _text_identity_rate(
            df.assign(
                _perc=df["predicted_perceptions"].apply(
                    lambda v: tuple(sorted(v)) if isinstance(v, list) else v
                )
            ),
            "_perc",
        ),
    }

    rows = []
    for mod, _ in MODALITIES:
        sub = wc[wc["modality"] == mod].dropna(subset=["cross_mean"])
        rows.append(
            {
                "modality": mod,
                "within_mean": float(sub["within_mean"].mean()),
                "cross_mean": float(sub["cross_mean"].mean()),
                "cross_std": float(sub["cross_mean"].std()),
                "n_images": len(sub),
                "identical_rate": float(id_rates[mod]),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "t0_econ_summary.csv", index=False)

    cap_x = wc[wc.modality == "caption"].set_index("image_id")["cross_mean"]
    just_x = wc[wc.modality == "justification"].set_index("image_id")["cross_mean"]
    common = cap_x.index.intersection(just_x.index)
    stat, p = wilcoxon(cap_x.loc[common], just_x.loc[common])

    print(f"\n=== {display_name(OUTPUTS.name)} (T=0, {DIMENSION}) ===")
    for _, r in summary.iterrows():
        print(
            f"  {r.modality:<13s} within={r.within_mean:.3f}  "
            f"cross(High-vs-Low)={r.cross_mean:.3f}±{r.cross_std:.3f}  "
            f"identical={r.identical_rate:.0%} of {r.n_images}"
        )
    print(
        f"  Wilcoxon caption-vs-justification cross-level: "
        f"W={stat:.1f}, p={p:.2e} (caption more similar => justifications diverge more)"
    )
    print(f"  Saved {out / 't0_econ_summary.csv'} and t0_within_cross.csv")
    return summary


def _per_image_cross(model: str) -> pd.DataFrame:
    wc = pd.read_csv(outputs_dir(model) / "t0_within_cross.csv")
    wc = wc[wc["dimension"] == DIMENSION].dropna(subset=["cross_mean"]).copy()
    wc["model"] = display_name(model)
    return wc


def make_summary(model_a: str, model_b: str) -> None:
    """Draw the T=0 ablation figure and write its LaTeX table.

    Args:
        model_a: First T=0 model directory id.
        model_b: Second T=0 model directory id.
    """
    sns.set_theme(style="whitegrid", font_scale=FONT_SCALE)
    fig_dir = PROJECT_ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    long = pd.concat([_per_image_cross(model_a), _per_image_cross(model_b)], ignore_index=True)
    mod_order = [m for m, _ in MODALITIES]
    mod_labels = dict(MODALITIES)
    long["mod_label"] = long["modality"].map(mod_labels)
    long = long[long["modality"].isin(mod_order)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    order = [mod_labels[m] for m in mod_order]
    sns.boxplot(
        data=long,
        x="mod_label",
        y="cross_mean",
        hue="model",
        order=order,
        palette="muted",
        width=0.6,
        fliersize=2,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Cross-level similarity\n(High income vs. Low income)")
    ax.set_ylim(0, 1.02)
    ax.legend(title="", loc="lower left", framealpha=0.9)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig_t0_ablation.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_dir / 'fig_t0_ablation.pdf'}")

    sa = pd.read_csv(outputs_dir(model_a) / "t0_econ_summary.csv").set_index("modality")
    sb = pd.read_csv(outputs_dir(model_b) / "t0_econ_summary.csv").set_index("modality")
    disp = {
        "caption": "Caption (cosine)",
        "justification": "Justification (cosine)",
        "perception": "Perception (Jaccard)",
    }
    lines = [
        r"\begin{tabular}{l|c|c}",
        r"    \toprule",
        f"    Modality & {display_name(model_a)} & {display_name(model_b)} \\\\",
        r"    \midrule",
    ]
    lines.extend(
        f"    {disp[m]} & ${sa.loc[m, 'cross_mean']:.3f}$ & ${sb.loc[m, 'cross_mean']:.3f}$ \\\\"
        for m in mod_order
    )
    lines += [r"    \bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines) + "\n"
    (PROJECT_ROOT / "outputs" / "t0_ablation_table.tex").write_text(tex)
    print("  Saved outputs/t0_ablation_table.tex")
    print("\n--- table ---\n" + tex)
    for name, s in [(model_a, sa), (model_b, sb)]:
        print(
            f"{display_name(name)} identical-across-income rates: "
            + ", ".join(f"{m}={s.loc[m, 'identical_rate']:.0%}" for m in [k for k, _ in MODALITIES])
        )


def main() -> None:
    """Run the command-line interface."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summary", action="store_true", help="build the cross-model figure + LaTeX table"
    )
    ap.add_argument("--model-a", default="qwen-vl-t0")
    ap.add_argument("--model-b", default="gemma4-t0")
    args = ap.parse_args()
    if args.summary:
        make_summary(args.model_a, args.model_b)
    else:
        run_model()


if __name__ == "__main__":
    main()

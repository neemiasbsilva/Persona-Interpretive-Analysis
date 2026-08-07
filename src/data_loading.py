"""Annotation loading and demographic parsing."""

import ast
import json
from pathlib import Path

import pandas as pd

from .config import DATA_PATH


def load_annotations(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load JSONL annotations into a DataFrame."""
    with Path(path).open() as handle:
        records = [json.loads(line) for line in handle]
    df = pd.DataFrame(records)
    df["caption_len"] = df["caption"].str.split().str.len()
    df["justification_len"] = df["justification"].str.split().str.len()
    df["n_perceptions"] = df["predicted_perceptions"].apply(len)
    return df


def _parse_demo(val: object) -> dict:
    if isinstance(val, dict):
        return val
    if not isinstance(val, str):
        return {}
    try:
        parsed = ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Add gender, economic_status, political_spectrum, personality columns."""
    df = df.copy()
    df["_demo"] = df["raw_demographics"].apply(_parse_demo)
    for col in ["gender", "economic_status", "political_spectrum", "personality"]:
        df[col] = df["_demo"].apply(lambda d, key=col: d.get(key, ""))
    return df.drop(columns=["_demo"])


def abbreviate_profile(profile: str) -> str:
    """Shorten 'Female / Low income / Conservative / Pragmatic' → 'F/Low /Cons/Prag'."""
    parts = [x.strip() for x in profile.split("/")]
    return f"{parts[0][0]}/{parts[1][:4]}/{parts[2][:4]}/{parts[3][:4]}"


def create_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'profile' (full string) and 'profile_abbr' columns."""
    df = df.copy()
    df["profile"] = (
        df["gender"]
        + " / "
        + df["economic_status"]
        + " / "
        + df["political_spectrum"]
        + " / "
        + df["personality"]
    )
    df["profile_abbr"] = df["profile"].apply(abbreviate_profile)
    return df

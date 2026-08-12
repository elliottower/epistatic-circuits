"""Greater-than task prompt loading.

Loads the canonical greater-than dataset (Hanna et al. 2023):
  "The X lasted from the year XXYY to the year XX"
Model must predict two-digit year tokens > YY.

Metric: prob_diff = P(years > YY) - P(years <= YY) over 100 year tokens.
"""

import csv
from pathlib import Path


DATA_PATHS = [
    Path(__file__).parent.parent / "data" / "greater_than_data.csv",
]


def get_year_token_ids(tokenizer):
    """Get token IDs for '00' through '99' (two-digit year completions)."""
    ids = []
    for year in range(100):
        toks = tokenizer(f"{year:02d}").input_ids
        assert len(toks) == 1, f"Year {year:02d} tokenizes to {len(toks)} tokens"
        ids.append(toks[0])
    return ids


def load_prompts(csv_path=None):
    """Load greater-than prompts from CSV.

    Returns list of dicts with keys: clean, corrupted, year_yy.
    """
    if csv_path is None:
        for p in DATA_PATHS:
            if p.exists():
                csv_path = p
                break
        if csv_path is None:
            raise FileNotFoundError(
                f"No greater-than CSV found. Looked in: {DATA_PATHS}"
            )

    prompts = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            year_col = "correct_idx" if "correct_idx" in row else "label"
            prompts.append({
                "clean": row["clean"],
                "corrupted": row["corrupted"],
                "year_yy": int(row[year_col]),
            })
    return prompts

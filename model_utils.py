import re
import numpy as np
import pandas as pd

NUM_FEATURES = [
    "floor_area_sqm",
    "Hawker_Within_2km",
    "mrt_nearest_distance",
    "Tranc_Year",
    "year_completed",
    "mid_storey",
]

CAT_FEATURES = ["flat_type"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

SYNONYMS = {
    "trac_year": "Tranc_Year",
    "tracyear": "Tranc_Year",
    "tranc_year": "Tranc_Year",
    "trancyear": "Tranc_Year",
    "hawker_within_2km": "Hawker_Within_2km",
    "hawkerwithin2km": "Hawker_Within_2km",
    "hawker__within_2km": "Hawker_Within_2km",
    "mrt_nearest_dist": "mrt_nearest_distance",
    "mrt_distance": "mrt_nearest_distance",
}

def _normalize_col_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s

def canonicalize_and_select_np(X):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    df = X.copy()
    normalized_to_actual = {_normalize_col_name(c): c for c in df.columns}

    rename_map = {}
    for norm, actual in normalized_to_actual.items():
        if norm in SYNONYMS:
            rename_map[actual] = SYNONYMS[norm]
            continue
        for canonical in ALL_FEATURES:
            if norm == _normalize_col_name(canonical):
                rename_map[actual] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = np.nan

    return df[ALL_FEATURES].to_numpy()

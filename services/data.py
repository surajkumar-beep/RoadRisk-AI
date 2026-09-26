"""
Shared dataset loading for the RoadRisk dashboard services.

Every analytics-style page needs *the data the model actually saw* (same
cleaning & feature-engineering rules) so that filters and breakdowns align
with the encoded categories.  This module applies the exact preprocessing
consistent-value logic from ``preprocessing.utils.clean_string_value`` and
the grouping rules from ``preprocessing.feature_engineering`` to a cached
DataFrame.  The original rows are never modified on disk.
"""
import os

import pandas as pd

from preprocessing.feature_engineering import group_driving_experience, group_vehicle_age
from preprocessing.utils import clean_string_value, DATASET_PATH, NOMINAL_FEATURES

# Display columns that keep their raw string form (Time clocks, target).
_RAW_STRING_COLUMNS = {"Time", "Accident_severity"}

_HOUR_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Cache handles: the whole DataFrame (~12k x 32) fits comfortably in memory
# and is reused by every page, so it is loaded at most once per process.
_DF = None
_PROCESSED = None


def _object_columns(df, raw_string_columns):
    """List of string-dtype columns, skipping the raw-string display columns.

    Uses a version-agnostic dtype probe (select_dtypes' 'object' alias is
    deprecated in pandas >= 2.2 in favour of 'str', which emits a warning on
    newer releases).
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        candidates = df.select_dtypes(include=["object"]).columns.tolist()
    return [c for c in candidates if c not in raw_string_columns]


def load_raw_dataset(use_cache=True):
    """Load the raw RTA dataset and apply the pipeline-consistent cleaning.

    Returns a copy with:
      - consistent casing / known corruption fixes (same as the ML pipeline)
      - grouped Driving_experience / Service_year_of_vehicle (model categories)
      - an extra numeric ``Hour`` column derived from ``Time``
    """
    global _DF
    if use_cache and _DF is not None:
        return _DF.copy()

    df = pd.read_csv(DATASET_PATH)

    # 1) Consistent string cleaning (same function used by preprocessing).
    for col in _object_columns(df, _RAW_STRING_COLUMNS):
        df[col] = df[col].apply(lambda v: clean_string_value(v, column=col))

    # 2) Group driving experience / vehicle age exactly like feature_engineering.
    df = group_driving_experience(df, verbose=False)
    df = group_vehicle_age(df, verbose=False)

    # 3) Numeric hour (clock time is available; calendar dates are NOT).
    hour_series = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour
    df["Hour"] = hour_series.fillna(-1).astype(int)

    _DF = df.copy()
    return _DF.copy()


def load_processed_dataset(use_cache=True):
    """Load the scaled 165-feature matrix used to train/score the model."""
    global _PROCESSED
    if use_cache and _PROCESSED is not None:
        return _PROCESSED.copy()
    df = pd.read_csv(os.path.join(os.path.dirname(DATASET_PATH), "processed_dataset.csv"))
    df = df.drop(columns=["Accident_severity"], errors="ignore")
    _PROCESSED = df
    return df.copy()


def severity_counts(df):
    """Counts per severity, always in the canonical display order."""
    counts = df["Accident_severity"].value_counts().to_dict()
    return {severity: int(counts.get(severity, 0)) for severity in
            ["Slight Injury", "Serious Injury", "Fatal injury"]}


def hour_counts(df):
    """dict hour -> accident count (0..23)."""
    out = {h: 0 for h in range(24)}
    for hour, count in df.groupby("Hour").size().items():
        if 0 <= hour <= 23:
            out[int(hour)] = int(count)
    return out


def day_counts(df):
    """dict day -> accident count, ordered Monday..Sunday."""
    counts = df["Day_of_week"].value_counts()
    return {day: int(counts.get(day, 0)) for day in _HOUR_ORDER}


def top_categories(df, column, limit=10):
    """Sorted (labels desc, values desc) for a categorical column."""
    counts = df[column].value_counts()
    pairs = sorted(
        ((str(k) if not pd.isna(k) else "Unknown", int(v)) for k, v in counts.items()),
        key=lambda p: p[1],
        reverse=True,
    )
    return pairs[:limit]
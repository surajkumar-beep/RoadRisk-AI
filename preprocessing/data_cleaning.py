"""
Data Cleaning Module for the RTA preprocessing pipeline.

Handles:
  - Duplicate record removal (4.2)
  - Missing value imputation (4.3)
  - Inconsistent value cleaning - whitespace, case, spelling (4.4)
  - Data type conversion (4.5)
"""
import pandas as pd
import numpy as np

from preprocessing.utils import (
    clean_string_value,
    DROP_FEATURES,
    ORDINAL_MAPPINGS,
    NOMINAL_FEATURES,
)


def remove_duplicates(df, verbose=True):
    """
    Remove duplicate records from the DataFrame.

    Task 4.2: Remove duplicate records.
    """
    initial_rows = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_rows - len(df_clean)
    if verbose:
        print(f"Duplicates removed: {duplicates_removed}")
        print(f"Rows after deduplication: {len(df_clean)}")
    return df_clean, {"duplicates_removed": duplicates_removed}


def clean_inconsistent_values(df, verbose=True):
    """
    Clean inconsistent string values across all object columns.

    Task 4.4: Clean inconsistent values:
      - Trim whitespace
      - Standardize text/case
      - Fix spelling inconsistencies
      - Replace invalid/unknown values
    """
    df_clean = df.copy()
    cleaned_counts = {}

    # Apply string cleaning to all object/string columns (exclude target & Time)
    for col in df_clean.columns:
        is_string_col = (
            df_clean[col].dtype == "object"
            or str(df_clean[col].dtype).startswith("string")
            or str(df_clean[col].dtype) == "str"
        )
        if is_string_col and col not in ["Accident_severity", "Time"]:
            before = df_clean[col].astype(str).str.strip().str.lower().value_counts().to_dict()
            df_clean[col] = df_clean[col].apply(lambda v: clean_string_value(v, column=col))
            after = df_clean[col].astype(str).str.strip().str.lower().value_counts().to_dict()
            # Detect changes (unique value count should reduce if corruptions fixed)
            changed = sum(after.values()) - sum(before.values())
            if len(before) != len(after):
                cleaned_counts[col] = {
                    "before_unique": len(before),
                    "after_unique": len(after),
                }

    # Fix case inconsistencies: map 'other' -> 'Other' done in clean_string_value
    # Normalize the 'Area_accident_occured' - has padding issues
    if "Area_accident_occured" in df_clean.columns:
        df_clean["Area_accident_occured"] = (
            df_clean["Area_accident_occured"].str.strip().str.replace(r"\s+", " ", regex=True)
        )

    if verbose:
        print("Inconsistent value cleaning completed.")
        if cleaned_counts:
            print(f"Columns with value standardization: {list(cleaned_counts.keys())}")

    return df_clean


def impute_missing_values(df, verbose=True):
    """
    Handle missing values.

    Task 4.3: Handle missing values:
      - Categorical -> Mode
      - Apply dataset-specific strategies where appropriate.
    """
    df_imputed = df.copy()
    imputation_log = {}

    # Identify columns with missing values
    missing_cols = df_imputed.columns[df_imputed.isnull().any()].tolist()

    for col in missing_cols:
        if col == "Accident_severity":
            continue

        missing_count = df_imputed[col].isnull().sum()

        # Check if column is categorical (object or string dtype)
        is_categorical = (
            df_imputed[col].dtype == "object"
            or str(df_imputed[col].dtype).startswith("string")
            or str(df_imputed[col].dtype) == "str"
        )

        if is_categorical:
            # Categorical -> mode imputation
            mode_value = df_imputed[col].mode()[0]
            df_imputed[col] = df_imputed[col].fillna(mode_value)
            imputation_log[col] = {
                "type": "mode",
                "value": mode_value,
                "missing_count": missing_count,
            }
        else:
            # Numerical -> median imputation (robust to outliers)
            median_value = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_value)
            imputation_log[col] = {
                "type": "median",
                "value": float(median_value),
                "missing_count": missing_count,
            }

    if verbose:
        print("Missing value imputation completed:")
        for col, info in imputation_log.items():
            print(
                f"  - {col}: {info['missing_count']} missing -> "
                f"{info['type']} ({info['value']})"
            )

    return df_imputed, imputation_log


def drop_low_value_features(df, verbose=True):
    """
    Drop features with high missing rates or low predictive value.

    Uses DROP_FEATURES constant from utils.
    """
    df_dropped = df.drop(columns=[c for c in DROP_FEATURES if c in df.columns])
    if verbose:
        print(f"Dropped features: {[c for c in DROP_FEATURES if c in df.columns]}")
    return df_dropped


def convert_data_types(df, verbose=True):
    """
    Convert columns to appropriate data types.

    Task 4.5: Convert columns to appropriate data types:
      - Time -> datetime/time (handled in feature engineering -> Hour)
      - Numerical -> int/float
      - Categorical -> category/object
    """
    df_converted = df.copy()

    # Convert integer-like columns
    for col in ["Number_of_vehicles_involved", "Number_of_casualties"]:
        if col in df_converted.columns:
            df_converted[col] = pd.to_numeric(df_converted[col], errors="coerce").astype("Int64")

    # 'Defect_of_vehicle' - contains defect codes '5', '7' as strings; convert after
    # missing handling (it may be dropped) - handled in pipeline.

    if verbose:
        print("Data type conversion completed.")

    return df_converted


def run_data_cleaning(df, verbose=True):
    """
    Run the full data cleaning pipeline:
      1. Remove duplicates
      2. Clean inconsistent values
      3. Drop low-value features
      4. Impute missing values
      5. Convert data types

    Returns cleaned DataFrame and a log of cleaning operations.
    """
    log = {}

    # 1. Remove duplicates
    df_result, dup_log = remove_duplicates(df, verbose=verbose)
    log["duplicates"] = dup_log

    # 2. Clean inconsistent values
    df_result = clean_inconsistent_values(df_result, verbose=verbose)

    # 3. Drop low-value features (high missing %, low value)
    df_result = drop_low_value_features(df_result, verbose=verbose)
    log["dropped_features"] = DROP_FEATURES

    # 4. Impute missing values
    df_result, imputation_log = impute_missing_values(df_result, verbose=verbose)
    log["imputation"] = imputation_log

    # 5. Convert data types
    df_result = convert_data_types(df_result, verbose=verbose)

    return df_result, log
"""
Encoding Module for the RTA preprocessing pipeline.

Handles:
  - Label Encoding for ordinal features (4.7)
  - One-Hot Encoding for nominal features (4.7)
  - Target encoding (4.8)
  - Saving encoding artifacts (4.12)
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from preprocessing.utils import (
    ORDINAL_MAPPINGS,
    NOMINAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_MAPPING,
    LABEL_ENCODERS_PATH,
)


def encode_ordinal_features(df, verbose=True):
    """
    Label-encode ordinal categorical features using predefined mappings.

    Task 4.7: Label Encoding for ordinal features.

    Ordinal features: Age_band_of_driver, Driving_experience,
    Service_year_of_vehicle, Educational_level, Light_conditions,
    Road_surface_conditions, Age_band_of_casualty, Casualty_severity
    """
    df_encoded = df.copy()
    ordinal_cols = [col for col in ORDINAL_MAPPINGS.keys() if col in df_encoded.columns]

    for col in ordinal_cols:
        mapping = ORDINAL_MAPPINGS[col]
        # Apply mapping; any value not in mapping becomes np.nan (shouldn't happen after cleaning)
        df_encoded[col] = df_encoded[col].map(mapping)
        # For Age_band_of_casualty, 'Below 1yr' doesn't exist but '5' maps to 0 (Under 18)

    if verbose:
        print(f"Label-encoded ordinal features: {ordinal_cols}")

    return df_encoded


def encode_nominal_features(df, verbose=True):
    """
    One-hot encode nominal categorical features.

    Task 4.7: One-Hot Encoding for nominal features.

    Returns encoded DataFrame and the fitted OneHotEnoder for reuse.
    """
    df_encoded = df.copy()

    # Determine which nominal features exist in the DataFrame
    nominal_present = [col for col in NOMINAL_FEATURES if col in df_encoded.columns]

    if not nominal_present:
        if verbose:
            print("No nominal features to encode.")
        return df_encoded, None

    # One-hot encode with handle_unknown='ignore' for new categories
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.int64)

    encoded_array = encoder.fit_transform(df_encoded[nominal_present])

    # Build new column names: {original_col}_{category}
    feature_names = encoder.get_feature_names_out(nominal_present)
    # Clean column names (remove brackets etc.)
    feature_names = [name.replace(" ", "_").replace("(", "").replace(")", "") for name in feature_names]

    # Remove brackets from names - replace any remaining special chars
    feature_names = [name.replace("[", "").replace("]", "") for name in feature_names]

    # Create DataFrame of encoded columns
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=feature_names,
        index=df_encoded.index,
    )

    # Drop original nominal columns and concat encoded
    df_encoded = df_encoded.drop(columns=nominal_present)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

    if verbose:
        print(f"One-hot encoded nominal features: {nominal_present}")
        print(f"  Original columns: {len(nominal_present)}, New columns: {len(feature_names)}")

    return df_encoded, encoder


def encode_target(df, verbose=True):
    """
    Encode the target column Accident_severity into numerical labels.

    Task 4.8: Encode target (Slight Injury=0, Serious Injury=1, Fatal injury=2)
    """
    df_encoded = df.copy()

    if "Accident_severity" in df_encoded.columns:
        df_encoded["Accident_severity"] = df_encoded["Accident_severity"].map(TARGET_MAPPING)

        if verbose:
            print("Target encoded: Slight Injury -> 0, Serious Injury -> 1, Fatal injury -> 2")
            print(df_encoded["Accident_severity"].value_counts().sort_index())

    return df_encoded


def save_label_encoders(encoders_dict, path=LABEL_ENCODERS_PATH, verbose=True):
    """
    Save label/mapping encoders to a pickle file.

    Task 4.12: Save preprocessing artifacts.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(encoders_dict, f)
    if verbose:
        print(f"Saved label encoders to: {path}")


def save_encoder(encoder, path, verbose=True):
    """Save a fitted sklearn encoder to pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(encoder, f)
    if verbose:
        print(f"Saved encoder to: {path}")


def run_encoding(df, verbose=True):
    """
    Run the full encoding pipeline:
      1. Label-encode ordinal features
      2. One-hot encode nominal features
      3. Encode target column

    Returns encoded DataFrame and encoders dict.
    """
    log = {}

    # 1. Encode ordinal features
    df_encoded = encode_ordinal_features(df, verbose=verbose)
    log["ordinal_encoded"] = [col for col in ORDINAL_MAPPINGS.keys() if col in df.columns]

    # 2. Encode target
    df_encoded = encode_target(df_encoded, verbose=verbose)
    log["target_encoded"] = True

    # 3. One-hot encode nominal features
    nominal_present = [col for col in NOMINAL_FEATURES if col in df_encoded.columns]
    # Extract original categories before one-hot for encoders dict
    ordinal_mappings_used = {col: ORDINAL_MAPPINGS[col] for col in ORDINAL_MAPPINGS if col in df_encoded.columns}

    df_encoded, ohe = encode_nominal_features(df_encoded, verbose=verbose)
    log["nominal_encoded"] = nominal_present

    # Build encoders dict for saving
    encoders_dict = {
        "ordinal_mappings": ordinal_mappings_used,
        "target_mapping": TARGET_MAPPING,
        "one_hot_encoder": ohe,
        "nominal_features": nominal_present,
        "ordinal_features": list(ordinal_mappings_used.keys()),
    }

    return df_encoded, encoders_dict, log
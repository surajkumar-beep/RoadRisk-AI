"""
Scaling Module for the RTA preprocessing pipeline.

Handles:
  - Scaling numerical features (StandardScaler / MinMaxScaler)
  - Saving scaler artifacts
"""
import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from preprocessing.utils import NUMERICAL_FEATURES


def scale_numerical_features(df, scaler_type="standard", verbose=True):
    """
    Scale numerical features.

    Task 4.7/4.12: Scale numerical features and save scaler.

    Parameters:
      - scaler_type: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
    """
    df_scaled = df.copy()

    # Determine which numerical features exist
    numerical_present = [col for col in NUMERICAL_FEATURES if col in df_scaled.columns]

    if not numerical_present:
        if verbose:
            print("No numerical features to scale.")
        return df_scaled, None

    # Choose scaler
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    # Fit and transform
    scaled_array = scaler.fit_transform(df_scaled[numerical_present])

    # Replace original columns with scaled values
    df_scaled[numerical_present] = scaled_array

    if verbose:
        print(f"Scaled numerical features: {numerical_present} using {type(scaler).__name__}")

    return df_scaled, scaler


def save_scaler(scaler, path, verbose=True):
    """Save a fitted scaler to pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    if verbose:
        print(f"Saved scaler to: {path}")


def run_scaling(df, scaler_type="standard", verbose=True):
    """
    Run the scaling pipeline.

    Returns scaled DataFrame and fitted scaler.
    """
    df_scaled, scaler = scale_numerical_features(df, scaler_type=scaler_type, verbose=verbose)
    return df_scaled, scaler
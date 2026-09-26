"""
Main Preprocessing Pipeline for the RTA dataset.

Orchestrates the full preprocessing flow:
  1. Load raw dataset
  2. Data cleaning (duplicates, missing values, inconsistencies, types)
  3. Feature engineering (Hour extraction, grouping)
  4. Encoding (ordinal label, nominal one-hot, target)
  5. Scaling (numerical features)
  6. Save processed dataset and artifacts
"""
import os
import pickle
import pandas as pd

from preprocessing.utils import (
    DATASET_PATH,
    PROCESSED_DATASET_PATH,
    MODELS_DIR,
    LABEL_ENCODERS_PATH,
    PIPELINE_PATH,
    load_raw_dataset,
    save_dataframe,
)
from preprocessing.data_cleaning import run_data_cleaning
from preprocessing.feature_engineering import run_feature_engineering
from preprocessing.encoding import run_encoding, save_label_encoders
from preprocessing.scaling import run_scaling, save_scaler


def run_preprocessing_pipeline(
    input_path=DATASET_PATH,
    output_path=PROCESSED_DATASET_PATH,
    scaler_type="standard",
    verbose=True,
    save_artifacts=True,
):
    """
    Run the complete preprocessing pipeline.

    Returns:
      - processed_df: Final processed DataFrame
      - pipeline_log: Dictionary of all preprocessing decisions
    """
    pipeline_log = {}

    # ------------------------------------------------------------------
    # 1. Load raw dataset (Task 4.1)
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 80)
        print("STEP 1: LOAD RAW DATASET")
        print("=" * 80)
    df = load_raw_dataset(input_path)
    pipeline_log["raw_shape"] = df.shape

    # ------------------------------------------------------------------
    # 2. Data cleaning (Tasks 4.2, 4.3, 4.4, 4.5)
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2: DATA CLEANING")
        print("=" * 80)
    df_clean, cleaning_log = run_data_cleaning(df, verbose=verbose)
    pipeline_log["cleaning"] = cleaning_log
    pipeline_log["cleaned_shape"] = df_clean.shape

    # ------------------------------------------------------------------
    # 3. Feature engineering (Task 4.6)
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 3: FEATURE ENGINEERING")
        print("=" * 80)
    df_fe, fe_log = run_feature_engineering(df_clean, verbose=verbose)
    pipeline_log["feature_engineering"] = fe_log
    pipeline_log["feature_engineered_shape"] = df_fe.shape

    # ------------------------------------------------------------------
    # 4. Encoding (Tasks 4.7, 4.8)
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 4: ENCODING")
        print("=" * 80)
    df_encoded, encoders_dict, encoding_log = run_encoding(df_fe, verbose=verbose)
    pipeline_log["encoding"] = encoding_log
    pipeline_log["encoded_shape"] = df_encoded.shape

    # ------------------------------------------------------------------
    # 5. Scaling (Task 4.7 - numerical features)
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 5: SCALING")
        print("=" * 80)
    df_scaled, scaler = run_scaling(df_encoded, scaler_type=scaler_type, verbose=verbose)
    pipeline_log["scaling"] = {"scaler_type": scaler_type}
    pipeline_log["final_shape"] = df_scaled.shape

    # ------------------------------------------------------------------
    # 6. Save processed dataset (Task 4.11)
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 6: SAVE PROCESSED DATASET")
        print("=" * 80)
    save_dataframe(df_scaled, output_path)
    pipeline_log["output_path"] = output_path

    # ------------------------------------------------------------------
    # 7. Save artifacts (Task 4.12)
    # ------------------------------------------------------------------
    if save_artifacts:
        if verbose:
            print("\n" + "=" * 80)
            print("STEP 7: SAVE PREPROCESSING ARTIFACTS")
            print("=" * 80)

        # Save label encoders (ordinal mappings, target mapping, one-hot encoder)
        save_label_encoders(encoders_dict, LABEL_ENCODERS_PATH, verbose=verbose)

        # Save scaler
        if scaler is not None:
            scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
            save_scaler(scaler, scaler_path, verbose=verbose)

        # Save full pipeline config
        pipeline_config = {
            "cleaning_log": cleaning_log,
            "feature_engineering_log": fe_log,
            "encoding_log": encoding_log,
            "scaling": {"scaler_type": scaler_type},
            "final_shape": df_scaled.shape,
            "columns": list(df_scaled.columns),
        }
        with open(PIPELINE_PATH, "wb") as f:
            pickle.dump(pipeline_config, f)
        if verbose:
            print(f"Saved preprocessing pipeline config to: {PIPELINE_PATH}")

    return df_scaled, pipeline_log


if __name__ == "__main__":
    processed_df, log = run_preprocessing_pipeline()
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Final shape: {processed_df.shape}")
    print(f"Final columns: {list(processed_df.columns)}")
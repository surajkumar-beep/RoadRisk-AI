"""
Preprocessing package for the RoadRisk project.

Provides data cleaning, feature engineering, encoding, scaling,
and the main preprocessing pipeline for the RTA dataset.
"""

from preprocessing.utils import (
    DATASET_PATH,
    PROCESSED_DATASET_PATH,
    MODELS_DIR,
    LABEL_ENCODERS_PATH,
    PIPELINE_PATH,
    TARGET_MAPPING,
    ORDINAL_MAPPINGS,
    NOMINAL_FEATURES,
    NUMERICAL_FEATURES,
    DROP_FEATURES,
    clean_string_value,
    load_raw_dataset,
    save_dataframe,
)
from preprocessing.data_cleaning import (
    remove_duplicates,
    clean_inconsistent_values,
    impute_missing_values,
    drop_low_value_features,
    convert_data_types,
    run_data_cleaning,
)
from preprocessing.feature_engineering import (
    extract_hour_from_time,
    group_driving_experience,
    group_vehicle_age,
    run_feature_engineering,
)
from preprocessing.encoding import (
    encode_ordinal_features,
    encode_nominal_features,
    encode_target,
    save_label_encoders,
    run_encoding,
)
from preprocessing.scaling import (
    scale_numerical_features,
    save_scaler,
    run_scaling,
)
from preprocessing.preprocessing_pipeline import run_preprocessing_pipeline

__all__ = [
    # Paths
    "DATASET_PATH",
    "PROCESSED_DATASET_PATH",
    "MODELS_DIR",
    "LABEL_ENCODERS_PATH",
    "PIPELINE_PATH",
    # Mappings
    "TARGET_MAPPING",
    "ORDINAL_MAPPINGS",
    "NOMINAL_FEATURES",
    "NUMERICAL_FEATURES",
    "DROP_FEATURES",
    # Utils
    "clean_string_value",
    "load_raw_dataset",
    "save_dataframe",
    # Data cleaning
    "remove_duplicates",
    "clean_inconsistent_values",
    "impute_missing_values",
    "drop_low_value_features",
    "convert_data_types",
    "run_data_cleaning",
    # Feature engineering
    "extract_hour_from_time",
    "group_driving_experience",
    "group_vehicle_age",
    "run_feature_engineering",
    # Encoding
    "encode_ordinal_features",
    "encode_nominal_features",
    "encode_target",
    "save_label_encoders",
    "run_encoding",
    # Scaling
    "scale_numerical_features",
    "save_scaler",
    "run_scaling",
    # Pipeline
    "run_preprocessing_pipeline",
]
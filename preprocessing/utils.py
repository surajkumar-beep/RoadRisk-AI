"""
Utility functions for the preprocessing pipeline.

Provides common helpers, path constants, and value mapping dictionaries
used across all preprocessing modules.
"""
import os
import re
import pandas as pd

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "RTA Dataset.csv")
PROCESSED_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "processed_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LABEL_ENCODERS_PATH = os.path.join(MODELS_DIR, "label_encoders.pkl")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")

# ---------------------------------------------------------------------------
# Target Encoding Mapping
# ---------------------------------------------------------------------------
TARGET_MAPPING = {
    "Slight Injury": 0,
    "Serious Injury": 1,
    "Fatal injury": 2,
}

# ---------------------------------------------------------------------------
# Ordinal Features (Label Encoded) - original -> encoded value
# ---------------------------------------------------------------------------
ORDINAL_MAPPINGS = {
    "Age_band_of_driver": {
        "Under 18": 0,
        "18-30": 1,
        "31-50": 2,
        "Over 51": 3,
        "Unknown": -1,
    },
    "Driving_experience": {
        "No Licence": 0,
        "Below 1yr": 1,
        "1-2yr": 2,
        "2-5yr": 3,
        "5-10yr": 4,
        "Above 10yr": 5,
        "unknown": -1,
        "Unknown": -1,
    },
    "Service_year_of_vehicle": {
        "Below 5yr": 0,
        "5-10yr": 1,
        "Above 10yr": 2,
        "Unknown": -1,
    },
    "Educational_level": {
        "Illiterate": 0,
        "Writing & reading": 1,
        "Elementary school": 2,
        "Junior high school": 3,
        "High school": 4,
        "Above high school": 5,
        "Unknown": -1,
    },
    "Light_conditions": {
        "Daylight": 0,
        "Darkness - lights lit": 1,
        "Darkness - lights unlit": 2,
        "Darkness - no lighting": 3,
    },
    "Road_surface_conditions": {
        "Dry": 0,
        "Wet or damp": 1,
        "Snow": 2,
        "Flood over 3cm. deep": 3,
    },
    "Age_band_of_casualty": {
        "Under 18": 0,
        "5": 0,
        "18-30": 1,
        "31-50": 2,
        "Over 51": 3,
        "na": -1,
    },
    "Casualty_severity": {
        "1": 1,
        "2": 2,
        "3": 3,
        "na": -1,
    },
}

# ---------------------------------------------------------------------------
# Nominal Features (One-Hot Encoded)
# ---------------------------------------------------------------------------
NOMINAL_FEATURES = [
    "Day_of_week",
    "Sex_of_driver",
    "Vehicle_driver_relation",
    "Type_of_vehicle",
    "Owner_of_vehicle",
    "Defect_of_vehicle",
    "Area_accident_occured",
    "Lanes_or_Medians",
    "Road_allignment",
    "Types_of_Junction",
    "Road_surface_type",
    "Weather_conditions",
    "Type_of_collision",
    "Vehicle_movement",
    "Casualty_class",
    "Sex_of_casualty",
    "Work_of_casuality",
    "Fitness_of_casuality",
    "Pedestrian_movement",
    "Cause_of_accident",
]

# ---------------------------------------------------------------------------
# Numerical Features
# ---------------------------------------------------------------------------
NUMERICAL_FEATURES = [
    "Hour",
    "Number_of_vehicles_involved",
    "Number_of_casualties",
]

# ---------------------------------------------------------------------------
# Features to drop (high missing %, not useful)
# ---------------------------------------------------------------------------
DROP_FEATURES = [
    "Defect_of_vehicle",       # 4427/12316 missing (36%) - too high for useful imputation
]

# ---------------------------------------------------------------------------
# Features consumed by feature engineering
# ---------------------------------------------------------------------------
FEATURE_ENGINEERING_SOURCE = {
    "Time": "Hour",             # Time -> Hour extracted
}

# ---------------------------------------------------------------------------
# String Cleaning Patterns - known data errors in the dataset
# ---------------------------------------------------------------------------
# 'Pedestrian_movement' has corrupted concatenations like:
#   'Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle'
#   -> should be 'Crossing from nearside - masked by parked or stationary vehicle'
#   'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)'
#   -> should be 'In carriageway, stationary - not crossing (standing or playing)'
PEDESTRIAN_CORRUPTION_PATTERN = re.compile(r"statioNot a Pedestrianry")
PEDESTRIAN_CORRUPTION_REPLACEMENT = "stationary"

# 'Fitness_of_casuality' has 'NormalNormal' - a concatenation error
FITNESS_CORRUPTION_PATTERN = re.compile(r"^(Normal)+$")
FITNESS_CORRUPTION_REPLACEMENT = "Normal"

# 'Area_accident_occured' has 'Rural village areasOffice areas' - concatenation error
AREA_CORRUPTION_PATTERN = re.compile(r"^(.*?)Office areas$")
AREA_CORRUPTION_REPLACEMENT = "Rural village areas"

# 'Type_of_vehicle' has '?' encoding artifacts
TYPE_VEHICLE_CORRUPTION_PATTERN = re.compile(r"\?+Q|51\?100Q")
TYPE_VEHICLE_ARTIFACT_REPLACEMENTS = {
    r"41\?100Q": "41-100Q",
    r"11\?40Q": "11-40Q",
    r"\?": "-",
}


def clean_string_value(value, column=None):
    """
    Clean a single string value:
      1. Trim leading/trailing whitespace
      2. Collapse multiple internal spaces
      3. Fix known dataset inconsistencies
      4. Standardize 'unknown'/'other' casing
    """
    if pd.isna(value):
        return value

    s = str(value).strip()
    s = " ".join(s.split())

    # Fix pedestrian corruption ('statioNot a Pedestrianry' -> 'stationary')
    s = PEDESTRIAN_CORRUPTION_PATTERN.sub(PEDESTRIAN_CORRUPTION_REPLACEMENT, s)

    # Fix fitness corruption ('NormalNormal' -> 'Normal')
    s = FITNESS_CORRUPTION_PATTERN.sub(FITNESS_CORRUPTION_REPLACEMENT, s)

    # Fix area corruption ('Rural village areasOffice areas' -> 'Rural village areas')
    s = AREA_CORRUPTION_PATTERN.sub(AREA_CORRUPTION_REPLACEMENT, s)

    # Fix Type_of_vehicle '?' artifacts
    if column == "Type_of_vehicle":
        for pattern, replacement in TYPE_VEHICLE_ARTIFACT_REPLACEMENTS.items():
            s = re.sub(pattern, replacement, s)

    # Standardize lowercase 'unknown' / 'other' -> capitalized
    if s.lower() in ("unknown", "other"):
        s = s.lower().capitalize()  # -> 'Unknown' / 'Other'

    return s


def load_raw_dataset(path=DATASET_PATH):
    """Load the raw RTA dataset."""
    df = pd.read_csv(path)
    print(f"Raw dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def save_dataframe(df, path):
    """Save a DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"Saved dataset to: {path} ({df.shape[0]} rows, {df.shape[1]} columns)")
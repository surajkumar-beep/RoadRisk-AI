"""
Feature Engineering Module for the RTA preprocessing pipeline.

Handles:
  - Extract Hour from Time (4.6)
  - Group driving experience and vehicle age where beneficial (4.6)
"""
import pandas as pd


def extract_hour_from_time(df, verbose=True):
    """
    Extract the hour of day from the 'Time' column (HH:MM:SS format).

    Task 4.6: Extract Hour from Time.
    """
    df_fe = df.copy()

    if "Time" in df_fe.columns:
        # Parse time strings -> extract hour
        df_fe["Hour"] = pd.to_datetime(df_fe["Time"], format="%H:%M:%S").dt.hour
        # Drop the original Time column after extracting
        df_fe = df_fe.drop(columns=["Time"])

        if verbose:
            print("Extracted 'Hour' from 'Time' column.")
            print(df_fe["Hour"].describe())

    return df_fe


def group_driving_experience(df, verbose=True):
    """
    Create a grouped driving experience feature.

    Task 4.6: Group driving experience if beneficial.

    Groups: No Licence, 1-2yr, 2-5yr, 5-10yr, Above 10yr
    (Below 1yr is grouped into 1-2yr)
    """
    df_fe = df.copy()

    if "Driving_experience" in df_fe.columns:
        mapping = {
            "No Licence": "No Licence",
            "Below 1yr": "1-2yr",
            "1-2yr": "1-2yr",
            "2-5yr": "2-5yr",
            "5-10yr": "5-10yr",
            "Above 10yr": "Above 10yr",
            "unknown": "Unknown",
            "Unknown": "Unknown",
        }
        df_fe["Driving_experience_grouped"] = df_fe["Driving_experience"].map(mapping)
        # Replace original with grouped version
        df_fe["Driving_experience"] = df_fe["Driving_experience_grouped"]
        df_fe = df_fe.drop(columns=["Driving_experience_grouped"])

        if verbose:
            print("Grouped 'Driving_experience' (Below 1yr -> 1-2yr).")

    return df_fe


def group_vehicle_age(df, verbose=True):
    """
    Create a grouped vehicle age feature.

    Task 4.6: Group vehicle age if beneficial.

    Groups: Below 5yr, 5-10yr, Above 10yr
    (Below 1yr, 1-2yr, 2-5yrs grouped into Below 5yr)
    """
    df_fe = df.copy()

    if "Service_year_of_vehicle" in df_fe.columns:
        mapping = {
            "Below 1yr": "Below 5yr",
            "1-2yr": "Below 5yr",
            "2-5yrs": "Below 5yr",
            "5-10yrs": "5-10yr",
            "Above 10yr": "Above 10yr",
            "Unknown": "Unknown",
        }
        df_fe["Service_year_of_vehicle"] = df_fe["Service_year_of_vehicle"].map(mapping)

        if verbose:
            print("Grouped 'Service_year_of_vehicle' (Below 5yr, 5-10yr, Above 10yr).")

    return df_fe


def run_feature_engineering(df, verbose=True):
    """
    Run the full feature engineering pipeline:
      1. Extract Hour from Time
      2. Group driving experience
      3. Group vehicle age

    Returns feature-engineered DataFrame.
    """
    log = {}

    # 1. Extract Hour from Time
    df_fe = extract_hour_from_time(df, verbose=verbose)
    log["extracted_features"] = ["Hour (from Time)"]

    # 2. Group driving experience
    df_fe = group_driving_experience(df_fe, verbose=verbose)

    # 3. Group vehicle age
    df_fe = group_vehicle_age(df_fe, verbose=verbose)

    return df_fe, log
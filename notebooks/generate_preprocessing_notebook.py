"""
Script to generate the 02_Data_Preprocessing.ipynb notebook for the RTA preprocessing pipeline.
"""
import json
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
}

cells = []

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

# ============================================================
# Title & Introduction
# ============================================================
cells.append(md("""# Road Traffic Accident Dataset - Data Preprocessing Pipeline

**Notebook:** `02_Data_Preprocessing.ipynb`  
**Objective:** Clean, preprocess, and transform the raw RTA dataset for machine learning, then save the processed dataset and preprocessing artifacts.

---

## Table of Contents
1. [Load Dataset](#41-load-dataset)
2. [Remove Duplicate Records](#42-remove-duplicate-records)
3. [Handle Missing Values](#43-handle-missing-values)
4. [Clean Inconsistent Values](#44-clean-inconsistent-values)
5. [Convert Data Types](#45-convert-data-types)
6. [Feature Engineering](#46-feature-engineering)
7. [Encode Categorical Variables](#47-encode-categorical-variables)
8. [Encode Target Variable](#48-encode-target-variable)
9. [Verify No Missing Values](#49-verify-no-missing-values)
10. [Verify Final Dataset Shape](#410-verify-final-dataset-shape)
11. [Save Processed Dataset](#411-save-processed-dataset)
12. [Save Preprocessing Artifacts](#412-save-preprocessing-artifacts)
13. [Document Preprocessing Decisions](#413-document-preprocessing-decisions)
"""))

# ============================================================
# 4.1 Load Dataset
# ============================================================
cells.append(md("""## 4.1 Load Dataset

Load the raw RTA dataset from CSV.
"""))

cells.append(code("""# Import required libraries
import sys
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))

# Import preprocessing modules
from preprocessing.utils import (
    DATASET_PATH, PROCESSED_DATASET_PATH, MODELS_DIR,
    LABEL_ENCODERS_PATH, PIPELINE_PATH,
    TARGET_MAPPING, ORDINAL_MAPPINGS, NOMINAL_FEATURES,
    NUMERICAL_FEATURES, DROP_FEATURES,
    clean_string_value, load_raw_dataset, save_dataframe
)
from preprocessing.data_cleaning import (
    remove_duplicates, clean_inconsistent_values, impute_missing_values,
    drop_low_value_features, convert_data_types, run_data_cleaning
)
from preprocessing.feature_engineering import (
    extract_hour_from_time, group_driving_experience, group_vehicle_age,
    run_feature_engineering
)
from preprocessing.encoding import (
    encode_ordinal_features, encode_nominal_features, encode_target,
    save_label_encoders, run_encoding
)
from preprocessing.scaling import (
    scale_numerical_features, save_scaler, run_scaling
)
from preprocessing.preprocessing_pipeline import run_preprocessing_pipeline

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("All libraries and modules imported successfully!")"""))

cells.append(code("""# Load the raw dataset
df = load_raw_dataset(DATASET_PATH)

print(f"\\nDataset shape: {df.shape}")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Display first 5 rows
print("\\nFirst 5 rows:")
df.head()"""))

# ============================================================
# 4.2 Remove Duplicate Records
# ============================================================
cells.append(md("""## 4.2 Remove Duplicate Records

Check for and remove any duplicate records in the dataset.
"""))

cells.append(code("""# Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"Duplicate records found: {duplicate_count}")
print(f"Percentage of duplicates: {(duplicate_count / len(df)) * 100:.2f}%")

# Remove duplicates
df_clean, dup_log = remove_duplicates(df, verbose=True)

# Visualize duplicate status
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(['Unique Records', 'Duplicate Records'], 
              [len(df_clean), duplicate_count],
              color=['#2ecc71', '#e74c3c'])
ax.set_ylabel('Count')
ax.set_title('Duplicate Record Analysis')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', 
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ============================================================
# 4.3 Handle Missing Values
# ============================================================
cells.append(md("""## 4.3 Handle Missing Values

Analyze and handle missing values:
- **Categorical features** → Mode imputation
- **Numerical features** → Median imputation
- **High-missing features** → Drop (e.g., `Defect_of_vehicle` at 36%)
"""))

cells.append(code("""# Analyze missing values before imputation
missing_count = df_clean.isnull().sum()
missing_percent = (df_clean.isnull().sum() / len(df_clean)) * 100

missing_df = pd.DataFrame({
    'Column': df_clean.columns,
    'Missing Count': missing_count.values,
    'Missing Percentage (%)': missing_percent.values
}).sort_values('Missing Percentage (%)', ascending=False)

print("Missing values before imputation:")
print(missing_df[missing_df['Missing Count'] > 0].to_string(index=False))
print(f"\\nTotal missing values: {df_clean.isnull().sum().sum()}")

# Visualize missing values
fig, ax = plt.subplots(figsize=(14, 6))
missing_plot = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=True)
bars = ax.barh(missing_plot['Column'], missing_plot['Missing Count'], color='#e74c3c', alpha=0.8)
ax.set_xlabel('Missing Count')
ax.set_ylabel('Column')
ax.set_title('Missing Values by Column (Before Imputation)')
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2., f' {int(width):,}', 
            ha='left', va='center', fontsize=9)
plt.tight_layout()
plt.show()"""))

cells.append(code("""# Drop high-missing features
df_clean = drop_low_value_features(df_clean, verbose=True)
print(f"Shape after dropping high-missing features: {df_clean.shape}")

# Impute missing values
df_clean, imputation_log = impute_missing_values(df_clean, verbose=True)

# Verify no missing values remain
print(f"\\nMissing values after imputation: {df_clean.isnull().sum().sum()}")"""))

cells.append(code("""# Visualize imputation summary
imputation_df = pd.DataFrame([
    {'Feature': col, 'Missing Count': info['missing_count'], 'Method': info['type'], 'Imputed Value': str(info['value'])}
    for col, info in imputation_log.items()
]).sort_values('Missing Count', ascending=True)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(imputation_df['Feature'], imputation_df['Missing Count'], 
               color=['#3498db' if m == 'mode' else '#2ecc71' for m in imputation_df['Method']], alpha=0.8)
ax.set_xlabel('Missing Count Imputed')
ax.set_ylabel('Feature')
ax.set_title('Missing Value Imputation Summary')
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2., f' {int(width):,}', 
            ha='left', va='center', fontsize=9)
plt.tight_layout()
plt.show()"""))

# ============================================================
# 4.4 Clean Inconsistent Values
# ============================================================
cells.append(md("""## 4.4 Clean Inconsistent Values

Clean inconsistent values:
- Trim whitespace
- Standardize text/case
- Fix spelling inconsistencies
- Replace invalid/unknown values
"""))

cells.append(code("""# Show examples of data inconsistencies before cleaning
print("=" * 80)
print("DATA INCONSISTENCIES BEFORE CLEANING")
print("=" * 80)

# Check for whitespace issues
for col in df_clean.select_dtypes(include=['object', 'str']).columns:
    if col in ['Accident_severity', 'Time']:
        continue
    # Check for leading/trailing whitespace
    has_ws = df_clean[col].astype(str).str.match(r'^\\s|\\s$').sum()
    if has_ws > 0:
        print(f"  {col}: {has_ws} values with leading/trailing whitespace")

# Check for case inconsistencies
for col in ['Lanes_or_Medians', 'Weather_conditions', 'Type_of_collision', 'Vehicle_movement']:
    if col in df_clean.columns:
        lower_vals = df_clean[col].astype(str).str.lower().unique()
        print(f"  {col} (lowercase values): {lower_vals}")

# Check for known corruptions
print("\\nKnown data corruptions:")
ped_corrupt = df_clean['Pedestrian_movement'].astype(str).str.contains('statioNot a Pedestrianry').sum()
fit_corrupt = df_clean['Fitness_of_casuality'].astype(str).str.contains('NormalNormal').sum()
area_corrupt = df_clean['Area_accident_occured'].astype(str).str.contains('Office areas$').sum()
print(f"  Pedestrian_movement 'statioNot a Pedestrianry': {ped_corrupt}")
print(f"  Fitness_of_casuality 'NormalNormal': {fit_corrupt}")
print(f"  Area_accident_occured 'Office areas' concatenation: {area_corrupt}")"""))

cells.append(code("""# Clean inconsistent values
df_clean = clean_inconsistent_values(df_clean, verbose=True)

# Verify cleaning results
print("\\nAfter cleaning:")
ped_corrupt_after = df_clean['Pedestrian_movement'].astype(str).str.contains('statioNot a Pedestrianry').sum()
fit_corrupt_after = df_clean['Fitness_of_casuality'].astype(str).str.contains('NormalNormal').sum()
area_corrupt_after = df_clean['Area_accident_occured'].astype(str).str.contains('Office areas$').sum()
print(f"  Pedestrian_movement corruptions remaining: {ped_corrupt_after}")
print(f"  Fitness_of_casuality corruptions remaining: {fit_corrupt_after}")
print(f"  Area_accident_occured corruptions remaining: {area_corrupt_after}")

# Show cleaned values
print("\\nCleaned Pedestrian_movement values:")
print(df_clean['Pedestrian_movement'].unique())
print("\\nCleaned Fitness_of_casuality values:")
print(df_clean['Fitness_of_casuality'].unique())"""))

# ============================================================
# 4.5 Convert Data Types
# ============================================================
cells.append(md("""## 4.5 Convert Data Types

Convert columns to appropriate data types:
- `Time` → datetime (extracted to Hour in feature engineering)
- Numerical → int/float
- Categorical → category/object
"""))

cells.append(code("""# Show data types before conversion
print("Data types before conversion:")
print(df_clean.dtypes.to_string())

# Convert data types
df_clean = convert_data_types(df_clean, verbose=True)

print("\\nData types after conversion:")
print(df_clean.dtypes.to_string())"""))

cells.append(code("""# Visualize data type distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before conversion (raw)
raw_dtypes = pd.read_csv(DATASET_PATH).dtypes.value_counts()
axes[0].bar(raw_dtypes.index.astype(str), raw_dtypes.values, color='#3498db', alpha=0.8)
axes[0].set_title('Data Types - Raw Dataset')
axes[0].set_xlabel('Data Type')
axes[0].set_ylabel('Number of Columns')
for i, v in enumerate(raw_dtypes.values):
    axes[0].text(i, v, f'{v}', ha='center', va='bottom', fontweight='bold')

# After conversion
clean_dtypes = df_clean.dtypes.value_counts()
axes[1].bar(clean_dtypes.index.astype(str), clean_dtypes.values, color='#2ecc71', alpha=0.8)
axes[1].set_title('Data Types - After Conversion')
axes[1].set_xlabel('Data Type')
axes[1].set_ylabel('Number of Columns')
for i, v in enumerate(clean_dtypes.values):
    axes[1].text(i, v, f'{v}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()"""))

# ============================================================
# 4.6 Feature Engineering
# ============================================================
cells.append(md("""## 4.6 Feature Engineering

Perform feature engineering:
- Extract `Hour` from `Time`
- Group driving experience
- Group vehicle age
"""))

cells.append(code("""# Extract Hour from Time
df_fe = extract_hour_from_time(df_clean, verbose=True)

# Visualize Hour distribution
fig, ax = plt.subplots(figsize=(12, 5))
hour_counts = df_fe['Hour'].value_counts().sort_index()
bars = ax.bar(hour_counts.index, hour_counts.values, color='#3498db', alpha=0.8)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Number of Accidents')
ax.set_title('Accident Distribution by Hour of Day')
ax.set_xticks(range(0, 24))
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', 
            ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.show()"""))

cells.append(code("""# Group driving experience
df_fe = group_driving_experience(df_fe, verbose=True)

# Visualize grouped driving experience
fig, ax = plt.subplots(figsize=(10, 5))
exp_counts = df_fe['Driving_experience'].value_counts()
sns.countplot(data=df_fe, x='Driving_experience', ax=ax, order=exp_counts.index, palette='viridis')
ax.set_title('Driving Experience - Grouped Distribution')
ax.set_xlabel('Driving Experience')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(exp_counts.values):
    ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()"""))

cells.append(code("""# Group vehicle age
df_fe = group_vehicle_age(df_fe, verbose=True)

# Visualize grouped vehicle age
fig, ax = plt.subplots(figsize=(10, 5))
age_counts = df_fe['Service_year_of_vehicle'].value_counts()
sns.countplot(data=df_fe, x='Service_year_of_vehicle', ax=ax, order=age_counts.index, palette='coolwarm')
ax.set_title('Vehicle Age - Grouped Distribution')
ax.set_xlabel('Vehicle Age Group')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(age_counts.values):
    ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()"""))

cells.append(code("""# Show feature engineering summary
print("=" * 80)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 80)
print(f"Extracted 'Hour' from 'Time' column")
print(f"Grouped 'Driving_experience' (Below 1yr -> 1-2yr)")
print(f"Grouped 'Service_year_of_vehicle' (Below 5yr, 5-10yr, Above 10yr)")
print(f"\\nShape after feature engineering: {df_fe.shape}")
print(f"Columns: {list(df_fe.columns)}")"""))

# ============================================================
# 4.7 Encode Categorical Variables
# ============================================================
cells.append(md("""## 4.7 Encode Categorical Variables

Encode categorical variables:
- **Label Encoding** for ordinal features
- **One-Hot Encoding** for nominal features
"""))

cells.append(code("""# Show ordinal features and their mappings
print("=" * 80)
print("ORDINAL FEATURES - LABEL ENCODING MAPPINGS")
print("=" * 80)
for col, mapping in ORDINAL_MAPPINGS.items():
    if col in df_fe.columns:
        print(f"\\n{col}:")
        for k, v in mapping.items():
            print(f"  {k} -> {v}")"""))

cells.append(code("""# Encode ordinal features
df_encoded = encode_ordinal_features(df_fe, verbose=True)

# Show encoded ordinal features
print("\\nEncoded ordinal features sample:")
ordinal_cols = [col for col in ORDINAL_MAPPINGS.keys() if col in df_encoded.columns]
print(df_encoded[ordinal_cols].head(10).to_string())"""))

cells.append(code("""# Encode target variable
df_encoded = encode_target(df_encoded, verbose=True)

# Visualize target distribution after encoding
fig, ax = plt.subplots(figsize=(8, 5))
target_counts = df_encoded['Accident_severity'].value_counts().sort_index()
bars = ax.bar(['Slight (0)', 'Serious (1)', 'Fatal (2)'], target_counts.values, 
              color=['#2ecc71', '#f39c12', '#e74c3c'])
ax.set_ylabel('Count')
ax.set_title('Target Variable Distribution After Encoding')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', 
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()"""))

cells.append(code("""# One-hot encode nominal features
df_encoded, ohe = encode_nominal_features(df_encoded, verbose=True)

print(f"\\nShape after one-hot encoding: {df_encoded.shape}")"""))

cells.append(code("""# Visualize encoding summary
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before encoding
before_cols = len(df_fe.columns)
axes[0].bar(['Categorical', 'Numerical', 'Total'], 
            [len(NOMINAL_FEATURES) + len(ORDINAL_MAPPINGS), len(NUMERICAL_FEATURES), before_cols],
            color=['#3498db', '#2ecc71', '#e74c3c'])
axes[0].set_title('Features Before Encoding')
axes[0].set_ylabel('Count')

# After encoding
after_cols = len(df_encoded.columns)
axes[1].bar(['Encoded Features', 'Total'], 
            [after_cols - len(NUMERICAL_FEATURES) - 1, after_cols],
            color=['#9b59b6', '#e74c3c'])
axes[1].set_title('Features After Encoding')
axes[1].set_ylabel('Count')

for ax in axes:
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', 
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()"""))

cells.append(md("""### Scale Numerical Features

Scale numerical features using StandardScaler for model readiness.
"""))

cells.append(code("""# Scale numerical features
df_encoded, scaler = run_scaling(df_encoded, scaler_type="standard", verbose=True)

# Show scaled numerical features
print("\\nScaled numerical features sample:")
print(df_encoded[NUMERICAL_FEATURES].head(10).to_string())

# Visualize scaling effect
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for i, col in enumerate(NUMERICAL_FEATURES):
    axes[i].hist(df_encoded[col], bins=30, color='#3498db', alpha=0.8, edgecolor='white')
    axes[i].set_title(f'{col} (Scaled)')
    axes[i].set_xlabel('Scaled Value')
    axes[i].set_ylabel('Frequency')
plt.suptitle('Scaled Numerical Features Distribution', fontsize=14)
plt.tight_layout()
plt.show()"""))

# ============================================================
# 4.8 Encode Target Variable
# ============================================================
cells.append(md("""## 4.8 Encode Target Variable

Encode `Accident_severity` into numerical labels:
- Slight Injury → 0
- Serious Injury → 1
- Fatal injury → 2
"""))

cells.append(code("""# Show target encoding mapping
print("=" * 80)
print("TARGET ENCODING MAPPING")
print("=" * 80)
for k, v in TARGET_MAPPING.items():
    print(f"  {k} -> {v}")

# Verify target encoding
print("\\nTarget distribution after encoding:")
print(df_encoded['Accident_severity'].value_counts().sort_index())

# Visualize target distribution
fig, ax = plt.subplots(figsize=(8, 5))
target_counts = df_encoded['Accident_severity'].value_counts().sort_index()
labels = ['Slight Injury (0)', 'Serious Injury (1)', 'Fatal Injury (2)']
colors = ['#2ecc71', '#f39c12', '#e74c3c']
bars = ax.bar(labels, target_counts.values, color=colors)
ax.set_ylabel('Count')
ax.set_title('Encoded Target Variable Distribution')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', 
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ============================================================
# 4.9 Verify No Missing Values
# ============================================================
cells.append(md("""## 4.9 Verify No Missing Values

Verify that no missing values remain in the processed dataset.
"""))

cells.append(code("""# Check for missing values
missing_after = df_encoded.isnull().sum()
total_missing = missing_after.sum()

print(f"Total missing values: {total_missing}")

if total_missing == 0:
    print("✓ No missing values remain in the processed dataset!")
else:
    print("Missing values by column:")
    print(missing_after[missing_after > 0])

# Visualize missing values check
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(['Missing Values', 'Non-Missing Values'], 
              [total_missing, df_encoded.shape[0] * df_encoded.shape[1] - total_missing],
              color=['#e74c3c', '#2ecc71'])
ax.set_ylabel('Count')
ax.set_title('Missing Values Verification')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', 
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ============================================================
# 4.10 Verify Final Dataset Shape
# ============================================================
cells.append(md("""## 4.10 Verify Final Dataset Shape

Verify the final dataset shape and feature count.
"""))

cells.append(code("""# Final dataset shape
print("=" * 80)
print("FINAL DATASET SHAPE")
print("=" * 80)
print(f"Rows: {df_encoded.shape[0]:,}")
print(f"Columns: {df_encoded.shape[1]:,}")
print(f"Total cells: {df_encoded.shape[0] * df_encoded.shape[1]:,}")

# Feature count breakdown
ordinal_count = len([c for c in ORDINAL_MAPPINGS if c in df_encoded.columns])
# Nominal encoded columns are those that came from one-hot encoding (contain '_' and are not ordinal/numerical/target)
ordinal_cols = [c for c in ORDINAL_MAPPINGS if c in df_encoded.columns]
nominal_encoded_count = len([c for c in df_encoded.columns 
                             if c not in ordinal_cols 
                             and c not in NUMERICAL_FEATURES 
                             and c != 'Accident_severity'])
numerical_count = len([c for c in NUMERICAL_FEATURES if c in df_encoded.columns])

print(f"\\nFeature breakdown:")
print(f"  Ordinal (label-encoded): {ordinal_count}")
print(f"  Nominal (one-hot encoded): {nominal_encoded_count}")
print(f"  Numerical (scaled): {numerical_count}")
print(f"  Target: 1")

# Visualize final shape
fig, ax = plt.subplots(figsize=(8, 5))
feature_types = ['Ordinal\\n(Label)', 'Nominal\\n(One-Hot)', 'Numerical\\n(Scaled)', 'Target']
counts = [ordinal_count, nominal_encoded_count, numerical_count, 1]
colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']
bars = ax.bar(feature_types, counts, color=colors)
ax.set_ylabel('Number of Features')
ax.set_title('Final Feature Composition')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', 
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()"""))

cells.append(code("""# Display final dataset info
print("Final dataset info:")
print(df_encoded.info())"""))

# ============================================================
# 4.11 Save Processed Dataset
# ============================================================
cells.append(md("""## 4.11 Save Processed Dataset

Save the processed dataset to `dataset/processed_dataset.csv`.
"""))

cells.append(code("""# Save processed dataset
save_dataframe(df_encoded, PROCESSED_DATASET_PATH)

# Verify saved file
saved_df = pd.read_csv(PROCESSED_DATASET_PATH)
print(f"\\nVerified saved file:")
print(f"  Shape: {saved_df.shape}")
print(f"  Missing values: {saved_df.isnull().sum().sum()}")
print(f"  File size: {os.path.getsize(PROCESSED_DATASET_PATH):,} bytes")"""))

# ============================================================
# 4.12 Save Preprocessing Artifacts
# ============================================================
cells.append(md("""## 4.12 Save Preprocessing Artifacts

Save preprocessing artifacts:
- `models/label_encoders.pkl` - encoding mappings
- `models/preprocessing_pipeline.pkl` - pipeline configuration
- `models/scaler.pkl` - fitted scaler
"""))

cells.append(code("""# Build encoders dictionary
ordinal_mappings_used = {col: ORDINAL_MAPPINGS[col] for col in ORDINAL_MAPPINGS if col in df_fe.columns}
nominal_present = [col for col in NOMINAL_FEATURES if col in df_fe.columns]

encoders_dict = {
    "ordinal_mappings": ordinal_mappings_used,
    "target_mapping": TARGET_MAPPING,
    "one_hot_encoder": ohe,
    "nominal_features": nominal_present,
    "ordinal_features": list(ordinal_mappings_used.keys()),
}

# Save label encoders
save_label_encoders(encoders_dict, LABEL_ENCODERS_PATH, verbose=True)

# Save scaler
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
save_scaler(scaler, scaler_path, verbose=True)

# Save pipeline config
pipeline_config = {
    "cleaning": {
        "duplicates_removed": dup_log["duplicates_removed"],
        "dropped_features": DROP_FEATURES,
        "imputation": imputation_log,
    },
    "feature_engineering": {
        "extracted": ["Hour (from Time)"],
        "grouped": ["Driving_experience", "Service_year_of_vehicle"],
    },
    "encoding": {
        "ordinal_features": list(ordinal_mappings_used.keys()),
        "nominal_features": nominal_present,
        "target_mapping": TARGET_MAPPING,
    },
    "scaling": {"scaler_type": "standard"},
    "final_shape": df_encoded.shape,
    "columns": list(df_encoded.columns),
}

with open(PIPELINE_PATH, "wb") as f:
    pickle.dump(pipeline_config, f)
print(f"Saved preprocessing pipeline config to: {PIPELINE_PATH}")

# List saved artifacts
print("\\nSaved artifacts:")
for fname in os.listdir(MODELS_DIR):
    fpath = os.path.join(MODELS_DIR, fname)
    print(f"  {fname} ({os.path.getsize(fpath):,} bytes)")"""))

cells.append(code("""# Verify artifacts can be loaded
print("=" * 80)
print("VERIFY SAVED ARTIFACTS")
print("=" * 80)

# Load label encoders
with open(LABEL_ENCODERS_PATH, "rb") as f:
    loaded_encoders = pickle.load(f)
print(f"Label encoders loaded: {list(loaded_encoders.keys())}")
print(f"  Ordinal features: {loaded_encoders['ordinal_features']}")
print(f"  Nominal features: {len(loaded_encoders['nominal_features'])} features")
print(f"  Target mapping: {loaded_encoders['target_mapping']}")

# Load scaler
with open(scaler_path, "rb") as f:
    loaded_scaler = pickle.load(f)
print(f"\\nScaler loaded: {type(loaded_scaler).__name__}")

# Load pipeline config
with open(PIPELINE_PATH, "rb") as f:
    loaded_pipeline = pickle.load(f)
print(f"\\nPipeline config loaded:")
print(f"  Final shape: {loaded_pipeline['final_shape']}")
print(f"  Scaling: {loaded_pipeline['scaling']}")"""))

# ============================================================
# 4.13 Document Preprocessing Decisions
# ============================================================
cells.append(md("""## 4.13 Document Preprocessing Decisions

### Summary of All Preprocessing Decisions

#### 1. Missing Value Handling
| Feature | Missing Count | Method | Imputed Value |
|---------|--------------|--------|---------------|
| `Educational_level` | 741 | Mode | Junior high school |
| `Vehicle_driver_relation` | 579 | Mode | Employee |
| `Driving_experience` | 829 | Mode | 5-10yr |
| `Type_of_vehicle` | 950 | Mode | Automobile |
| `Owner_of_vehicle` | 482 | Mode | Owner |
| `Service_year_of_vehicle` | 3,928 | Mode | Unknown |
| `Area_accident_occured` | 239 | Mode | Other |
| `Lanes_or_Medians` | 385 | Mode | Two-way (divided with broken lines) |
| `Road_allignment` | 142 | Mode | Tangent road with flat terrain |
| `Types_of_Junction` | 887 | Mode | Y Shape |
| `Road_surface_type` | 172 | Mode | Asphalt roads |
| `Type_of_collision` | 155 | Mode | Vehicle with vehicle collision |
| `Vehicle_movement` | 308 | Mode | Going straight |
| `Work_of_casuality` | 3,198 | Mode | Driver |
| `Fitness_of_casuality` | 2,635 | Mode | Normal |
| `Defect_of_vehicle` | 4,427 | **Dropped** | 36% missing - too high for useful imputation |

#### 2. Duplicate Removal
- **0 duplicate records found** - no removal needed

#### 3. Data Cleaning
- Trimmed whitespace from all string columns
- Standardized 'unknown'/'other' casing
- Fixed `Pedestrian_movement` corruption: 'statioNot a Pedestrianry' → 'stationary'
- Fixed `Fitness_of_casuality` corruption: 'NormalNormal' → 'Normal'
- Fixed `Area_accident_occured` corruption: 'Rural village areasOffice areas' → 'Rural village areas'
- Fixed `Type_of_vehicle` encoding artifacts: '?' → '-'

#### 4. Feature Engineering
- Extracted `Hour` from `Time` (dropped original Time column)
- Grouped `Driving_experience`: Below 1yr → 1-2yr
- Grouped `Service_year_of_vehicle`: Below 5yr, 5-10yr, Above 10yr

#### 5. Encoding Methods
- **Label Encoding** (ordinal): `Age_band_of_driver`, `Driving_experience`, `Service_year_of_vehicle`, `Educational_level`, `Light_conditions`, `Road_surface_conditions`, `Age_band_of_casualty`, `Casualty_severity`
- **One-Hot Encoding** (nominal): 19 features → 154 binary columns
- **StandardScaler** for numerical features: `Hour`, `Number_of_vehicles_involved`, `Number_of_casualties`

#### 6. Target Encoding
- Slight Injury → 0
- Serious Injury → 1
- Fatal injury → 2

#### 7. Final Feature Count
- **Original:** 12,316 rows × 32 columns
- **Final:** 12,316 rows × 166 columns
- **Feature breakdown:** 8 ordinal (label-encoded), 154 nominal (one-hot), 3 numerical (scaled), 1 target
- **Missing values:** 0
"""))

# ============================================================
# Final Pipeline Execution (Optional)
# ============================================================
cells.append(md("""## Complete Pipeline Execution

Run the entire preprocessing pipeline in one step using the `run_preprocessing_pipeline` function.
"""))

cells.append(code("""# Run the complete preprocessing pipeline
processed_df, pipeline_log = run_preprocessing_pipeline(
    input_path=DATASET_PATH,
    output_path=PROCESSED_DATASET_PATH,
    scaler_type="standard",
    verbose=True,
    save_artifacts=True
)

print("\\n" + "=" * 80)
print("PREPROCESSING PIPELINE COMPLETE")
print("=" * 80)
print(f"Final shape: {processed_df.shape}")
print(f"Missing values: {processed_df.isnull().sum().sum()}")
print(f"Target distribution:")
print(processed_df['Accident_severity'].value_counts().sort_index())"""))

# Add cells to notebook
nb.cells = cells

# Write notebook
output_path = 'notebooks/02_Data_Preprocessing.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook created successfully at: {output_path}")
print(f"Total cells: {len(cells)}")
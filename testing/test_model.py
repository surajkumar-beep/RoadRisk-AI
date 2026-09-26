"""
STEP 6 - Model Testing & Prediction.

Evaluates the SAVED Step 5 model on the held-out 20% test set (same split,
no re-fitting) and provides prediction for completely new accident records.

Reuses the exact Step 5/Step 4 conventions:
  - split: train_test_split(test_size=0.2, stratify=y, random_state=42)
  - target: Accident_severity (Slight Injury=0, Serious Injury=1, Fatal injury=2)
  - metrics: weighted (primary) + macro; ROC-AUC OvR macro
  - preprocessing artifacts: models/*.pkl produced by Steps 4-5 (no fitting here)

Outputs written to reports/:
  - test_confusion_matrix.png   (this step's plot; Step 5 file untouched)
  - test_classification_report.txt
  - step6_test_results.json
"""
import os
import pickle
import sys
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Make the project root importable when run as a bare script.
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import matplotlib

# Only force the headless Agg backend when running as a standalone CLI script;
# inside a notebook kernel we keep the inline backend so plt.show() embeds figures.
try:
    from IPython import get_ipython

    _IN_IPYTHON = get_ipython() is not None
except Exception:  # noqa: BLE001
    _IN_IPYTHON = False

if not _IN_IPYTHON:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.constants import TARGET_COLUMN, RANDOM_STATE, TEST_SIZE
from utils.helpers import load_dataset
from utils.logger import logger

from preprocessing.utils import NUMERICAL_FEATURES
from preprocessing.feature_engineering import (
    extract_hour_from_time,
    group_driving_experience,
    group_vehicle_age,
)
from preprocessing.encoding import encode_ordinal_features
from training.metrics import compute_classification_metrics
from training.save_model import (
    MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    SCALER_PATH,
    LABEL_ENCODERS_PATH,
)

PROCESSED_DATASET_PATH = os.path.join(_BASE, "dataset", "processed_dataset.csv")
REPORTS_DIR = os.path.join(_BASE, "reports")

# ---------------------------------------------------------------------------
# 1. Load saved artifacts
# ---------------------------------------------------------------------------


def load_artifacts(
    model_path=MODEL_PATH,
    feature_columns_path=FEATURE_COLUMNS_PATH,
    scaler_path=SCALER_PATH,
    encoders_path=LABEL_ENCODERS_PATH,
):
    """Load and verify every Step 5 artifact (model, features, scaler, encoders)."""
    artifacts = {}
    checks = {
        "model": model_path,
        "feature_columns": feature_columns_path,
        "scaler": scaler_path,
        "label_encoders": encoders_path,
    }
    for name, path in checks.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing artifact: {name} -> {path}")

    with open(model_path, "rb") as f:
        artifacts["model"] = pickle.load(f)
    with open(feature_columns_path, "rb") as f:
        artifacts["feature_columns"] = pickle.load(f)
    with open(scaler_path, "rb") as f:
        artifacts["scaler"] = pickle.load(f)
    with open(encoders_path, "rb") as f:
        artifacts["encoders"] = pickle.load(f)

    # Actual class labels derived from the saved target mapping (never hardcoded).
    target_mapping = artifacts["encoders"]["target_mapping"]
    code_to_label = {int(v): k for k, v in target_mapping.items()}
    artifacts["class_labels"] = [code_to_label[c] for c in sorted(code_to_label)]

    for name in checks:
        logger.info(f"Loaded artifact -> {name}: {checks[name]}")
    logger.info(f"Class labels (from encoders): {artifacts['class_labels']}")
    return artifacts


# ---------------------------------------------------------------------------
# 2. Load test data (recreate the SAME Step 5 split; no fitting)
# ---------------------------------------------------------------------------


def load_test_data(path=PROCESSED_DATASET_PATH):
    """Load the processed dataset and recreate the exact Step 5 80/20 split."""
    df = load_dataset(path)
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(
        f"Recreated Step 5 split: train={len(X_train)}, test={len(X_test)} "
        f"(random_state={RANDOM_STATE}, test_size={TEST_SIZE})"
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 4. Evaluation (same averaging strategy as Step 5)
# ---------------------------------------------------------------------------


def evaluate_model(model, X_test, y_test, class_labels):
    """Compute all required metrics on the held-out test set (no fitting)."""
    y_pred = model.predict(X_test)
    metrics = compute_classification_metrics(y_test, y_pred)  # weighted + macro

    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(
        y_test, y_pred, target_names=class_labels, zero_division=0, digits=4
    )

    roc_auc = None
    try:
        if hasattr(model, "predict_proba"):
            roc_auc = float(
                roc_auc_score(
                    y_test,
                    model.predict_proba(X_test),
                    multi_class="ovr",
                    average="macro",
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"ROC-AUC not computed: {exc}")

    return {
        "metrics": metrics,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_text,
        "y_pred": y_pred,
    }


def predict_with_proba(model, X, class_labels):
    """Return predicted label, class indices and per-class probabilities."""
    proba = model.predict_proba(X)
    pred_idx = proba.argmax(axis=1)
    pred_labels = [class_labels[i] for i in pred_idx]
    return pred_labels, proba


def print_prediction(actual_label, pred_label, proba_row, class_labels):
    """Pretty-print one prediction with per-class probabilities."""
    print(f"  Actual   : {actual_label}")
    print(f"  Predicted: {pred_label}")
    print("  Probability:")
    for i, label in enumerate(class_labels):
        pct = proba_row[i] * 100.0
        marker = " <--" if label == pred_label else ""
        print(f"    {label:16s} {pct:5.1f}%{marker}")
    print("-" * 40)


# ---------------------------------------------------------------------------
# New-record preprocessing (uses ONLY saved artifacts - never fits)
# ---------------------------------------------------------------------------


def _ohe_column_names(ohe, nominal_features):
    """Replicate the exact column-name cleaning used in Step 4's encoding."""
    names = ohe.get_feature_names_out(nominal_features)
    names = [n.replace(" ", "_").replace("(", "").replace(")", "") for n in names]
    return [n.replace("[", "").replace("]", "") for n in names]


def preprocess_new_record(raw_record, artifacts):
    """
    Transform ONE raw accident record into model features using the saved
    preprocessing artifacts (ordinal mappings, one-hot encoder, scaler).

    No fitting is performed - everything reuses the artifacts saved in Step 5.
    `raw_record` must use the raw dataset schema (see notebooks/01 for columns).
    """
    feature_columns = artifacts["feature_columns"]
    encoders = artifacts["encoders"]
    scaler = artifacts["scaler"]

    df = pd.DataFrame([raw_record])
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])
    # Step 4 rule: Defect_of_vehicle is dropped during cleaning.
    if "Defect_of_vehicle" in df.columns:
        df = df.drop(columns=["Defect_of_vehicle"])

    # Feature engineering (same as Step 4)
    df = extract_hour_from_time(df, verbose=False)
    df = group_driving_experience(df, verbose=False)
    df = group_vehicle_age(df, verbose=False)

    # Ordinal encoding using the saved mappings (no fit).
    df = encode_ordinal_features(df, verbose=False)
    # Neutral fill for any category that was 'Unknown' / not in mappings.
    for col in encoders["ordinal_features"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

    nominal_features = [c for c in encoders["nominal_features"] if c in df.columns]
    ohe = encoders["one_hot_encoder"]
    ohe_array = ohe.transform(df[nominal_features])  # handle_unknown -> zeros
    ohe_df = pd.DataFrame(
        ohe_array, columns=_ohe_column_names(ohe, nominal_features), index=df.index
    )
    df = df.drop(columns=nominal_features)

    # Numerical scaling with the saved scaler (fit on train only - reused).
    numeric_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    if scaler is not None and numeric_cols:
        df[numeric_cols] = scaler.transform(df[numeric_cols].astype(float))

    # Combine all feature blocks and reorder to the model's feature columns.
    df = pd.concat([df, ohe_df], axis=1)
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df.astype(float)


def predict_new_record(raw_record, artifacts):
    """Predict severity + probabilities for ONE completely new accident record."""
    X_new = preprocess_new_record(raw_record, artifacts)
    pred_label, proba = predict_with_proba(artifacts["model"], X_new, artifacts["class_labels"])
    return pred_label[0], proba[0], X_new

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(
        f"Recreated Step 5 split: train={len(X_train)}, test={len(X_test)} "
        f"(random_state={RANDOM_STATE}, test_size={TEST_SIZE})"
    )

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def save_test_confusion_matrix(cm, class_labels, path):
    """Save a labelled confusion-matrix heatmap (does not touch Step 5's plot)."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    import numpy as np

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_labels)),
        yticks=np.arange(len(class_labels)),
        xticklabels=class_labels,
        yticklabels=class_labels,
        xlabel="Predicted",
        ylabel="Actual",
        title="Confusion Matrix - Test Set (held-out 20%)",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[0]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved test confusion matrix -> {path}")


def save_step6_results(results, path=os.path.join(REPORTS_DIR, "step6_test_results.json")):
    """Persist test metrics as JSON (overwrites the stale Step 6 file)."""
    import json

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved test results -> {path}")


# ---------------------------------------------------------------------------
# 3. Preprocess test data (order-matching only; no fitting)
# ---------------------------------------------------------------------------


def preprocess_test_data(X_test, feature_columns):
    """Match test feature order to models/feature_columns.pkl (no fitting)."""
    missing = [c for c in feature_columns if c not in X_test.columns]
    if missing:
        raise ValueError(f"Test features missing columns present in model: {missing[:10]}")
    extra = [c for c in X_test.columns if c not in feature_columns]
    if extra:
        logger.warning(f"Dropping {len(extra)} extra test columns not in model: {extra[:5]}")
    X_test = X_test[feature_columns]
    logger.info(
        f"Test set aligned to feature_columns: {X_test.shape[1]} features "
        f"(order match: {list(X_test.columns) == list(feature_columns)})"
    )
    return X_test

# ---------------------------------------------------------------------------
# Example NEW accident record (illustrative, NOT from the training/test data)
# ---------------------------------------------------------------------------

# NOTE: This is a synthetic example record used only to demonstrate the
# prediction API. It is clearly NOT a real accident observation.
EXAMPLE_NEW_RECORD = {
    "Time": "14:30:00",
    "Day_of_week": "Saturday",
    "Age_band_of_driver": "31-50",
    "Sex_of_driver": "Male",
    "Educational_level": "High school",
    "Vehicle_driver_relation": "Owner",
    "Driving_experience": "5-10yr",
    "Type_of_vehicle": "Automobile",
    "Owner_of_vehicle": "Owner",
    "Service_year_of_vehicle": "5-10yr",
    "Defect_of_vehicle": None,
    "Area_accident_occured": "Residential areas",
    "Lanes_or_Medians": "Two-way (divided with broken lines)",
    "Road_allignment": "Tangent road with flat terrain",
    "Types_of_Junction": "Y Shape",
    "Road_surface_type": "Asphalt roads",
    "Road_surface_conditions": "Dry",
    "Light_conditions": "Daylight",
    "Weather_conditions": "Normal",
    "Type_of_collision": "Vehicle with vehicle collision",
    "Number_of_vehicles_involved": 2,
    "Number_of_casualties": 1,
    "Vehicle_movement": "Going straight",
    "Casualty_class": "Driver or rider",
    "Sex_of_casualty": "Male",
    "Age_band_of_casualty": "31-50",
    "Casualty_severity": "2",
    "Work_of_casuality": "Driver",
    "Fitness_of_casuality": "Normal",
    "Pedestrian_movement": "Not a Pedestrian",
    "Cause_of_accident": "Driving at high speed",
}


def run_step6(sample_show=5):
    """Run the full Step 6 pipeline: load -> split -> eval -> predict -> reports."""
    print("=" * 80)
    print("STEP 6: MODEL TESTING & PREDICTION")
    print("=" * 80)

    # 1. Load artifacts
    artifacts = load_artifacts()
    model = artifacts["model"]
    class_labels = artifacts["class_labels"]
    feature_columns = artifacts["feature_columns"]
    print(f"Model loaded: {type(model).__name__} | features: {len(feature_columns)}")
    print(f"Class labels: {class_labels}")

    # 2. Load test data + recreate split, 3. align features (no fitting)
    X_train, X_test, y_train, y_test = load_test_data()
    X_test = preprocess_test_data(X_test, feature_columns)

    # 4. Evaluate
    results = evaluate_model(model, X_test, y_test, class_labels)
    m = results["metrics"]

    print("\n" + "=" * 80)
    print("FINAL TEST METRICS (held-out 20%, weighted + macro)")
    print("=" * 80)
    print(f"  Accuracy              : {m['accuracy']:.4f}")
    print(f"  Precision (weighted)  : {m['precision_weighted']:.4f}")
    print(f"  Recall    (weighted)  : {m['recall_weighted']:.4f}")
    print(f"  F1        (weighted)  : {m['f1_weighted']:.4f}")
    print(f"  Precision (macro)     : {m['precision_macro']:.4f}")
    print(f"  Recall    (macro)     : {m['recall_macro']:.4f}")
    print(f"  F1        (macro)     : {m['f1_macro']:.4f}")
    print(f"  ROC-AUC (OvR macro)   : {results['roc_auc']}")

    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    print(results["classification_report"])

    # 5. Confusion matrix report
    cm_path = os.path.join(REPORTS_DIR, "test_confusion_matrix.png")
    save_test_confusion_matrix(results["confusion_matrix"], class_labels, cm_path)
    cr_path = os.path.join(REPORTS_DIR, "test_classification_report.txt")
    with open(cr_path, "w", encoding="utf-8") as f:
        f.write(results["classification_report"])
    logger.info(f"Saved classification report -> {cr_path}")

    save_step6_results(
        {
            "model": type(model).__name__,
            "samples_tested": int(len(X_test)),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "accuracy": m["accuracy"],
            "precision_weighted": m["precision_weighted"],
            "recall_weighted": m["recall_weighted"],
            "f1_weighted": m["f1_weighted"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"],
            "roc_auc_ovr_macro": results["roc_auc"],
        }
    )


    # 6. Sample test predictions with probabilities
    print("\n" + "=" * 80)
    print(f"SAMPLE PREDICTIONS FROM THE TEST SET (first {sample_show})")
    print("=" * 80)
    proba = model.predict_proba(X_test)
    y_pred_idx = proba.argmax(axis=1)
    for i in range(min(sample_show, len(X_test))):
        actual = class_labels[int(y_test.iloc[i])]
        pred = class_labels[y_pred_idx[i]]
        print_prediction(actual, pred, proba[i], class_labels)

    # 7. Completely new accident record
    print("\n" + "=" * 80)
    print("PREDICTION FOR A COMPLETELY NEW ACCIDENT RECORD (synthetic example)")
    print("=" * 80)
    pred_label, proba_row, X_new = predict_new_record(EXAMPLE_NEW_RECORD, artifacts)
    print(f"  New-record feature vector shape: {X_new.shape[1]} "
          f"(matches model: {X_new.shape[1] == len(feature_columns)})")
    print_prediction("(no ground truth - new/unseen record)", pred_label, proba_row, class_labels)

    print("\nSTEP 6 COMPLETE.")
    print(f"Outputs: {cm_path} | {cr_path} | {os.path.join(REPORTS_DIR, 'step6_test_results.json')}")
    return results


if __name__ == "__main__":
    run_step6()


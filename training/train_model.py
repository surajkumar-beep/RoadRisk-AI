"""
STEP 5 – Accident Severity Prediction Model (XGBoost) training pipeline.

Orchestrates:
  1. Load processed dataset (already cleaned/encoded/scaled by Step 4)
  2. Feature/target separation + train/test split (stratified, fixed seed)
  3. Baselines: Logistic Regression, Decision Tree, Random Forest
  4. Initial XGBoost training
  5. Hyperparameter tuning (RandomizedSearchCV)
  6. Model selection (best F1-weighted) + persistence
  7. Reports: confusion_matrix.png, feature_importance.png,
     classification_report.txt, model_metrics.json

Averaging choice: weighted is used as the primary metric (imbalanced target).
Scaling: the processed dataset is already scaled by the Step 4 pipeline
(StandardScaler). The existing scaler is re-used so no re-scaling/refitting
is needed here; XGBoost/tree models are scale-invariant anyway.
"""
import json
import os
import pickle
import sys

# Ensure the project root is importable when run as a bare script.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from utils.constants import TARGET_COLUMN, RANDOM_STATE, TEST_SIZE
from utils.helpers import load_dataset
from utils.logger import logger

from preprocessing.utils import TARGET_MAPPING, LABEL_ENCODERS_PATH

from training.evaluate_model import CLASS_NAMES, evaluate_model
from training.hyperparameter_tuning import build_xgb_model, tune_xgboost
from training import save_model as sm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "processed_dataset.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------------
# Report writing helpers
# ---------------------------------------------------------------------------


def plot_confusion_matrix(cm, path, title="Confusion Matrix"):
    """Save a labelled confusion-matrix heatmap (counts + normalized)."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted",
        ylabel="Actual",
        title=title,
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot -> {path}")


def plot_feature_importance(model, feature_names, path, top_n=20):
    """Save the top-N XGBoost feature importance (gain) bar chart."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.arange(len(values))[::-1], values, align="center")
    ax.set_yticks(np.arange(len(values))[::-1])
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Importance (gain)")
    ax.set_title("XGBoost – Top Feature Importances")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved feature importance plot -> {path}")


def write_classification_report(report_text, path):
    """Persist the classification report as plain text."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"Saved classification report -> {path}")


def write_metrics_json(report, path):
    """Persist all model metrics as JSON."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved metrics report -> {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_training_pipeline(processed_path=PROCESSED_DATASET_PATH, tune_n_iter=20):
    """
    Execute the full Step 5 training pipeline and write models/reports.

    Returns a dict with baseline / initial XGBoost / tuned XGBoost / final
    model results plus the tuning summary.
    """
    print("=" * 80)
    print("STEP 5: ACCIDENT SEVERITY PREDICTION MODEL (XGBOOST)")
    print("=" * 80)

    # --- Data preparation ---------------------------------------------------
    df = load_dataset(processed_path)
    logger.info(f"Processed dataset shape: {df.shape}")

    # Target / feature separation
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Target leakage check: ensure no residual/encoded target duplicates remain.
    leakage_cols = [
        c for c in X.columns if c.lower() in ("severity", "severity_code", "target")
    ]
    if leakage_cols:
        logger.warning(f"Removing leakage columns: {leakage_cols}")
        X = X.drop(columns=leakage_cols)

    # Train/test split (80/20, stratified, fixed seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(
        f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}, "
        f"Features: {X.shape[1]}"
    )

    # Scaling: the processed dataset is already scaled by Step 4. Re-use the
    # existing fitted scaler so predictions stay compatible with saved data.
    # XGBoost/tree models are scale-invariant; LR consumes the pre-scaled data.
    scaler = None
    prefs_scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(prefs_scaler_path):
        with open(prefs_scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Reusing existing fitted scaler from preprocessing (Step 4).")

    # --- Baselines ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("BASELINE MODELS")
    print("=" * 80)

    baselines = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    baseline_results = {}
    trained_baselines = {}
    for name, model in baselines.items():
        logger.info(f"Training {name} ...")
        model.fit(X_train, y_train)
        trained_baselines[name] = model
        res = evaluate_model(model, X_test, y_test, label=name)
        baseline_results[name] = {
            "accuracy": res["metrics"]["accuracy"],
            "precision": res["metrics"]["precision_weighted"],
            "recall": res["metrics"]["recall_weighted"],
            "f1": res["metrics"]["f1_weighted"],
            "precision_macro": res["metrics"]["precision_macro"],
            "recall_macro": res["metrics"]["recall_macro"],
            "f1_macro": res["metrics"]["f1_macro"],
            "roc_auc": res["roc_auc"],
        }
        print(
            f"{name:24s} acc={res['metrics']['accuracy']:.4f} "
            f"prec={res['metrics']['precision_weighted']:.4f} "
            f"rec={res['metrics']['recall_weighted']:.4f} "
            f"f1={res['metrics']['f1_weighted']:.4f}"
        )

    # --- Initial XGBoost ----------------------------------------------------
    print("\n" + "=" * 80)
    print("INITIAL XGBOOST")
    print("=" * 80)

    xgb_initial = build_xgb_model()
    xgb_initial.fit(X_train, y_train)
    initial_res = evaluate_model(xgb_initial, X_test, y_test, label="XGBoost (initial)")
    initial_xgb = {
        "accuracy": initial_res["metrics"]["accuracy"],
        "precision": initial_res["metrics"]["precision_weighted"],
        "recall": initial_res["metrics"]["recall_weighted"],
        "f1": initial_res["metrics"]["f1_weighted"],
        "precision_macro": initial_res["metrics"]["precision_macro"],
        "recall_macro": initial_res["metrics"]["recall_macro"],
        "f1_macro": initial_res["metrics"]["f1_macro"],
        "roc_auc": initial_res["roc_auc"],
        "params": xgb_initial.get_params(),
    }
    print(
        f"XGBoost (initial)    acc={initial_res['metrics']['accuracy']:.4f} "
        f"prec={initial_res['metrics']['precision_weighted']:.4f} "
        f"rec={initial_res['metrics']['recall_weighted']:.4f} "
        f"f1={initial_res['metrics']['f1_weighted']:.4f}"
    )


    # --- Hyperparameter tuning ----------------------------------------------
    print("\n" + "=" * 80)
    print(f"HYPERPARAMETER TUNING (RandomizedSearchCV, n_iter={tune_n_iter})")
    print("=" * 80)
    xgb_tuned, tuning_summary = tune_xgboost(
        X_train, y_train, n_iter=tune_n_iter, random_state=RANDOM_STATE
    )
    tuned_res = evaluate_model(xgb_tuned, X_test, y_test, label="XGBoost (tuned)")
    tuned_xgb = {
        "accuracy": tuned_res["metrics"]["accuracy"],
        "precision": tuned_res["metrics"]["precision_weighted"],
        "recall": tuned_res["metrics"]["recall_weighted"],
        "f1": tuned_res["metrics"]["f1_weighted"],
        "precision_macro": tuned_res["metrics"]["precision_macro"],
        "recall_macro": tuned_res["metrics"]["recall_macro"],
        "f1_macro": tuned_res["metrics"]["f1_macro"],
        "roc_auc": tuned_res["roc_auc"],
        "params": xgb_tuned.get_params(),
    }
    print(
        f"XGBoost (tuned)      acc={tuned_res['metrics']['accuracy']:.4f} "
        f"prec={tuned_res['metrics']['precision_weighted']:.4f} "
        f"rec={tuned_res['metrics']['recall_weighted']:.4f} "
        f"f1={tuned_res['metrics']['f1_weighted']:.4f}"
    )

    # --- Model selection (primary: F1-weighted) ------------------------------
    print("\n" + "=" * 80)
    print("MODEL SELECTION")
    print("=" * 80)
    candidates = {"XGBoost (tuned)": tuned_xgb}
    for name in baselines:
        candidates[name] = baseline_results[name]
    candidates["XGBoost (initial)"] = initial_xgb

    best_name = max(candidates, key=lambda k: candidates[k]["f1"])
    best_metrics = candidates[best_name]

    if best_name == "XGBoost (tuned)":
        final_model = xgb_tuned
    elif best_name == "XGBoost (initial)":
        final_model = xgb_initial
    else:
        final_model = trained_baselines[best_name]

    final_eval = evaluate_model(final_model, X_test, y_test, label=best_name)
    final_result = {
        "selected_model": best_name,
        "metrics": best_metrics,
        "roc_auc": best_metrics.get("roc_auc"),
    }
    print(f"Selected model: {best_name} (F1-weighted = {best_metrics['f1']:.4f})")


    # --- Reports -------------------------------------------------------------
    final_cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(
        final_eval["confusion_matrix"],
        final_cm_path,
        title=f"Confusion Matrix – {best_name}",
    )

    # Feature importance from the tuned XGBoost model (best gradient-boosted fit)
    fi_model = xgb_tuned
    fi_path = os.path.join(REPORTS_DIR, "feature_importance.png")
    plot_feature_importance(fi_model, list(X.columns), fi_path)

    cr_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    write_classification_report(final_eval["classification_report"], cr_path)

    metrics_report = {
        "target": TARGET_COLUMN,
        "class_names": CLASS_NAMES,
        "metric_average": "weighted (primary) + macro",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "train_size": int(X_train.shape[0]),
        "test_size_actual": int(X_test.shape[0]),
        "baselines": baseline_results,
        "initial_xgboost": initial_xgb,
        "tuned_xgboost": tuned_xgb,
        "tuning_summary": tuning_summary,
        "final_model": final_result,
    }
    metrics_path = os.path.join(REPORTS_DIR, "model_metrics.json")
    write_metrics_json(metrics_report, metrics_path)

    # --- Persist models + artifacts -------------------------------------------
    sm.save_model(final_model)
    sm.save_feature_columns(X.columns)
    if scaler is not None:
        sm.save_scaler(scaler)

    # Preserve the existing label-encoding logic/artifacts from Step 4. If the
    # full encoder dict exists, re-save it unchanged; otherwise persist the
    # authoritative target mapping so encoders remain compatible.
    if os.path.exists(LABEL_ENCODERS_PATH):
        with open(LABEL_ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)
        sm.save_label_encoders(encoders)
    else:
        sm.save_label_encoders({"target_mapping": TARGET_MAPPING})

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"Saved final model ({best_name}) -> {sm.MODEL_PATH}")
    print(f"Saved feature columns -> {sm.FEATURE_COLUMNS_PATH}")
    print("=" * 80)

    return metrics_report


if __name__ == "__main__":
    report = run_training_pipeline()


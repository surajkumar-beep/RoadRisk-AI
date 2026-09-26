"""
Model evaluation helpers: full metric set, confusion matrix,
classification report and multiclass ROC-AUC.

The processed dataset is already scaled/encoded by the preprocessing
pipeline (Step 4), so models are evaluated on that representation.
"""
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from training.metrics import compute_classification_metrics

# Human-readable nominal labels for the encoded target (0/1/2).
CLASS_NAMES = ["Slight Injury", "Serious Injury", "Fatal injury"]


def evaluate_model(model, X_test, y_test, label="model"):
    """
    Evaluate a fitted model on a test set.

    Returns a dict containing:
      - metrics : accuracy / precision / recall / F1 (weighted + macro)
      - confusion_matrix
      - classification_report (text)
      - roc_auc   : multiclass one-vs-rest macro AUC (or None if invalid)
      - y_pred
    """
    y_pred = model.predict(X_test)

    metrics = compute_classification_metrics(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, zero_division=0, digits=4
    )

    # Multiclass ROC-AUC (one-vs-rest, macro average) when proba is available.
    roc_auc = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            roc_auc = float(
                roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
            )
    except Exception:
        roc_auc = None

    return {
        "label": label,
        "metrics": metrics,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_text,
        "y_pred": y_pred,
    }

"""
Reusable metric functions for evaluating the multiclass accident severity
prediction model.

The target (Accident_severity) has 3 classes and is imbalanced
(Slight: 10415, Serious: 1743, Fatal: 158), so both weighted and macro
averages are reported. Primary metric used for model selection: F1-weighted.
"""
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Averaging strategy for the imbalanced multiclass target (state for the record).
PRIMARY_AVERAGE = "weighted"
SECONDARY_AVERAGE = "macro"


def compute_classification_metrics(y_true, y_pred):
    """
    Compute accuracy, precision, recall and F1 for the multiclass target.

    Precision/Recall/F1 are provided for both 'weighted' and 'macro'
    averaging. 'weighted' (class-frequency weighted) is the primary choice
    because the severity classes are imbalanced.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    for average in (PRIMARY_AVERAGE, SECONDARY_AVERAGE):
        metrics[f"precision_{average}"] = float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        )
        metrics[f"recall_{average}"] = float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        )
        metrics[f"f1_{average}"] = float(
            f1_score(y_true, y_pred, average=average, zero_division=0)
        )
    return metrics

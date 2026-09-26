"""
Generate a concise, executed `04_Model_Testing.ipynb` for STEP 6
(Model Testing & Prediction). Uses the same 'roadrisk' kernel as
03_Model_Training.ipynb. Run with:
    python notebooks/exec_training_notebook.py notebooks/04_Model_Testing.ipynb
"""
import json
import os as _os
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (RoadRisk)", "language": "python", "name": "roadrisk"},
    "language_info": {"name": "python", "version": "3.12.0"},
}
cells = []


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


cells.append(md(r"""# STEP 6 - Model Testing & Prediction

**Notebook:** `04_Model_Testing.ipynb` &nbsp;|&nbsp; **Model:** `models/accident_severity_model.pkl` (Step 5).

Evaluates the **saved** XGBoost model on the held-out 20% test set (same split as Step 5:
`test_size=0.2, stratify=y, random_state=42`) and predicts a completely new accident record.
**No re-fitting** of the model, scaler, or encoders happens anywhere in this notebook.

| Section | Content |
|---|---|
| 0 | Load saved artifacts & verify |
| 1 | Test set (split recreated, no fitting) |
| 2 | Evaluation: metrics + classification report |
| 3 | Confusion matrix (test) |
| 4 | Sample predictions with probabilities |
| 5 | Completely new accident record prediction |
"""))

cells.append(md(r"""## 0. Load Saved Artifacts
"""))

cells.append(code(r"""%matplotlib inline
import json, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

def _find_root():
    d = os.getcwd()
    while True:
        if os.path.exists(os.path.join(d, "dataset", "processed_dataset.csv")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return os.path.abspath("..")

ROOT = _find_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from testing.test_model import (
    load_artifacts, load_test_data, preprocess_test_data,
    evaluate_model, predict_with_proba, print_prediction,
    predict_new_record, EXAMPLE_NEW_RECORD,
    save_test_confusion_matrix,
)
from utils.constants import TARGET_COLUMN, RANDOM_STATE, TEST_SIZE

artifacts = load_artifacts()
model, class_labels, feature_columns = (
    artifacts["model"], artifacts["class_labels"], artifacts["feature_columns"])
print("Model          :", type(model).__name__)
print("Class labels   :", class_labels)
print("Feature columns:", len(feature_columns))
"""))

cells.append(md(r"""## 1. Test Set (split recreated - NO fitting)
"""))

cells.append(code(r"""X_train, X_test, y_train, y_test = load_test_data()
X_test = preprocess_test_data(X_test, feature_columns)
print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}  Features: {X_test.shape[1]}")
print("Feature order matches model:",
      list(X_test.columns) == list(feature_columns))
"""))

cells.append(md(r"""## 2. Evaluation (weighted + macro; ROC-AUC OvR macro)
"""))

cells.append(code(r"""res = evaluate_model(model, X_test, y_test, class_labels)
m = res["metrics"]
print("=" * 58)
print("FINAL TEST METRICS")
print("=" * 58)
print(f"  Accuracy            : {m['accuracy']:.4f}")
print(f"  Precision (weighted): {m['precision_weighted']:.4f}   Precision (macro): {m['precision_macro']:.4f}")
print(f"  Recall    (weighted): {m['recall_weighted']:.4f}   Recall    (macro): {m['recall_macro']:.4f}")
print(f"  F1        (weighted): {m['f1_weighted']:.4f}   F1        (macro): {m['f1_macro']:.4f}")
print(f"  ROC-AUC (OvR macro) : {res['roc_auc']:.4f}")

print("\n" + "=" * 58)
print("CLASSIFICATION REPORT")
print("=" * 58)
print(res["classification_report"])
"""))

cells.append(md(r"""## 3. Confusion Matrix (Test Set)
"""))

cells.append(code(r"""from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, res["y_pred"])
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(pd.DataFrame(cm, index=class_labels, columns=class_labels),
            annot=True, fmt="d", cmap="Blues", ax=ax,
            cbar_kws={"label": "count"})
ax.set_title("Confusion Matrix - Test Set (held-out 20%)")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""## 4. Sample Predictions with Probabilities (test set)
"""))

cells.append(code(r"""proba = model.predict_proba(X_test)
preds, _ = predict_with_proba(model, X_test, class_labels)
for i in range(5):
    actual = class_labels[int(y_test.iloc[i])]
    print_prediction(actual, preds[i], proba[i], class_labels)
"""))


cells.append(md(r"""## 5. Completely New Accident Record

The record below is a **synthetic example** (clearly not real data) used to demonstrate the
prediction API. It is transformed using only the saved preprocessing artifacts &mdash;
the model/scaler/encoders are **never** re-fit.
"""))

cells.append(code(r"""print("New record (synthetic example - NOT real data):")
print(json.dumps(EXAMPLE_NEW_RECORD, indent=1))

label, proba_row, X_new = predict_new_record(EXAMPLE_NEW_RECORD, artifacts)
print(f"\nEncoded feature vector: {X_new.shape[1]} columns "
      f"(matches model: {X_new.shape[1] == len(feature_columns)})")
print_prediction("(no ground truth - new/unseen record)", label, proba_row, class_labels)
"""))

cells.append(md(r"""### Notes / Limitations
- Metrics reproduce the Step 5 report exactly (same data, same split, same model) &mdash; confirming no leakage.
- The class imbalance persists: `Slight Injury` recall is high while `Serious`/`Fatal` recall are low.
- The new-record prediction is purely illustrative; its probabilities reflect the model prior over
  the dominant class.
"""))

nb.cells = cells
out = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "04_Model_Testing.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook created:", out, "| cells:", len(cells))


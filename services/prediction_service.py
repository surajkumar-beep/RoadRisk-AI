"""
Prediction service (STEP 7 logic, preserved verbatim).

Hosts the model artifacts, form options, record validation and prediction
chain previously living inside app.py:

    raw form values
      -> Step 4 feature engineering (Hour, experience / vehicle-age groups)
      -> saved ordinal mappings
      -> saved OneHotEncoder (handle_unknown='ignore')
      -> feature alignment via models/feature_columns.pkl
      -> saved scaler (fit on train only, reused as-is)
      -> model.predict() + model.predict_proba()

Nothing is re-trained or re-fitted anywhere in the service layer; every rule
comes from the SAVED Step 5 artifacts (models/*.pkl) exactly as STEP 6 used
them.  If an artifact is missing or corrupt the app still boots and every
page renders a clear friendly error instead of a traceback.
"""
import os
import re

import joblib
import pandas as pd

from testing.test_model import (  # exact STEP 6 new-record preprocessing, reused
    predict_with_proba,
    preprocess_new_record,
)
from training.save_model import (  # artifact path constants only
    FEATURE_COLUMNS_PATH,
    LABEL_ENCODERS_PATH,
    MODEL_PATH,
    SCALER_PATH,
)
from preprocessing.encoding import encode_ordinal_features
from preprocessing.feature_engineering import group_driving_experience, group_vehicle_age
from preprocessing.utils import NUMERICAL_FEATURES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_COLUMN = "Accident_severity"  # same target as Steps 4-6 (utils/constants.py)

# User-facing numeric sanity bounds (the saved preprocessing is never changed).
VEHICLES_MIN, VEHICLES_MAX = 1, 100      # vehicles involved must be > 0
CASUALTIES_MIN, CASUALTIES_MAX = 0, 100  # casualties may be >= 0

TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")

# Fields that Step 4 groups BEFORE ordinal encoding: their valid form options
# are the RAW dataset values (read from the project's own raw dataset file).
GROUPED_BEFORE_ORDINAL = {"Service_year_of_vehicle"}
RAW_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "RTA Dataset.csv")

# ---------------------------------------------------------------------------
# Form layout: grouped section cards (icon + title + short description) for
# the redesigned prediction / XAI forms.  Defect_of_vehicle is intentionally
# absent: Step 4 drops it before training.  Option VALUES are derived at
# startup from the saved artifacts - not hardcoded.
# ---------------------------------------------------------------------------
FORM_GROUPS = [
    {
        "title": "Accident Information",
        "icon": "⏱",
        "description": "When the accident happened and how the collision unfolded.",
        "fields": [
            "Time", "Day_of_week", "Number_of_vehicles_involved",
            "Number_of_casualties", "Type_of_collision", "Vehicle_movement",
            "Cause_of_accident",
        ],
    },
    {
        "title": "Location",
        "icon": "⌖",
        "description": "Where the accident happened on the road network.",
        "fields": [
            "Area_accident_occured", "Lanes_or_Medians", "Road_allignment",
            "Types_of_Junction",
        ],
    },
    {
        "title": "Road Conditions",
        "icon": "▤",
        "description": "Road surface type and its condition at the scene.",
        "fields": [
            "Road_surface_type", "Road_surface_conditions",
        ],
    },
    {
        "title": "Weather & Light",
        "icon": "☀",
        "description": "Weather and visibility at the time of the accident.",
        "fields": [
            "Light_conditions", "Weather_conditions",
        ],
    },
    {
        "title": "Vehicle / Driver Information",
        "icon": "⬡",
        "description": "Driver profile and the vehicle involved in the accident.",
        "fields": [
            "Age_band_of_driver", "Sex_of_driver", "Educational_level",
            "Vehicle_driver_relation", "Driving_experience", "Type_of_vehicle",
            "Owner_of_vehicle", "Service_year_of_vehicle",
        ],
    },
    {
        "title": "Additional Conditions",
        "icon": "＋",
        "description": "Casualty details and pedestrian movement context.",
        "fields": [
            "Casualty_class", "Sex_of_casualty", "Age_band_of_casualty",
            "Casualty_severity", "Work_of_casuality", "Fitness_of_casuality",
            "Pedestrian_movement",
        ],
    },
]

FORM_FIELDS = [field for group in FORM_GROUPS for field in group["fields"]]
_NUMERIC_FIELDS = {"Number_of_vehicles_involved", "Number_of_casualties"}

# Display tone for each predicted severity class (matches the Plotly palette).
TONE_BY_LABEL = {
    "Slight Injury": "green",
    "Serious Injury": "amber",
    "Fatal injury": "red",
}

# Populated at startup via initialize(); kept at module scope for route use.
_ARTIFACTS = None
_OPTIONS = None
_LOAD_ERROR = None
# ---------------------------------------------------------------------------
# Dropdown options + startup guards (all derived from the SAVED artifacts)
# ---------------------------------------------------------------------------


def _field_label(field):
    """Human-friendly label for a raw dataset field name."""
    return field.replace("_", " ")


def _field_widget(field):
    """Widget type per field: 'time' | 'number' | 'select'."""
    if field == "Time":
        return "time"
    if field in _NUMERIC_FIELDS:
        return "number"
    return "select"


def _ordinal_code(column, value):
    """
    Ordinal code the ACTUAL Step 4 chain (grouping -> saved ordinal mapping)
    assigns to one raw value. Returns None when the value would end up as a
    NaN fill (silently treated as Unknown) so it is never offered in the form.
    """
    df = pd.DataFrame([{column: value}])
    if column == "Driving_experience":
        df = group_driving_experience(df, verbose=False)
    if column == "Service_year_of_vehicle":
        df = group_vehicle_age(df, verbose=False)
    df = encode_ordinal_features(df, verbose=False)
    code = df[column].iloc[0]
    return None if pd.isna(code) else int(code)


def _raw_dataset_uniques(column):
    """Unique raw values for a column, taken from the project's raw dataset."""
    if not os.path.exists(RAW_DATASET_PATH):
        return []
    try:
        series = pd.read_csv(RAW_DATASET_PATH, usecols=[column])[column]
    except Exception:  # noqa: BLE001 - fall back to the saved mapping keys
        return []
    return sorted(series.dropna().astype(str).unique().tolist())


def build_form_options(artifacts):
    """
    Derive every dropdown's allowed values from the SAVED artifacts:
      - nominal fields: categories known to the saved OneHotEncoder
      - ordinal fields: values that map to a defined ordinal code through the
        real Step 4 chain (grouping -> saved ordinal mappings), with the
        neutral code (-1, e.g. 'Unknown'/'na') listed last.
    """
    enc = artifacts["encoders"]
    options = {}

    for column, categories in zip(enc["nominal_features"], enc["one_hot_encoder"].categories_):
        options[column] = [str(category) for category in categories]

    for column, mapping in enc["ordinal_mappings"].items():
        if column in GROUPED_BEFORE_ORDINAL:
            candidates = _raw_dataset_uniques(column)
            dedupe = False  # distinct raw inputs - keep all even if they group together
            if not candidates:
                candidates = [str(key) for key in mapping.keys()]
                dedupe = True
        else:
            candidates = [str(key) for key in mapping.keys()]
            dedupe = True
        candidates = sorted(candidates, key=lambda value: (value.isdigit(), value))

        coded, seen = [], set()
        for value in candidates:
            code = _ordinal_code(column, value)
            if code is None or (dedupe and code in seen):
                continue
            seen.add(code)
            coded.append((value, code))
        coded.sort(key=lambda item: (item[1] < 0, item[1], item[0]))
        options[column] = [value for value, _ in coded]

    return options
def _validate_form_coverage(artifacts, options):
    """Guarantee the form covers exactly the model's raw input schema."""
    enc = artifacts["encoders"]
    numeric_inputs = [c for c in NUMERICAL_FEATURES if c != "Hour"]  # Hour derives from Time
    required = (
        set(enc["ordinal_features"]) | set(enc["nominal_features"])
        | set(numeric_inputs) | {"Time"}
    )
    form_fields = set(FORM_FIELDS)
    missing = sorted(required - form_fields)
    unexpected = sorted(form_fields - required)
    if missing or unexpected:
        raise ValueError(
            "Form does not match the model's raw input schema. "
            f"Missing: {missing}; unexpected: {unexpected}"
        )
    without_options = sorted(
        f for f in form_fields
        if f not in options and f not in set(numeric_inputs) | {"Time"}
    )
    if without_options:
        raise ValueError(f"No allowed values could be derived for: {without_options}")


def load_artifacts_joblib():
    """Load the four saved Step 5 artifacts with joblib (no fitting anywhere)."""
    paths = {
        "model": MODEL_PATH,
        "feature_columns": FEATURE_COLUMNS_PATH,
        "scaler": SCALER_PATH,
        "encoders": LABEL_ENCODERS_PATH,
    }
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifact(s): {', '.join(missing)} "
            f"(looked in {os.path.dirname(MODEL_PATH)})"
        )

    artifacts = {name: joblib.load(path) for name, path in paths.items()}

    # Class labels strictly from the saved target mapping (never hardcoded).
    target_mapping = artifacts["encoders"]["target_mapping"]
    code_to_label = {int(code): label for label, code in target_mapping.items()}
    artifacts["class_labels"] = [code_to_label[code] for code in sorted(code_to_label)]

    # Guard against a model / feature_columns.pkl mismatch before serving.
    expected = len(artifacts["feature_columns"])
    model = artifacts["model"]
    n_features = getattr(model, "n_features_in_", expected)
    if n_features != expected:
        raise ValueError(
            f"Model expects {n_features} features but models/feature_columns.pkl has {expected}."
        )
    return artifacts


def initialize():
    """Load artifacts once at app startup; degrade gracefully on failure."""
    global _ARTIFACTS, _OPTIONS, _LOAD_ERROR
    try:
        artifacts = load_artifacts_joblib()
        options = build_form_options(artifacts)
        _validate_form_coverage(artifacts, options)
        _ARTIFACTS, _OPTIONS, _LOAD_ERROR = artifacts, options, None
    except Exception as exc:  # noqa: BLE001 - keep the app alive with friendly errors
        _ARTIFACTS, _OPTIONS, _LOAD_ERROR = None, None, str(exc)
    return _ARTIFACTS


def get_artifacts():
    if _ARTIFACTS is None:
        initialize()
    return _ARTIFACTS


def get_options():
    if _OPTIONS is None:
        initialize()
    return _OPTIONS


def get_load_error():
    return _LOAD_ERROR


def is_ready():
    return _ARTIFACTS is not None and _OPTIONS is not None
# ---------------------------------------------------------------------------
# Validation -> raw record (never crashes on bad/unknown input)
# ---------------------------------------------------------------------------


def validate_and_build_record(form_data, options=None):
    """Validate the submitted form and build a raw record in the dataset schema."""
    if options is None:
        options = get_options()
    errors = []
    record = {}

    # 1. Time - required, valid 24h clock (normalised to HH:MM:SS for Step 4).
    raw_time = str(form_data.get("Time") or "").strip()
    if not raw_time:
        errors.append("Time is required.")
    else:
        match = TIME_RE.match(raw_time)
        if not match:
            errors.append("Time must be a valid 24-hour time, e.g. 14:30 or 14:30:00.")
        else:
            hour, minute, second = (
                int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)
            )
            if hour > 23 or minute > 59 or second > 59:
                errors.append("Time is out of range: hours 00-23, minutes/seconds 00-59.")
            else:
                record["Time"] = f"{hour:02d}:{minute:02d}:{second:02d}"

    # 2. Numeric fields - required, whole numbers within sane ranges.
    numeric_specs = [
        ("Number_of_vehicles_involved", "Number of vehicles involved",
         VEHICLES_MIN, VEHICLES_MAX, "must be a whole number greater than 0"),
        ("Number_of_casualties", "Number of casualties",
         CASUALTIES_MIN, CASUALTIES_MAX, "must be a whole number of 0 or more"),
    ]
    for field, label, low, high, rule in numeric_specs:
        raw_value = str(form_data.get(field) or "").strip()
        if not raw_value:
            errors.append(f"{label} is required.")
            continue
        try:
            value = int(raw_value)
        except ValueError:
            errors.append(f"{label} {rule}.")
            continue
        if not low <= value <= high:
            errors.append(f"{label} {rule} (allowed range {low}-{high}).")
            continue
        record[field] = value

    # 3. Categorical fields - must be categories known to the saved encoders.
    for field, allowed in options.items():
        value = str(form_data.get(field) or "").strip()
        if not value:
            errors.append(f"{_field_label(field)} is required.")
            continue
        if value not in allowed:
            errors.append(
                f'Unknown category "{value}" for {_field_label(field)}. '
                "Please choose one of the listed options."
            )
            continue
        record[field] = value

    return record, errors


# ---------------------------------------------------------------------------
# Prediction (identical chain to STEP 6 / old app.py /predict)
# ---------------------------------------------------------------------------


def make_prediction(record):
    """Run the exact STEP 6 new-record preprocessing + model scoring.

    Returns a dict: {prediction, probabilities: [{label, pct}], confidence_pct,
    record_items: [(field, value)], model}.  Re-raises a predictable
    :class:`RuntimeError` whose message is safe to show users on failure.
    """
    artifacts = get_artifacts()
    if artifacts is None:
        raise RuntimeError("Model artifacts are unavailable. Please try again later.")

    try:
        X_new = preprocess_new_record(record, artifacts)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "The model's feature requirements could not be matched. Please try again."
        ) from exc

    if X_new.shape[1] != len(artifacts["feature_columns"]):
        raise RuntimeError("The model's feature requirements could not be matched. Please try again.")

    try:
        pred_labels, proba = predict_with_proba(
            artifacts["model"], X_new, artifacts["class_labels"]
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("The model could not make a prediction for this record. Please try again.") from exc

    prediction = pred_labels[0]
    probabilities = [
        {"label": label, "pct": round(float(p) * 100.0, 2)}
        for label, p in zip(artifacts["class_labels"], proba[0])
    ]
    confidence_pct = max(item["pct"] for item in probabilities)
    record_items = [(field, record[field]) for field in FORM_FIELDS if field in record]

    return {
        "prediction": prediction,
        "prediction_tone": TONE_BY_LABEL.get(prediction, "green"),
        "probabilities": probabilities,
        "confidence_pct": confidence_pct,
        "record_items": record_items,
        "model": type(artifacts["model"]).__name__,
    }
def form_view():
    """Template-ready form structure: groups -> fields -> {widget, options}.

    Used by both the prediction page and the XAI explain page so the two forms
    can never drift apart.
    """
    options = get_options()
    groups = []
    for group in FORM_GROUPS:
        group_fields = []
        for field in group["fields"]:
            widget = _field_widget(field)
            entry = {"name": field, "label": _field_label(field), "widget": widget}
            if widget == "select":
                entry["options"] = options.get(field, [])
            elif widget == "number":
                entry["min"] = VEHICLES_MIN if field == "Number_of_vehicles_involved" else CASUALTIES_MIN
                entry["max"] = VEHICLES_MAX if field == "Number_of_vehicles_involved" else CASUALTIES_MAX
            group_fields.append(entry)
        groups.append({
            "title": group["title"],
            "icon": group["icon"],
            "description": group["description"],
            "fields": group_fields,
        })
    return groups


# ---------------------------------------------------------------------------
# Random record sampler (used by the XAI "explain a sample" button)
# ---------------------------------------------------------------------------


def sample_record(seed=None, time="14:30:00", vehicles="2", casualties="1"):
    """Build a structurally valid synthetic record from the real option lists.

    Every categorical value is drawn from the categories the model knows, and
    ``time`` is ``HH:MM:SS`` (exactly what the Step 4 pipeline parses), so the
    generated record is always accepted by the prediction/explain pipeline.
    """
    import random

    options = get_options()
    rng = random.Random(seed)
    record = {"Time": time,
              "Number_of_vehicles_involved": vehicles,
              "Number_of_casualties": casualties}
    for field, allowed in options.items():
        record[field] = rng.choice(allowed)
    return record
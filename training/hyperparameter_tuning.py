"""
Hyperparameter tuning for the XGBoost severity model.

Uses RandomizedSearchCV with:
  - stratified K-fold cross-validation
  - F1-weighted scoring (matches the primary selection metric)
  - a fixed random_state for reproducibility
  - n_iter kept moderate to bound runtime

The best estimator is refit on the full training split.
"""
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier


def get_xgb_param_grid():
    """Tunable XGBoost hyperparameter grid."""
    return {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }


def build_xgb_model(**overrides):
    """Create an XGBClassifier with sensible defaults + optional overrides."""
    defaults = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    defaults.update(overrides)
    return XGBClassifier(**defaults)


def tune_xgboost(X_train, y_train, n_iter=20, cv_folds=5, random_state=42, scoring="f1_weighted"):
    """
    Run RandomizedSearchCV over the XGBoost grid.

    Returns (best_estimator, cv_results_dict) where cv_results_dict keeps the
    top configurations for the report.
    """
    model = build_xgb_model()
    param_grid = get_xgb_param_grid()

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
        refit=True,
    )

    search.fit(X_train, y_train)

    results = search.cv_results_
    top_candidates = sorted(
        [
            {
                "rank": int(results["rank_test_score"][i]),
                "params": results["params"][i],
                "mean_test_score": float(results["mean_test_score"][i]),
                "std_test_score": float(results["std_test_score"][i]),
            }
            for i in range(len(results["rank_test_score"]))
        ],
        key=lambda c: c["rank"],
    )[:5]

    summary = {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "scoring": scoring,
        "n_iter": n_iter,
        "cv_folds": cv_folds,
        "top_candidates": top_candidates,
    }
    return search.best_estimator_, summary

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def validate_binary_target(
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """Validate whether a target can be used for binary classification."""

    if target_column not in df.columns:
        return {
            "valid": False,
            "reason": "Target column does not exist in the dataset.",
        }

    target = df[target_column].dropna()
    unique_values = target.unique()

    if len(unique_values) != 2:
        return {
            "valid": False,
            "reason": (
                "This first ML pipeline currently supports binary "
                "classification targets only."
            ),
            "unique_values": int(len(unique_values)),
        }

    counts = target.value_counts()

    if counts.min() < 5:
        return {
            "valid": False,
            "reason": (
                "The minority class contains fewer than 5 rows, "
                "which is too small for a reliable train/test split."
            ),
        }

    return {
        "valid": True,
        "unique_values": 2,
        "class_counts": counts.to_dict(),
    }


def _prepare_training_data(
    df: pd.DataFrame,
    target_column: str,
    max_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Clean target rows and optionally take a stratified training sample."""

    working_df = df.dropna(subset=[target_column]).copy()

    X = working_df.drop(columns=[target_column])
    y = working_df[target_column]

    original_rows = len(working_df)
    sampled = False

    if max_rows is not None and len(working_df) > max_rows:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=max_rows,
            stratify=y,
            random_state=random_state,
        )
        sampled = True

    metadata = {
        "original_rows": original_rows,
        "rows_used": len(X),
        "sampled_for_training": sampled,
        "features": X.shape[1],
    }

    return X, y, metadata


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical features."""

    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    max_categories=30,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def _evaluate_binary_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Calculate classification metrics for a fitted binary model."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    classes = model.classes_

    negative_class = classes[0]
    positive_class = classes[1]

    positive_index = list(classes).index(positive_class)
    positive_probabilities = probabilities[:, positive_index]

    binary_truth = (y_test == positive_class).astype(int)
    binary_predictions = (predictions == positive_class).astype(int)

    cm = confusion_matrix(
        binary_truth,
        binary_predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    pr_precision, pr_recall, _ = precision_recall_curve(
        binary_truth,
        positive_probabilities,
    )

    roc_fpr, roc_tpr, _ = roc_curve(
        binary_truth,
        positive_probabilities,
    )

    return {
        "accuracy": float(
            accuracy_score(binary_truth, binary_predictions)
        ),
        "precision": float(
            precision_score(
                binary_truth,
                binary_predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                binary_truth,
                binary_predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                binary_truth,
                binary_predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                binary_truth,
                positive_probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                binary_truth,
                positive_probabilities,
            )
        ),
        "pr_curve": {
            "recall": pr_recall.tolist(),
            "precision": pr_precision.tolist(),
        },
        "roc_curve": {
            "fpr": roc_fpr.tolist(),
            "tpr": roc_tpr.tolist(),
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "negative_class": str(negative_class),
        "positive_class": str(positive_class),
    }


def train_binary_models(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    max_rows: int | None = 120_000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train Logistic Regression and Random Forest models.

    A stratified sample is used for very large datasets so the
    Streamlit portfolio remains responsive while preserving class ratios.
    """

    validation = validate_binary_target(df, target_column)

    if not validation["valid"]:
        return {
            "status": "blocked",
            "validation": validation,
        }

    X, y, metadata = _prepare_training_data(
        df=df,
        target_column=target_column,
        max_rows=max_rows,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    preprocessor = _build_preprocessor(X_train)

    logistic_model = Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )

    random_forest_model = Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=12,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
    }

    results: dict[str, Any] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        results[model_name] = _evaluate_binary_model(
            model,
            X_test,
            y_test,
        )

    metadata.update(
        {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_size": test_size,
            "random_state": random_state,
        }
    )

    return {
        "status": "completed",
        "target_column": target_column,
        "validation": validation,
        "metadata": metadata,
        "models": results,
    }


def model_comparison_table(
    result: dict[str, Any],
) -> pd.DataFrame:
    """Convert model results into a recruiter-friendly comparison table."""

    if result.get("status") != "completed":
        return pd.DataFrame()

    rows = []

    for model_name, metrics in result["models"].items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "ROC-AUC": metrics["roc_auc"],
                "PR-AUC": metrics["pr_auc"],
            }
        )

    return pd.DataFrame(rows)

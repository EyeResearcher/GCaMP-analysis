"""Train a logistic regression spike classifier from precomputed features."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("model_runs/GCaMP8s_Olympus_Glass/spike_filtering/spike_features.csv"),
        help="Path to spike_features.csv (must contain spike_key column).",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("model_runs/GCaMP8s_Olympus_Glass/spike_filtering/spike_annotations.csv"),
        help="Path to spike_annotations.csv with spike_key and label columns.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("spike_filtering/models/spike_logreg.joblib"),
        help="Where to save the trained logistic regression model.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Optional path to save a CSV summary of feature importances.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out fraction for the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum iterations for the logistic regression solver.",
    )
    return parser.parse_args()


def load_datasets(features_path: Path, annotations_path: Path) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    features = pd.read_csv(features_path)
    annotations = pd.read_csv(annotations_path)

    if "spike_key" not in features.columns:
        raise ValueError("features file must contain a 'spike_key' column")
    if "label" not in annotations.columns:
        raise ValueError("annotations file must contain a 'label' column")

    merged = annotations.merge(features, on="spike_key", how="inner")
    merged = merged.dropna(axis=0)

    if merged.empty:
        raise ValueError("Merged dataset is empty after alignment/cleaning.")

    duplicate_keys = merged.duplicated(subset=["spike_key"], keep=False)
    if duplicate_keys.any():
        merged = merged[~duplicate_keys]

    return merged


def train_logistic_regression(df: pd.DataFrame, max_iter: int, test_size: float, random_state: int):
    feature_columns = [col for col in df.columns if col not in {"spike_key", "label"}]
    X = df[feature_columns].astype(float).to_numpy()
    y = df["label"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=max_iter,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, digits=3),
    }

    clf = pipeline.named_steps["clf"]
    coefficients = pd.Series(clf.coef_[0], index=feature_columns)
    importances = coefficients.abs().sort_values(ascending=False)

    pipeline.feature_columns_ = feature_columns
    pipeline.top_features_ = importances.index.tolist()

    return pipeline, metrics, importances


def main() -> None:
    args = parse_args()
    df = load_datasets(args.features, args.annotations)
    model, metrics, importances = train_logistic_regression(
        df,
        max_iter=args.max_iter,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print("Confusion Matrix:\n", metrics["confusion_matrix"])
    print("Classification Report:\n", metrics["classification_report"])

    print("Top 10 features by absolute coefficient:")
    for name, value in importances.head(10).items():
        print(f"  {name:30s} {value:.4f}")

    model_path = args.model_output
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    print(f"\nSaved logistic regression model to {model_path}")

    if args.report_output:
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        importances.to_frame(name="abs_coef").to_csv(report_path)
        print(f"Feature importances written to {report_path}")


if __name__ == "__main__":
    main()

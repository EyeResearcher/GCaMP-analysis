"""Shared inference utilities for ROI and spike classifiers."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from classifier_pipeline.datasets import apply_transform


def get_model_feature_names(model: Any) -> list[str]:
    """Get ordered feature names from a trained sklearn model.

    Parameters
    ----------
    model : Any
        Trained sklearn model with ``feature_names_in_`` attribute.

    Returns
    -------
    list[str]
        Ordered feature names the model was trained on.

    Raises
    ------
    ValueError
        If model doesn't have ``feature_names_in_`` attribute.
    """
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise ValueError(
            "Model was not trained with feature names. Retrain on a DataFrame."
        )
    return list(names)


def prepare_features(
    feats_df: pd.DataFrame,
    model: Any,
    transform: Optional[str] = None,
) -> pd.DataFrame:
    """Reorder columns to match training order and apply optional transform.

    Parameters
    ----------
    feats_df : pd.DataFrame
        Raw extracted features.
    model : Any
        Trained sklearn model with ``feature_names_in_``.
    transform : str, optional
        Transform name to apply (e.g. ``"log1p"``), by default None.

    Returns
    -------
    pd.DataFrame
        Features ready for ``model.predict()``.

    Raises
    ------
    ValueError
        If *feats_df* is missing required columns.
    """
    expected = get_model_feature_names(model)

    missing = set(expected) - set(feats_df.columns)
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = feats_df[expected].copy()

    if transform:
        X = apply_transform(X, transform)

    return X

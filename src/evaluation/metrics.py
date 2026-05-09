"""
Metrics: Common evaluation metrics for ML models.

Provides metric computation functions for regression and classification tasks.
"""

import numpy as np
from typing import Dict


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary with RMSE, MAE, R2
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": r2,
    }


def _check_binary_classes(y_true: np.ndarray) -> bool:
    """Return True when both classes (0 and 1) are present in y_true."""
    unique = np.unique(y_true)
    return len(unique) == 2


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute classification metrics.

    Handles single-class inputs gracefully — AUC and confusion-matrix
    fields default to 0.0 when only one class is present.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary with AUC, accuracy, precision, recall, F1
    """
    try:
        from sklearn.metrics import (
            roc_auc_score, accuracy_score, precision_score,
            recall_score, f1_score, confusion_matrix
        )
    except ImportError:
        return {"error": "sklearn not available"}

    binary_pred = (y_pred > threshold).astype(int)
    both_classes = _check_binary_classes(y_true)

    auc = roc_auc_score(y_true, y_pred) if both_classes else 0.0
    accuracy = accuracy_score(y_true, binary_pred)
    precision = precision_score(y_true, binary_pred, zero_division=0)
    recall = recall_score(y_true, binary_pred, zero_division=0)
    f1 = f1_score(y_true, binary_pred, zero_division=0)

    cm = confusion_matrix(y_true, binary_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = (int(cm[i]) if i < len(cm) else 0 for i in range(4))

    return {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }
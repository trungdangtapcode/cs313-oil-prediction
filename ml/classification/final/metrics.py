"""
Unified metrics for all classification steps.

Every step in ml/classification/final/ should use these functions
so that CSV outputs share identical column names and semantics.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(name, y_true, y_pred, y_score=None):
    """Compute the standard metric dict used across all steps.

    Parameters
    ----------
    name : str
        Model identifier (used as the ``Model`` column value).
    y_true : array-like
        Ground-truth binary labels (0/1).
    y_pred : array-like
        Predicted binary labels (0/1).
    y_score : array-like or None
        Predicted probabilities for the positive class (or decision
        function values).  ``None`` → AUC will be ``NaN``.

    Returns
    -------
    dict with keys:
        Model, Accuracy, F1_binary, F1_macro, AUC,
        Precision_UP, Recall_UP, TP, FP, TN, FN
    """
    acc = accuracy_score(y_true, y_pred)
    f1_bin = f1_score(y_true, y_pred, average='binary', zero_division=0)
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score) if y_score is not None else np.nan
    except (ValueError, TypeError):
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec_up = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_up = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        'Model': name,
        'Accuracy': acc,
        'F1_binary': f1_bin,
        'F1_macro': f1_mac,
        'AUC': auc,
        'Precision_UP': prec_up,
        'Recall_UP': rec_up,
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
    }


# Column order used when printing / saving leaderboards
METRIC_COLS = [
    'Model', 'Accuracy', 'F1_binary', 'F1_macro', 'AUC',
    'Precision_UP', 'Recall_UP', 'TP', 'FP', 'TN', 'FN',
]

# Sort priority: F1_macro → Accuracy → AUC (all descending)
SORT_COLS = ['F1_macro', 'Accuracy', 'AUC']


def get_scores(model, X):
    """Get probability or decision-function scores from a fitted model."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        return model.decision_function(X)
    return None

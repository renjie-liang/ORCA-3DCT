"""
Classification metrics for disease detection

Computes AUROC, F1, Precision, Recall for multi-label disease classification

Fail-fast principles (with exceptions for valid edge cases):
- No try-except for unexpected errors
- Direct sklearn calls
- Return np.nan for single-class edge cases (valid scenario)
"""
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix
)


# Disease names from CT-RATE dataset
DISEASE_NAMES = [
    'Medical material',
    'Arterial wall calcification',
    'Cardiomegaly',
    'Pericardial effusion',
    'Coronary artery wall calcification',
    'Hiatal hernia',
    'Lymphadenopathy',
    'Emphysema',
    'Atelectasis',
    'Lung nodule',
    'Lung opacity',
    'Pulmonary fibrotic sequela',
    'Pleural effusion',
    'Mosaic attenuation pattern',
    'Peribronchial thickening',
    'Consolidation',
    'Bronchiectasis',
    'Interlobular septal thickening'
]


# ============================================================================
# Single-Class Metrics
# ============================================================================

def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute AUROC (Area Under ROC Curve)

    Args:
        y_true: True binary labels (N,) with values 0 or 1
        y_score: Predicted scores (N,) in range [0, 1]

    Returns:
        AUROC value, or np.nan if only one class present
    """
    # Check if only one class (valid edge case)
    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(roc_auc_score(y_true, y_score))


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute AUPRC (Area Under Precision-Recall Curve)

    Also known as Average Precision (AP)

    Args:
        y_true: True binary labels (N,)
        y_score: Predicted scores (N,)

    Returns:
        AUPRC value, or np.nan if only one class present
    """
    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(average_precision_score(y_true, y_score))


def compute_f1_optimal(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Compute optimal F1 score by finding best threshold

    Args:
        y_true: True binary labels (N,)
        y_score: Predicted scores (N,)

    Returns:
        (optimal_f1, optimal_threshold)
    """
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # Compute F1 for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Find maximum F1
    optimal_idx = np.argmax(f1_scores)
    optimal_f1 = f1_scores[optimal_idx]

    # Get corresponding threshold
    if optimal_idx < len(thresholds):
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 1.0

    return float(optimal_f1), float(optimal_threshold)


def compute_f1_fixed(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute F1 score with fixed threshold (already binarized)

    Args:
        y_true: True binary labels (N,)
        y_pred: Predicted binary labels (N,)

    Returns:
        F1 score
    """
    if len(np.unique(y_true)) < 2:
        return np.nan

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Compute F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return float(f1)


def compute_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics from confusion matrix

    Args:
        y_true: True binary labels (N,)
        y_pred: Predicted binary labels (N,)

    Returns:
        {
            'precision': float,
            'recall': float,      # sensitivity
            'specificity': float,
            'f1': float,
            'accuracy': float
        }
    """
    if len(np.unique(y_true)) < 2:
        return {
            'precision': np.nan,
            'recall': np.nan,
            'specificity': np.nan,
            'f1': np.nan,
            'accuracy': np.nan
        }

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'accuracy': float(accuracy)
    }


# ============================================================================
# Multi-Class Aggregation
# ============================================================================

def aggregate_macro(
    per_class_metrics: Dict[str, Dict[str, float]],
    class_names: list
) -> Dict[str, float]:
    """
    Compute macro average (equal weight for each class)

    Args:
        per_class_metrics: {class_name: {metric_name: value}}
        class_names: List of class names

    Returns:
        {'macro_auroc': float, 'macro_f1': float, ...}
    """
    if not per_class_metrics or not class_names:
        return {}

    # Get metric names from first class
    first_class = class_names[0]
    metric_names = list(per_class_metrics[first_class].keys())

    aggregated = {}

    # Macro average for each metric
    for metric_name in metric_names:
        values = []

        for class_name in class_names:
            if class_name in per_class_metrics:
                value = per_class_metrics[class_name][metric_name]
                if not np.isnan(value):
                    values.append(value)

        if len(values) > 0:
            aggregated[f'macro_{metric_name}'] = float(np.mean(values))
        else:
            aggregated[f'macro_{metric_name}'] = np.nan

    return aggregated


def aggregate_weighted(
    per_class_metrics: Dict[str, Dict[str, float]],
    class_names: list,
    support: Dict[str, int]
) -> Dict[str, float]:
    """
    Compute weighted average (weight by number of positive samples)

    Args:
        per_class_metrics: {class_name: {metric_name: value}}
        class_names: List of class names
        support: {class_name: num_positive_samples}

    Returns:
        {'weighted_auroc': float, 'weighted_f1': float, ...}
    """
    if not per_class_metrics or not class_names:
        return {}

    # Get metric names
    first_class = class_names[0]
    metric_names = list(per_class_metrics[first_class].keys())

    # Compute total support
    total_support = sum(support[class_name] for class_name in class_names if class_name in support)

    if total_support == 0:
        # No support, return nan
        return {f'weighted_{name}': np.nan for name in metric_names}

    aggregated = {}

    # Weighted average for each metric
    for metric_name in metric_names:
        weighted_sum = 0.0
        valid_support = 0

        for class_name in class_names:
            if class_name in per_class_metrics and class_name in support:
                value = per_class_metrics[class_name][metric_name]
                class_support = support[class_name]

                if not np.isnan(value) and class_support > 0:
                    weighted_sum += value * class_support
                    valid_support += class_support

        if valid_support > 0:
            aggregated[f'weighted_{metric_name}'] = float(weighted_sum / valid_support)
        else:
            aggregated[f'weighted_{metric_name}'] = np.nan

    return aggregated


# ============================================================================
# Complete Disease Metrics
# ============================================================================

def compute_disease_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_names: list = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute complete disease detection metrics

    Always computes:
    - Macro AUROC, AUPRC, F1, Precision, Recall, Specificity, Accuracy
    - Weighted AUROC, AUPRC, F1, Precision, Recall, Specificity, Accuracy
    - Per-disease metrics (if class_names provided)

    Args:
        y_pred: Predicted scores (N, num_classes) in [0, 1]
        y_true: True labels (N, num_classes) with 0/1
        class_names: List of disease names (optional)
        threshold: Binarization threshold for precision/recall/accuracy computation

    Returns:
        {
            'macro_auroc': float,
            'macro_auprc': float,
            'macro_f1': float,
            'macro_precision': float,
            'macro_recall': float,
            'macro_specificity': float,
            'macro_accuracy': float,
            'weighted_auroc': float,
            'weighted_auprc': float,
            'weighted_f1': float,
            'weighted_precision': float,
            'weighted_recall': float,
            'weighted_specificity': float,
            'weighted_accuracy': float,
            'disease_name_auroc': float,  # Per-disease (if class_names provided)
            'disease_name_precision': float,
            'disease_name_recall': float,
            ...
        }
    """
    num_classes = y_pred.shape[1]

    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]

    if len(class_names) != num_classes:
        raise ValueError(f"class_names length {len(class_names)} != num_classes {num_classes}")

    # Compute per-class metrics
    per_class = {}
    support = {}

    for i, class_name in enumerate(class_names):
        y_true_cls = y_true[:, i]
        y_pred_cls = y_pred[:, i]

        # Support (number of positive samples)
        support[class_name] = int(np.sum(y_true_cls))

        # Metrics based on probabilities
        auroc = compute_auroc(y_true_cls, y_pred_cls)
        auprc = compute_auprc(y_true_cls, y_pred_cls)
        f1_opt, _ = compute_f1_optimal(y_true_cls, y_pred_cls)

        # Binarize predictions for confusion matrix metrics
        y_pred_binary = (y_pred_cls >= threshold).astype(int)

        # Confusion matrix based metrics
        confusion_metrics = compute_confusion_metrics(y_true_cls, y_pred_binary)

        per_class[class_name] = {
            'auroc': auroc,
            'auprc': auprc,
            'f1': f1_opt,
            'precision': confusion_metrics['precision'],
            'recall': confusion_metrics['recall'],
            'specificity': confusion_metrics['specificity'],
            'accuracy': confusion_metrics['accuracy']
        }

    # Aggregate metrics
    results = {}

    # Macro average
    macro_metrics = aggregate_macro(per_class, class_names)
    results.update(macro_metrics)

    # Weighted average
    weighted_metrics = aggregate_weighted(per_class, class_names, support)
    results.update(weighted_metrics)

    # Per-disease metrics
    for class_name in class_names:
        for metric_name, value in per_class[class_name].items():
            key = f"{class_name}_{metric_name}"
            results[key] = value

    return results

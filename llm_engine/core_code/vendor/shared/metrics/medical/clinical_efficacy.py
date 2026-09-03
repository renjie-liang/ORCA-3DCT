"""
Clinical Efficacy Metrics for Radiology Report Evaluation

Evaluates how accurately AI-generated reports identify medical conditions
compared to ground truth using RadBERT-based automated labeler.

Extracts 18 clinical findings from generated and reference texts,
then computes Precision, Recall, and F1-score.

Usage as function:
    from shared.metrics.medical.clinical_efficacy import compute_clinical_efficacy
    
    metrics = compute_clinical_efficacy(
        hypotheses=["generated report 1", "generated report 2"],
        references=["ground truth 1", "ground truth 2"],
        checkpoint_path="/path/to/RadBertClassifier.pth"
    )
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")

Usage as script:
    python clinical_efficacy.py --predictions predictions.jsonl --checkpoint model.pth
"""

import csv
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Union
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModel
from scipy.special import expit
from sklearn.metrics import multilabel_confusion_matrix


# 18 Clinical Findings (from CT-RATE dataset)
CLINICAL_FINDINGS = [
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


class RadBertClassifier(nn.Module):
    """
    RadBERT-based classifier for extracting clinical findings from reports
    """
    def __init__(self, n_classes: int = 18):
        super().__init__()
        self.config = AutoConfig.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
        self.model = AutoModel.from_pretrained('zzxslp/RadBERT-RoBERTa-4m', config=self.config)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attn_mask):
        output = self.model(input_ids=input_ids, attention_mask=attn_mask)
        output = self.classifier(output.pooler_output)
        return output


def extract_findings(
    reports: List[str],
    checkpoint_path: str,
    device: str = 'cuda',
    batch_size: int = 16,
    max_length: int = 512
) -> np.ndarray:
    """
    Extract clinical findings from radiology reports using RadBERT
    
    Args:
        reports: List of radiology reports
        checkpoint_path: Path to RadBertClassifier.pth
        device: Device to use ('cuda' or 'cpu')
        batch_size: Batch size for inference
        max_length: Max token length for tokenizer
    
    Returns:
        Binary predictions array of shape (num_reports, 18)
    """
    # Initialize model
    model = RadBertClassifier(n_classes=18)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load state dict with strict=False to ignore position_ids mismatch
    # This is a known issue with newer transformers versions
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('zzxslp/RadBERT-RoBERTa-4m', do_lower_case=True)
    
    # Extract findings
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(reports), batch_size):
            batch_reports = reports[i:i + batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_reports,
                return_tensors='pt',
                max_length=max_length,
                padding='max_length',
                truncation=True
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Convert logits to probabilities
            probs = expit(logits.cpu().numpy())
            
            # Binarize (threshold = 0.5)
            binary_preds = (probs >= 0.5).astype(int)
            
            all_predictions.append(binary_preds)
    
    # Concatenate all batches
    predictions = np.vstack(all_predictions)
    
    return predictions


def _compute_clinical_metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    cm = multilabel_confusion_matrix(y_true, y_pred)

    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []

    print("Per-finding metrics:")
    print("-" * 80)

    for finding_name, matrix in zip(CLINICAL_FINDINGS, cm):
        TN, FP, FN, TP = matrix.ravel()

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
        support = TP + FN

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        support_list.append(support)

        print(f"{finding_name:40s} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f} | Support: {support}")

    total_support = sum(support_list)

    if total_support > 0:
        weighted_precision = sum(p * s for p, s in zip(precision_list, support_list)) / total_support
        weighted_recall = sum(r * s for r, s in zip(recall_list, support_list)) / total_support
        weighted_f1 = sum(f * s for f, s in zip(f1_list, support_list)) / total_support
    else:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

    print()
    print("=" * 80)
    print("Overall Clinical Efficacy Metrics (Weighted Average)")
    print("=" * 80)
    print(f"Precision: {weighted_precision:.4f}")
    print(f"Recall:    {weighted_recall:.4f}")
    print(f"F1-Score:  {weighted_f1:.4f}")
    print("=" * 80)

    return {
        'precision': float(weighted_precision),
        'recall': float(weighted_recall),
        'f1': float(weighted_f1),
        'detailed': {
            finding_name: {
                'precision': float(precision_list[i]),
                'recall': float(recall_list[i]),
                'f1': float(f1_list[i]),
                'support': int(support_list[i])
            }
            for i, finding_name in enumerate(CLINICAL_FINDINGS)
        }
    }


def compute_clinical_efficacy_from_labels(
    hypotheses: List[str],
    y_true_labels: Union[List[List[int]], np.ndarray],
    checkpoint_path: str = "./checkpoints/RadBertClassifier.pth",
    device: str = 'cuda'
) -> Dict[str, float]:
    if len(hypotheses) == 0:
        raise ValueError("Empty input: need at least one hypothesis")

    y_true = np.asarray(y_true_labels)
    if y_true.ndim != 2 or y_true.shape[1] != len(CLINICAL_FINDINGS):
        raise ValueError(
            f"y_true_labels must have shape (N, {len(CLINICAL_FINDINGS)}), got {tuple(y_true.shape)}"
        )
    if y_true.shape[0] != len(hypotheses):
        raise ValueError(
            f"Length mismatch: {len(hypotheses)} hypotheses vs {y_true.shape[0]} y_true_labels"
        )

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"RadBERT checkpoint not found at: {checkpoint_path}\n"
            f"Download from: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/resolve/main/models/RadBertClassifier.pth"
        )

    print("=" * 80)
    print("Computing Clinical Efficacy Metrics")
    print("=" * 80)
    print(f"Number of reports: {len(hypotheses)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print()

    print("Extracting findings from generated reports...")
    y_pred = extract_findings(hypotheses, checkpoint_path, device)
    print(f"  Shape: {y_pred.shape}")
    print(f"  Using provided labels, shape: {y_true.shape}")
    print()

    return _compute_clinical_metrics_from_arrays(y_true=y_true.astype(int), y_pred=y_pred.astype(int))


# ============================================================
# Shared utilities for clinical label loading and F1 computation
# ============================================================

DEFAULT_CLINICAL_LABELS_CSV = "./data/labels/valid_predicted_labels.csv"
DEFAULT_RADBERT_CHECKPOINT = "./checkpoints/RadBertClassifier.pth"


def _normalize_volume_name(name: str) -> str:
    s = str(name).strip()
    s = Path(s).name
    if s.endswith(".nii.gz"):
        s = s[:-7]
    return s


def load_clinical_labels_csv(
    csv_path: str = DEFAULT_CLINICAL_LABELS_CSV,
) -> Dict[str, List[int]]:
    """Load VolumeName -> 18-dim binary label vector from CT-RATE CSV.

    Args:
        csv_path: Path to the multi_abnormality_labels CSV file.

    Returns:
        Dict mapping normalized volume name to list of 18 binary ints.
    """
    labels = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vol = _normalize_volume_name(row["VolumeName"])
            vec = []
            for finding in CLINICAL_FINDINGS:
                v = str(row[finding]).strip()
                try:
                    vec.append(1 if int(float(v)) == 1 else 0)
                except ValueError:
                    vec.append(0)
            labels[vol] = vec
    return labels


def compute_clinical_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    finding_names: List[str],
) -> Dict:
    """Compute per-finding P/R/F1, weighted/micro/macro averages.

    The primary precision/recall/f1 are **weighted** averages (by support),
    matching the CT-CLIP / CT-RATE official evaluation protocol
    (text_classifier/eval.py in https://github.com/ibrahimethemhamamci/CT-CLIP).

    Args:
        y_true: Ground truth binary array of shape (N, num_findings).
        y_pred: Predicted binary array of shape (N, num_findings).
        finding_names: List of finding names, length == num_findings.

    Returns:
        Dict with keys:
          precision, recall, f1        — weighted average (official metric)
          micro_precision, micro_recall, micro_f1
          macro_f1
          per_finding: {name: {precision, recall, f1, support}}
    """
    total_tp, total_fp, total_fn = 0, 0, 0
    per_finding = {}
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []

    for j, fname in enumerate(finding_names):
        tp = int(((y_pred[:, j] == 1) & (y_true[:, j] == 1)).sum())
        fp = int(((y_pred[:, j] == 1) & (y_true[:, j] == 0)).sum())
        fn = int(((y_pred[:, j] == 0) & (y_true[:, j] == 1)).sum())

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = int(y_true[:, j].sum())

        per_finding[fname] = {"precision": p, "recall": r, "f1": f1, "support": support}
        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f1)
        support_list.append(support)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Weighted average (CT-CLIP official metric)
    total_support = sum(support_list)
    if total_support > 0:
        weighted_p = sum(p * s for p, s in zip(precision_list, support_list)) / total_support
        weighted_r = sum(r * s for r, s in zip(recall_list, support_list)) / total_support
        weighted_f1 = sum(f * s for f, s in zip(f1_list, support_list)) / total_support
    else:
        weighted_p = 0.0
        weighted_r = 0.0
        weighted_f1 = 0.0

    # Micro average
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # Macro average
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    return {
        "precision": weighted_p,
        "recall": weighted_r,
        "f1": weighted_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_finding": per_finding,
    }

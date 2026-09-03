"""
Example usage of Clinical Efficacy Metrics

This shows how to use the clinical_efficacy module both as a function
and as a command-line tool.
"""

# ============================================================================
# Method 1: Use as a function in your Python code
# ============================================================================

from shared.metrics.medical import compute_clinical_efficacy_from_labels

# Example data
predictions = [
    "Cardiomegaly is present. Pleural effusion noted bilaterally.",
    "Normal chest CT scan. No acute findings.",
    "Emphysema and bronchiectasis observed. Small lung nodule in right upper lobe."
]

y_true_labels = [
    [0] * 18,
    [0] * 18,
    [0] * 18,
]

# Compute metrics
metrics = compute_clinical_efficacy_from_labels(
    hypotheses=predictions,
    y_true_labels=y_true_labels,
    checkpoint_path="./checkpoints/RadBertClassifier.pth",
    device='cuda'
)

print("\nResults:")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1-Score:  {metrics['f1']:.4f}")


# ============================================================================
# Method 2: Use as command-line tool with JSONL file
# ============================================================================

# From terminal, run:
"""
python shared/metrics/medical/clinical_efficacy.py \
    --predictions results/stage3_visd_20260129_191208/baseline/predictions/predictions_step100000.jsonl \
    --checkpoint ./checkpoints/RadBertClassifier.pth \
    --output metrics_results.json
"""

# Or with default checkpoint path:
"""
python shared/metrics/medical/clinical_efficacy.py \
    --predictions results/stage3_visd_20260129_191208/baseline/predictions/predictions_step100000.jsonl \
    --output metrics_results.json
"""


# ============================================================================
# Method 3: Load from JSONL and compute in Python
# ============================================================================

import json

def evaluate_from_jsonl(jsonl_path: str):
    """Load predictions from JSONL and compute metrics"""
    predictions = []
    y_true_labels = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data['prediction'])
            y_true_labels.append(data['y_true_labels'])
    
    metrics = compute_clinical_efficacy_from_labels(
        hypotheses=predictions,
        y_true_labels=y_true_labels
    )
    
    return metrics

# Usage:
# metrics = evaluate_from_jsonl('predictions.jsonl')
# print(metrics)

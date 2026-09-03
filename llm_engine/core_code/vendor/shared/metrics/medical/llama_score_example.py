"""
Example usage of Llama Score evaluation

This shows how to use the llama_score module both as a function
and as a command-line tool.
"""

# ============================================================================
# Method 1: Use as a function in your Python code
# ============================================================================

from shared.metrics.medical import compute_llama_score

# Example data
predictions = [
    "Cardiomegaly is present. Pleural effusion noted bilaterally. The heart is enlarged.",
    "Normal chest CT scan. No acute findings. Lung fields are clear.",
    "Emphysema and bronchiectasis observed. Small lung nodule in right upper lobe measuring 5mm."
]

references = [
    "Enlarged heart with bilateral pleural effusions. Cardiac silhouette is increased in size.",
    "Unremarkable chest CT examination. No abnormalities detected.",
    "Chronic obstructive changes with emphysema. Nodular density of 6mm in RUL."
]

# Compute Llama Score
results = compute_llama_score(
    hypotheses=predictions,
    references=references,
    task_type='report_generation',
    model_path="./checkpoints/Meta-Llama-3-70B-Instruct"
)

print("\nResults:")
print(f"Llama Score: {results['llama_score']:.2f} / 10")
print(f"Min Score: {results['min_score']:.1f}")
print(f"Max Score: {results['max_score']:.1f}")
print(f"Failed parses: {results['failed_parses']}")


# ============================================================================
# Method 2: Use as command-line tool with JSONL file
# ============================================================================

# From terminal, run:
"""
python shared/metrics/medical/llama_score.py \
    --predictions results/stage3_visd_20260129_191208/baseline/predictions/predictions_step100000.jsonl \
    --model_path ./checkpoints/Meta-Llama-3-70B-Instruct \
    --task_type report_generation \
    --output llama_score_results.json
"""

# Or with default model path:
"""
python shared/metrics/medical/llama_score.py \
    --predictions results/stage3_visd_20260129_191208/baseline/predictions/predictions_step100000.jsonl \
    --output llama_score_results.json
"""

# For multi-GPU inference (if you have multiple GPUs):
"""
python shared/metrics/medical/llama_score.py \
    --predictions results/stage3_visd_20260129_191208/baseline/predictions/predictions_step100000.jsonl \
    --tensor_parallel_size 2 \
    --output llama_score_results.json
"""


# ============================================================================
# Method 3: Load from JSONL and compute in Python
# ============================================================================

import json

def evaluate_from_jsonl(jsonl_path: str):
    """Load predictions from JSONL and compute Llama Score"""
    predictions = []
    references = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data['prediction'])
            references.append(data['reference'])
    
    results = compute_llama_score(
        hypotheses=predictions,
        references=references,
        task_type='report_generation'
    )
    
    return results

# Usage:
# results = evaluate_from_jsonl('predictions.jsonl')
# print(f"Llama Score: {results['llama_score']:.2f}/10")


# ============================================================================
# Method 4: With custom questions (for other task types)
# ============================================================================

# If you have custom questions (e.g., for long_answer, short_answer tasks)
questions = [
    "What pathologies are present in this chest CT?",
    "Is there any evidence of pneumonia?",
    "Describe the findings in the lung parenchyma."
]

predictions_custom = [
    "Cardiomegaly and pleural effusion are present.",
    "No, there is no evidence of pneumonia. The lungs are clear.",
    "The lung parenchyma shows emphysematous changes with some nodular densities."
]

references_custom = [
    "Enlarged heart and bilateral effusions.",
    "No pneumonia. Normal lung fields.",
    "Emphysema with multiple small nodules."
]

results_custom = compute_llama_score(
    hypotheses=predictions_custom,
    references=references_custom,
    questions=questions,
    task_type='short_answer'  # Can be 'long_answer', 'short_answer', 'multiple_choice'
)

print(f"\nCustom task Llama Score: {results_custom['llama_score']:.2f}/10")


# ============================================================================
# Notes on Usage
# ============================================================================

"""
Performance Tips:
- For large batches (>100 samples), inference may take 30+ minutes
- Use device_map="auto" for multi-GPU (model automatically uses all available GPUs)
- Temperature=0 ensures deterministic evaluation

Score Interpretation:
- 9-10: Excellent clinical accuracy
- 7-8: Good accuracy with minor issues
- 5-6: Moderate accuracy with noticeable errors
- 3-4: Poor accuracy with significant errors
- 0-2: Unacceptable quality

Model Requirements:
- Requires ~140GB GPU memory for 70B model (single GPU)
- Or ~70GB per GPU with device_map="auto" across multiple GPUs
"""

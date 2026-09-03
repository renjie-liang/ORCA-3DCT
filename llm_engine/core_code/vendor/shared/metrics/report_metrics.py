"""
Unified report evaluation metrics

Combines text metrics (BLEU, ROUGE, METEOR, CIDEr)
and medical metrics (CheXbert, RadGraph)

Fail-fast principles:
- Direct function calls
- No silent failures
- Explicit errors
"""
from typing import List, Dict

from .text.text_metrics import compute_all_text_metrics
from .medical.medical_metrics import compute_all_medical_metrics


def evaluate_reports(
    hypotheses: List[str],
    references: List[str],
    compute_text_metrics: bool = True,
    compute_medical_metrics: bool = True,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Comprehensive report evaluation

    Computes all available metrics for generated radiology reports

    Args:
        hypotheses: Generated reports (list of strings)
        references: Reference reports (list of strings)
        compute_text_metrics: Whether to compute text metrics
        compute_medical_metrics: Whether to compute medical metrics
        device: Device for medical metrics ('cuda' or 'cpu')

    Returns:
        {
            # Text metrics
            'bleu_1': float,
            'bleu_2': float,
            'bleu_3': float,
            'bleu_4': float,
            'bleu_sacre': float,
            'rouge_l': float,
            'meteor': float,
            'cider': float,

            # Medical metrics
            'chexbert_f1': float,
            'radgraph_simple': float,
            'radgraph_partial': float,
            'radgraph_complete': float
        }

    Example:
        >>> hyps = ["There is cardiomegaly.", "No acute findings."]
        >>> refs = ["Enlarged cardiac silhouette.", "Normal chest radiograph."]
        >>> metrics = evaluate_reports(hyps, refs)
        >>>
        >>> print(f"BLEU-4: {metrics['bleu_4']:.4f}")
        >>> print(f"CheXbert F1: {metrics['chexbert_f1']:.4f}")
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"Length mismatch: {len(hypotheses)} hypotheses vs {len(references)} references"
        )

    if len(hypotheses) == 0:
        raise ValueError("Empty input: need at least one hypothesis-reference pair")

    print("=" * 80)
    print(f"Evaluating {len(hypotheses)} generated reports")
    print("=" * 80)

    results = {}

    # Text metrics
    if compute_text_metrics:
        print("\n" + "=" * 80)
        print("Computing Text Metrics")
        print("=" * 80)

        text_metrics = compute_all_text_metrics(hypotheses, references)
        results.update(text_metrics)

    # Medical metrics
    if compute_medical_metrics:
        print("\n" + "=" * 80)
        print("Computing Medical Metrics")
        print("=" * 80)

        medical_metrics = compute_all_medical_metrics(
            hypotheses,
            references,
            device=device
        )
        results.update(medical_metrics)

    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)
    print("\nSummary:")

    if 'bleu_4' in results:
        print(f"  BLEU-4: {results['bleu_4']:.4f}")
    if 'rouge_l' in results:
        print(f"  ROUGE-L: {results['rouge_l']:.4f}")
    if 'chexbert_f1' in results:
        print(f"  CheXbert F1: {results['chexbert_f1']:.4f}")
    if 'radgraph_complete' in results:
        print(f"  RadGraph Complete F1: {results['radgraph_complete']:.4f}")

    print("=" * 80)

    return results

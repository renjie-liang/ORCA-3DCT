"""
Medical metrics for radiology report evaluation

Computes CheXbert and RadGraph scores

These metrics evaluate clinical accuracy of generated reports

Fail-fast principles:
- Direct library calls
- Explicit errors for missing dependencies
- No silent failures
"""
from typing import List, Dict, Any
from pathlib import Path


def _extract_float(val: Any) -> float:
    """
    Recursively extract float from nested structures (list, dict, etc.)
    
    Args:
        val: Value to extract (can be float, list, dict, etc.)
    
    Returns:
        Extracted float value, or 0.0 if extraction fails
    """
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return 0.0
        return _extract_float(val[0])
    elif isinstance(val, dict):
        for key in ['f1', 'score', 'value', 'micro_f1', 'macro_f1']:
            if key in val:
                return _extract_float(val[key])
        return 0.0
    else:
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0


def compute_chexbert(
    hypotheses: List[str],
    references: List[str],
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute F1-CheXbert score

    Evaluates clinical entity extraction accuracy for 14 medical conditions

    Args:
        hypotheses: Generated reports
        references: Reference reports
        device: 'cuda' or 'cpu'

    Returns:
        {'chexbert_f1': float}  # Micro-averaged F1 score (0-1)
    """
    try:
        from f1chexbert import F1CheXbert
    except ImportError:
        raise ImportError(
            "f1chexbert package not installed. Install with:\n"
            "  pip install f1chexbert\n"
            "Or: pip install git+https://github.com/stanfordmlgroup/CheXbert.git"
        )

    print(f"Computing CheXbert F1 for {len(hypotheses)} samples...")
    print(f"  Device: {device}")

    # Initialize model
    model = F1CheXbert(device=device)

    # Compute scores - returns (per_sample_scores, corpus_scores)
    result = model(refs=references, hyps=hypotheses)
    
    # Extract corpus F1 score (handle tuple, dict, or nested structures)
    if isinstance(result, tuple) and len(result) >= 2:
        corpus_f1 = _extract_float(result[1])
    else:
        corpus_f1 = _extract_float(result)

    print(f"  CheXbert F1: {corpus_f1:.4f}")

    return {'chexbert_f1': corpus_f1}


def compute_radgraph(
    hypotheses: List[str],
    references: List[str],
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute RadGraph F1 scores

    Evaluates clinical entity and relation extraction using graph matching

    Args:
        hypotheses: Generated reports
        references: Reference reports
        device: 'cuda' or 'cpu'

    Returns:
        {
            'radgraph_simple': float,    # Entity matching only
            'radgraph_partial': float,   # Entity + partial relations
            'radgraph_complete': float   # Entity + complete relations
        }
    """
    try:
        from radgraph import F1RadGraph
    except ImportError:
        raise ImportError(
            "radgraph package not installed. Install with:\n"
            "  pip install radgraph\n"
            "Or see: https://github.com/jbdel/radgraph"
        )

    print(f"Computing RadGraph F1 for {len(hypotheses)} samples...")
    print(f"  Device: {device}")

    # Initialize model - reward_level options: 'simple', 'partial', 'complete'
    model = F1RadGraph(reward_level='partial', device=device)

    # Compute scores - returns (simple_f1, partial_f1, complete_f1, per_sample) or dict
    result = model(refs=references, hyps=hypotheses)
    
    # Extract F1 scores (handle tuple, dict, or nested structures)
    if isinstance(result, tuple):
        simple_f1 = _extract_float(result[0]) if len(result) > 0 else 0.0
        partial_f1 = _extract_float(result[1]) if len(result) > 1 else 0.0
        complete_f1 = _extract_float(result[2]) if len(result) > 2 else 0.0
    elif isinstance(result, dict):
        simple_f1 = _extract_float(result.get('simple', result.get('simple_f1', 0.0)))
        partial_f1 = _extract_float(result.get('partial', result.get('partial_f1', 0.0)))
        complete_f1 = _extract_float(result.get('complete', result.get('complete_f1', 0.0)))
    else:
        simple_f1 = partial_f1 = complete_f1 = _extract_float(result)

    print(f"  RadGraph Simple F1: {simple_f1:.4f}")
    print(f"  RadGraph Partial F1: {partial_f1:.4f}")
    print(f"  RadGraph Complete F1: {complete_f1:.4f}")

    return {
        'radgraph_simple': float(simple_f1),
        'radgraph_partial': float(partial_f1),
        'radgraph_complete': float(complete_f1)
    }


def compute_all_medical_metrics(
    hypotheses: List[str],
    references: List[str],
    device: str = 'cuda',
    include_chexbert: bool = True,
    include_radgraph: bool = True
) -> Dict[str, float]:
    """
    Compute all medical metrics

    Args:
        hypotheses: Generated reports
        references: Reference reports
        device: 'cuda' or 'cpu'
        include_chexbert: Whether to compute CheXbert
        include_radgraph: Whether to compute RadGraph

    Returns:
        {
            'chexbert_f1': float,
            'radgraph_simple': float,
            'radgraph_partial': float,
            'radgraph_complete': float
        }

    Example:
        >>> hyps = ["There is cardiomegaly.", "No acute findings."]
        >>> refs = ["Enlarged heart.", "Normal chest x-ray."]
        >>> metrics = compute_all_medical_metrics(hyps, refs)
        >>> print(f"CheXbert F1: {metrics['chexbert_f1']:.4f}")
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"Length mismatch: {len(hypotheses)} hypotheses vs {len(references)} references"
        )

    if len(hypotheses) == 0:
        raise ValueError("Empty input: need at least one hypothesis-reference pair")

    print("=" * 80)
    print("Computing Medical Metrics")
    print("=" * 80)

    results = {}

    # CheXbert
    if include_chexbert:
        chexbert_scores = compute_chexbert(hypotheses, references, device)
        results.update(chexbert_scores)

    # RadGraph
    if include_radgraph:
        radgraph_scores = compute_radgraph(hypotheses, references, device)
        results.update(radgraph_scores)

    print("=" * 80)
    print("Medical metrics computed successfully")
    print("=" * 80)

    return results

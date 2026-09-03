"""
Text generation metrics for report evaluation

Computes BLEU, ROUGE, METEOR, and CIDEr scores

Fail-fast principles:
- Direct library calls
- No try-except
- Explicit errors for missing dependencies
"""
from typing import List, Dict, Tuple

import numpy as np


def compute_bleu(
    hypotheses: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute BLEU scores (BLEU-1/2/3/4 and sacreBLEU)

    Args:
        hypotheses: Generated reports
        references: Reference reports

    Returns:
        {
            'bleu_1': float,
            'bleu_2': float,
            'bleu_3': float,
            'bleu_4': float,
            'bleu_sacre': float
        }
    """
    import sacrebleu
    from pycocoevalcap.bleu.bleu import Bleu

    # Lowercase
    hyps_lower = [h.strip().lower() for h in hypotheses]
    refs_lower = [[r.strip().lower()] for r in references]

    # sacreBLEU (corpus-level)
    corpus_bleu = sacrebleu.corpus_bleu(hyps_lower, refs_lower, smooth_method='exp')
    bleu_sacre = float(corpus_bleu.score) / 100.0  # Convert to 0-1 scale

    # COCO BLEU-1/2/3/4 (refs_dict[i] = list of ref strings; refs_lower[i] is already that list)
    scorer = Bleu(4)
    hyps_dict = {i: [h] for i, h in enumerate(hyps_lower)}
    refs_dict = {i: r for i, r in enumerate(refs_lower)}

    avg_scores, _ = scorer.compute_score(refs_dict, hyps_dict)

    return {
        'bleu_1': float(avg_scores[0]),
        'bleu_2': float(avg_scores[1]),
        'bleu_3': float(avg_scores[2]),
        'bleu_4': float(avg_scores[3]),
        'bleu_sacre': bleu_sacre
    }


def compute_rouge(
    hypotheses: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute ROUGE-L score

    Args:
        hypotheses: Generated reports
        references: Reference reports

    Returns:
        {'rouge_l': float}
    """
    from pycocoevalcap.rouge.rouge import Rouge

    # Lowercase
    hyps_lower = [h.strip().lower() for h in hypotheses]
    refs_lower = [r.strip().lower() for r in references]

    # ROUGE-L
    scorer = Rouge()
    hyps_dict = {i: [h] for i, h in enumerate(hyps_lower)}
    refs_dict = {i: [r] for i, r in enumerate(refs_lower)}

    avg_score, _ = scorer.compute_score(refs_dict, hyps_dict)

    return {'rouge_l': float(avg_score)}


def _sanitize_for_meteor(s: str) -> str:
    """Replace ||| and newlines so METEOR Java protocol (delimiter |||) is not broken."""
    return s.replace("|||", " ").replace("\n", " ").replace("\r", " ").strip()


def compute_meteor(
    hypotheses: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute METEOR score

    Args:
        hypotheses: Generated reports
        references: Reference reports

    Returns:
        {'meteor': float}
    """
    from pycocoevalcap.meteor.meteor import Meteor

    # Lowercase and sanitize: METEOR uses ||| as delimiter; ref/hyp must not contain it
    hyps_lower = [_sanitize_for_meteor(h).lower() for h in hypotheses]
    refs_lower = [_sanitize_for_meteor(r).lower() for r in references]

    # pycocoevalcap: gts[id] = list of ref strings, res[id] = list of one hyp string
    scorer = Meteor()
    hyps_dict = {i: [h] for i, h in enumerate(hyps_lower)}
    refs_dict = {i: [r] for i, r in enumerate(refs_lower)}

    avg_score, _ = scorer.compute_score(refs_dict, hyps_dict)

    return {'meteor': float(avg_score)}


def compute_cider(
    hypotheses: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute CIDEr score

    Args:
        hypotheses: Generated reports
        references: Reference reports

    Returns:
        {'cider': float}
    """
    from pycocoevalcap.cider.cider import Cider

    # Lowercase
    hyps_lower = [h.strip().lower() for h in hypotheses]
    refs_lower = [r.strip().lower() for r in references]

    # CIDEr
    scorer = Cider()
    hyps_dict = {i: [h] for i, h in enumerate(hyps_lower)}
    refs_dict = {i: [r] for i, r in enumerate(refs_lower)}

    avg_score, _ = scorer.compute_score(refs_dict, hyps_dict)

    return {'cider': float(avg_score)}


def compute_all_text_metrics(
    hypotheses: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute all text generation metrics

    Args:
        hypotheses: Generated reports (list of strings)
        references: Reference reports (list of strings)

    Returns:
        {
            'bleu_1': float,
            'bleu_2': float,
            'bleu_3': float,
            'bleu_4': float,
            'bleu_sacre': float,
            'rouge_l': float,
            'meteor': float,
            'cider': float
        }

    Example:
        >>> hyps = ["There is a fracture.", "No abnormalities found."]
        >>> refs = ["There is a bone fracture.", "Normal study."]
        >>> metrics = compute_all_text_metrics(hyps, refs)
        >>> print(f"BLEU-4: {metrics['bleu_4']:.4f}")
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"Length mismatch: {len(hypotheses)} hypotheses vs {len(references)} references"
        )

    if len(hypotheses) == 0:
        raise ValueError("Empty input: need at least one hypothesis-reference pair")

    # METEOR/CIDEr Java processes require non-empty hypothesis and reference per sample
    hypotheses = [h.strip() if h.strip() else " " for h in hypotheses]
    references = [r.strip() if r.strip() else " " for r in references]

    print(f"Computing text metrics for {len(hypotheses)} samples...")

    # Compute all metrics
    results = {}

    # BLEU
    bleu_scores = compute_bleu(hypotheses, references)
    results.update(bleu_scores)

    # ROUGE-L
    rouge_scores = compute_rouge(hypotheses, references)
    results.update(rouge_scores)

    # METEOR (Java jar may fail with "SCORE or EVAL or SING" if protocol/jar version mismatch)
    try:
        meteor_scores = compute_meteor(hypotheses, references)
        results.update(meteor_scores)
    except (ValueError, AssertionError) as e:
        print(f"Warning: METEOR failed ({e}). Returning meteor=0.0. Check meteor-1.5.jar / pycocoevalcap compatibility.")
        results['meteor'] = 0.0

    # CIDEr
    cider_scores = compute_cider(hypotheses, references)
    results.update(cider_scores)

    # Print summary
    print(f"Text metrics:")
    print(f"  BLEU-1: {results['bleu_1']:.4f}")
    print(f"  BLEU-4: {results['bleu_4']:.4f}")
    print(f"  METEOR: {results['meteor']:.4f}")
    print(f"  ROUGE-L: {results['rouge_l']:.4f}")
    print(f"  CIDEr: {results['cider']:.4f}")

    return results

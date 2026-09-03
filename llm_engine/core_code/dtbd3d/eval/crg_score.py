"""
crg_score.py — Distribution-Aware Clinical Metric for Radiology Report Generation.

Implements the CRG Score from:

    Hamamci et al., "CRG Score: A Distribution-Aware Clinical Metric for
    Radiology Report Generation", MIDL 2025 (Short Papers), arXiv:2505.17167.

Algorithm (Eq. 1-3 of the paper):

    T = TP + FP + FN + TN          (total labels in test set)
    A = TP + FN                    (positive labels in ground truth)
    w_TP = w_FN = (T - A) / (2 A)
    w_FP = 1
    S_max = A * w_TP
    s = TP * w_TP - FN * w_FN - FP * w_FP
    CRG = S_max / (2 * S_max - s)

Properties:
- Trivial predictions (always-normal or always-abnormal) both yield CRG = 1/3.
- Higher values indicate better clinical performance.
- The metric uses dataset-level confusion-matrix totals, not per-volume averaging.
"""
from typing import Dict


def crg_score(TP: int, FP: int, FN: int, TN: int) -> float:
    """Compute CRG from dataset-level confusion-matrix totals.

    Args:
        TP, FP, FN, TN: integers, summed across all volumes and all 18 labels.

    Returns:
        CRG in (0, 1]; trivial predictions yield 1/3.

    Raises:
        ValueError: if TP+FN == 0 (no positive ground truth in the test set).
    """
    A = TP + FN
    if A <= 0:
        raise ValueError("CRG undefined: no positive labels in ground truth (A = TP + FN = 0)")
    T = TP + FP + FN + TN
    if T <= A:
        raise ValueError(f"CRG undefined: T (={T}) must exceed A (={A})")

    w_pos = (T - A) / (2.0 * A)            # w_TP = w_FN
    w_FP = 1.0
    S_max = A * w_pos
    s = TP * w_pos - FN * w_pos - FP * w_FP

    denom = 2.0 * S_max - s
    if denom == 0:
        raise ValueError(f"CRG undefined: denominator is zero (S_max={S_max}, s={s})")
    return S_max / denom


def crg_score_with_components(TP: int, FP: int, FN: int, TN: int) -> Dict[str, float]:
    """Same as crg_score, but returns intermediate components for diagnostics."""
    A = TP + FN
    T = TP + FP + FN + TN
    w_pos = (T - A) / (2.0 * A)
    w_FP = 1.0
    S_max = A * w_pos
    s = TP * w_pos - FN * w_pos - FP * w_FP
    crg = S_max / (2.0 * S_max - s)
    return {
        "crg": crg,
        "T": T, "A": A,
        "w_TP": w_pos, "w_FN": w_pos, "w_FP": w_FP,
        "S_max": S_max, "s": s,
    }


# ---------- Unit tests ----------
def _test_paper_table2():
    """Reproduces the four model rows of CRG paper Table 2 (CT-RATE valid)."""
    cases = [
        # model,       TP,    FP,   FN,    TN,    expected_CRG
        ("RadFM",      550,  1766, 9985, 42401, 0.335),
        ("CT2Rep",    1561,  1804, 8974, 42363, 0.359),
        ("CT-CHAT",   2224,  3081, 8311, 41086, 0.368),
        ("Merlin",    1504,  2694, 9031, 41473, 0.352),
    ]
    for model, tp, fp, fn, tn, expected in cases:
        got = crg_score(tp, fp, fn, tn)
        # Paper rounds to 3 decimals; allow 0.005 tolerance to match
        assert abs(got - expected) < 0.005, f"{model}: expected {expected}, got {got:.4f}"
        print(f"  {model:10s}  CRG={got:.4f}  (paper: {expected})  OK")


def _test_trivial_baselines():
    """Trivial predictions (always-normal or always-abnormal) must give CRG = 1/3."""
    # Use realistic CT-RATE valid scale: 3039 volumes * 18 labels = 54702 total labels.
    # Suppose A = 10535 (CT-CHAT-style), then T - A = 44167.
    A, NEG = 10535, 44167
    # Always-normal: TP=0, FP=0, FN=A, TN=NEG
    crg_normal = crg_score(0, 0, A, NEG)
    assert abs(crg_normal - 1/3) < 1e-9, f"always-normal: {crg_normal}"
    # Always-abnormal: TP=A, FP=NEG, FN=0, TN=0
    crg_abnormal = crg_score(A, NEG, 0, 0)
    assert abs(crg_abnormal - 1/3) < 1e-9, f"always-abnormal: {crg_abnormal}"
    # Perfect predictions: TP=A, FP=0, FN=0, TN=NEG -> s = S_max -> CRG = 1.0
    crg_perfect = crg_score(A, 0, 0, NEG)
    assert abs(crg_perfect - 1.0) < 1e-9, f"perfect: {crg_perfect}"
    print(f"  always-normal:    CRG={crg_normal:.4f}  (expected 1/3)  OK")
    print(f"  always-abnormal:  CRG={crg_abnormal:.4f}  (expected 1/3)  OK")
    print(f"  perfect:          CRG={crg_perfect:.4f}  (expected 1.000)  OK")


if __name__ == "__main__":
    print("CRG paper Table 2 reproduction:")
    _test_paper_table2()
    print("\nTrivial-baseline self-tests:")
    _test_trivial_baselines()
    print("\ncrg_score selftest: PASS")

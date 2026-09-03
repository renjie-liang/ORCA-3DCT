"""
recon_metrics.py — 3D reconstruction metrics (PSNR / SSIM / MSE).

Two API levels:

  1. Global metrics — whole-volume comparison, matches BTB3D Table 1.
        compute_recon_metrics(x, x_hat) -> dict[str, float]

  2. Stratified metrics — restrict to a region given by a binary mask
     (e.g., lesion vs background voxels for the DTBD3D P3 dual-signal claim).
        compute_recon_metrics_stratified(x, x_hat, mask) -> dict[str, dict]

All functions assume both volumes are in the same scale; for BTB3D the
canonical scale is [-1, 1] with `data_range=2.0`.
"""
from typing import Dict, Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_recon_metrics(
    x: np.ndarray,
    x_hat: np.ndarray,
    data_range: float = 2.0,
) -> Dict[str, float]:
    """Whole-volume PSNR / SSIM / MSE.

    Args:
        x:        ground-truth volume, shape (D, H, W) or (H, W, D), float
        x_hat:    reconstructed volume, same shape as x
        data_range: max - min of the value range (BTB3D uses [-1, 1] => 2.0)

    Returns:
        {'psnr': float (dB), 'ssim': float, 'mse': float}
    """
    if x.shape != x_hat.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {x_hat.shape}")
    return {
        "psnr": float(peak_signal_noise_ratio(x, x_hat, data_range=data_range)),
        "ssim": float(structural_similarity(x, x_hat, data_range=data_range)),
        "mse":  float(np.mean((x - x_hat) ** 2)),
    }


def compute_recon_metrics_stratified(
    x: np.ndarray,
    x_hat: np.ndarray,
    mask: np.ndarray,
    data_range: float = 2.0,
    region_names: tuple = ("lesion", "background"),
) -> Dict[str, Dict[str, float]]:
    """Region-stratified PSNR / SSIM / MSE.

    SSIM is intrinsically a window-based metric and is undefined on
    irregular masked regions. We therefore compute SSIM only globally; the
    stratified output reports PSNR and MSE per-region (the metrics that
    matter for the DTBD3D P3 dual-signal experiment).

    Args:
        x, x_hat:   ground-truth and reconstructed volumes, same shape
        mask:       binary mask, same shape, True = "lesion" / region of interest
        region_names: (positive_name, negative_name)

    Returns:
        {
            <positive_name>: {'psnr': float, 'mse': float, 'voxels': int},
            <negative_name>: {'psnr': float, 'mse': float, 'voxels': int},
            'ratio_psnr':   <positive_name>_psnr / <negative_name>_psnr,
        }
    """
    if x.shape != x_hat.shape or x.shape != mask.shape:
        raise ValueError(f"shape mismatch: x={x.shape} x_hat={x_hat.shape} mask={mask.shape}")

    pos, neg = region_names
    out: Dict[str, Dict[str, float]] = {}
    for name, region_mask in ((pos, mask.astype(bool)), (neg, ~mask.astype(bool))):
        n = int(region_mask.sum())
        if n == 0:
            out[name] = {"psnr": float("nan"), "mse": float("nan"), "voxels": 0}
            continue
        diff = (x[region_mask] - x_hat[region_mask]).astype(np.float64)
        mse = float(np.mean(diff ** 2))
        # PSNR = 10 * log10(data_range**2 / MSE)
        psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10((data_range ** 2) / mse))
        out[name] = {"psnr": psnr, "mse": mse, "voxels": n}

    if out[pos]["psnr"] == float("inf") or out[neg]["psnr"] == float("inf") \
       or np.isnan(out[pos]["psnr"]) or np.isnan(out[neg]["psnr"]):
        out["ratio_psnr"] = float("nan")
    else:
        out["ratio_psnr"] = out[pos]["psnr"] / out[neg]["psnr"]
    return out


# ------------------------- Self-test -------------------------
def _selftest():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(20, 20, 20)).astype(np.float32)
    # Identity case
    out = compute_recon_metrics(x, x)
    assert out["mse"] == 0.0 and out["psnr"] == float("inf"), out
    # Noisy case
    x_hat = x + rng.normal(0, 0.1, size=x.shape).astype(np.float32)
    out = compute_recon_metrics(x, x_hat)
    assert 15 < out["psnr"] < 35, out
    assert 0 < out["ssim"] < 1, out
    # Stratified
    mask = np.zeros_like(x, dtype=bool)
    mask[:5, :5, :5] = True   # 5^3 = 125 voxels in "lesion"
    s = compute_recon_metrics_stratified(x, x_hat, mask)
    assert s["lesion"]["voxels"] == 125
    assert s["background"]["voxels"] == 20**3 - 125
    print("recon_metrics selftest: PASS")
    print(f"  noisy PSNR={out['psnr']:.2f} SSIM={out['ssim']:.4f} MSE={out['mse']:.5f}")
    print(f"  stratified lesion PSNR={s['lesion']['psnr']:.2f} background PSNR={s['background']['psnr']:.2f}")


if __name__ == "__main__":
    _selftest()

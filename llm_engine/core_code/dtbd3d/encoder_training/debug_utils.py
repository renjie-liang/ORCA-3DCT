"""Temporary smoke-test debug helpers.

These helpers are intentionally simple and noisy. They are meant for the
interactive smoke pass and should be removed once the E008->E001 checks pass.
"""

from __future__ import annotations

import json
import hashlib
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def tensor_stats(name: str, tensor: torch.Tensor | None) -> dict[str, Any]:
    """Return shape/dtype/device and basic finite stats for a tensor."""
    if tensor is None:
        return {"name": name, "present": False}
    data = tensor.detach()
    finite = torch.isfinite(data.float())
    out: dict[str, Any] = {
        "name": name,
        "present": True,
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "device": str(data.device),
        "finite": bool(finite.all().detach().cpu()),
    }
    if data.numel() > 0:
        values = data.float()
        out.update(
            {
                "min": float(values.min().detach().cpu()),
                "max": float(values.max().detach().cpu()),
                "mean": float(values.mean().detach().cpu()),
                "std": float(values.std(unbiased=False).detach().cpu()),
            }
        )
    return out


def print_tensor_stats(prefix: str, **tensors: torch.Tensor | None) -> None:
    """Print one compact JSON stats line per tensor."""
    for name, tensor in tensors.items():
        print(f"[debug:{prefix}] {json.dumps(tensor_stats(name, tensor), sort_keys=True)}", flush=True)


def hu_stats(name: str, tensor: torch.Tensor | None) -> dict[str, Any]:
    """Stats for cached CT tensors after converting normalized value back to HU.

    Cache contract: raw HU is clipped to [-1000, 1000], then divided by 1000.
    Therefore HU = normalized_value * 1000.
    """
    if tensor is None:
        return {"name": name, "present": False}
    data = tensor.detach().float()
    hu = data * 1000.0
    finite = torch.isfinite(hu)
    out: dict[str, Any] = {
        "name": name,
        "present": True,
        "shape": list(data.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "unit": "HU",
        "finite": bool(finite.all().detach().cpu()),
        "non_air_frac": float((hu > -999.0).float().mean().detach().cpu()),
    }
    if hu.numel() > 0:
        out.update(
            {
                "min": float(hu.min().detach().cpu()),
                "max": float(hu.max().detach().cpu()),
                "mean": float(hu.mean().detach().cpu()),
                "std": float(hu.std(unbiased=False).detach().cpu()),
            }
        )
    return out


def print_hu_stats(prefix: str, **tensors: torch.Tensor | None) -> None:
    """Print one compact JSON stats line per CT tensor in HU units."""
    for name, tensor in tensors.items():
        print(f"[debug:{prefix}] {json.dumps(hu_stats(name, tensor), sort_keys=True)}", flush=True)


def assert_finite_tensor(name: str, tensor: torch.Tensor | None) -> None:
    """Fail fast on NaN/Inf tensors."""
    if tensor is None:
        return
    if not bool(torch.isfinite(tensor.float()).all().detach().cpu()):
        raise FloatingPointError(f"{name} contains NaN or Inf; stats={tensor_stats(name, tensor)}")


def _center_slice_3d(tensor_bcdhw: torch.Tensor) -> np.ndarray:
    """Extract B0/C0 axial center from expected [B,C,D,H,W] tensors."""
    if tensor_bcdhw.ndim != 5:
        raise ValueError(f"expected [B,C,D,H,W], got {tuple(tensor_bcdhw.shape)}")
    array = tensor_bcdhw.detach().float()[0, 0].cpu().numpy()
    return array[array.shape[0] // 2]


def _window_hu(normalized_image: np.ndarray, *, window: float, level: float) -> tuple[np.ndarray, float, float]:
    hu = normalized_image * 1000.0
    vmin = level - window / 2.0
    vmax = level + window / 2.0
    return hu, vmin, vmax


def render_smoke_debug_png(
    *,
    out_dir: Path,
    run_name: str,
    step: int,
    volume_id: str,
    input_data: torch.Tensor,
    model_input: torch.Tensor,
    decoded: torch.Tensor,
    importance_weights: torch.Tensor | None,
    diff_window: float = 500.0,
) -> list[Path]:
    """Write HU-windowed central-slice PNGs to ./tmp for interactive inspection."""
    import os

    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    windows = [
        ("w1000_l0", 1000.0, 0.0),
        ("w400_l40", 400.0, 40.0),
    ]
    paths = []
    for tag, window, level in windows:
        input_hu, vmin, vmax = _window_hu(_center_slice_3d(input_data), window=window, level=level)
        model_input_hu, _, _ = _window_hu(_center_slice_3d(model_input), window=window, level=level)
        decoded_hu, _, _ = _window_hu(np.clip(_center_slice_3d(decoded), -1.0, 1.0), window=window, level=level)
        diff_hu = decoded_hu - input_hu
        diff_v = diff_window / 2.0
        panels = [
            ("input HU", input_hu, "gray", vmin, vmax),
            ("model input HU", model_input_hu, "gray", vmin, vmax),
            ("decoded HU", decoded_hu, "gray", vmin, vmax),
            ("decoded-input HU", diff_hu, "coolwarm", -diff_v, diff_v),
        ]
        if importance_weights is not None:
            panels.append(("importance", _center_slice_3d(importance_weights), "magma", None, None))

        cols = len(panels)
        fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4), constrained_layout=True)
        if cols == 1:
            axes = [axes]
        for ax, (title, image, cmap, vmin_i, vmax_i) in zip(axes, panels):
            ax.imshow(image, cmap=cmap, vmin=vmin_i, vmax=vmax_i)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"{run_name} step={step} volume={volume_id} window={window:g} level={level:g}")
        path = out_dir / f"{run_name}_step{step:04d}_{volume_id}_{tag}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def _center_slice_3d_plain(tensor_dhw: torch.Tensor | np.ndarray) -> np.ndarray:
    array = tensor_dhw.detach().float().cpu().numpy() if isinstance(tensor_dhw, torch.Tensor) else np.asarray(tensor_dhw)
    if array.ndim != 3:
        raise ValueError(f"expected [D,H,W], got {array.shape}")
    return array[array.shape[0] // 2]


def _coronal_slice_3d_plain(tensor_dhw: torch.Tensor | np.ndarray) -> np.ndarray:
    array = tensor_dhw.detach().float().cpu().numpy() if isinstance(tensor_dhw, torch.Tensor) else np.asarray(tensor_dhw)
    if array.ndim != 3:
        raise ValueError(f"expected [D,H,W], got {array.shape}")
    return array[:, array.shape[1] // 2, :]


def _sagittal_slice_3d_plain(tensor_dhw: torch.Tensor | np.ndarray) -> np.ndarray:
    array = tensor_dhw.detach().float().cpu().numpy() if isinstance(tensor_dhw, torch.Tensor) else np.asarray(tensor_dhw)
    if array.ndim != 3:
        raise ValueError(f"expected [D,H,W], got {array.shape}")
    return array[:, :, array.shape[2] // 2]


def _three_views(tensor_dhw: torch.Tensor | np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("axial", np.rot90(_center_slice_3d_plain(tensor_dhw), k=-1)),
        ("sagittal", np.flipud(_sagittal_slice_3d_plain(tensor_dhw))),
        ("coronal", np.flipud(_coronal_slice_3d_plain(tensor_dhw))),
    ]


def _mask_title(prefix: str, pair: dict[str, Any], index: int) -> str:
    organ = str(pair.get("organ_group", ""))
    mask = str(pair.get("mask_name", ""))
    matched = str(pair.get("matched_key", ""))
    row = pair.get("embedding_row", "")
    return f"{prefix}{index} {organ}/{mask}->{matched} row={row}"


def text_pair_debug_records(batch: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    volume_ids = [str(x) for x in batch.get("volume_id", [])]
    for batch_index, pairs in enumerate(batch.get("text", []) or []):
        volume_id = volume_ids[batch_index] if batch_index < len(volume_ids) else ""
        for pair_index, pair in enumerate(pairs):
            records.append(
                {
                    "batch_index": batch_index,
                    "pair_index": pair_index,
                    "volume_id": volume_id,
                    "source_name": str(pair.get("source_name", "")),
                    "source_index": int(pair.get("source_index", -1)),
                    "embedding_row": int(pair.get("embedding_row", -1)),
                    "organ_group": str(pair.get("organ_group", "")),
                    "mask_name": str(pair.get("mask_name", "")),
                    "matched_key": str(pair.get("matched_key", "")),
                    "text_hash": str(pair.get("text_hash", "")),
                    "text": str(pair.get("text", "")),
                    "has_text": bool(str(pair.get("text", ""))),
                    "has_mask": pair.get("mask") is not None,
                    "embedding_shape": list(pair["embedding"].shape) if pair.get("embedding") is not None else None,
                    "mask_shape": list(pair["mask"].shape) if pair.get("mask") is not None else None,
                    "index_row": pair.get("index_row", {}),
                }
            )
    return records


def export_loaded_batch_debug_json(*, out_dir: Path, run_name: str, step: int, batch: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    disease_summary = []
    volume_ids = [str(x) for x in batch.get("volume_id", [])]
    for batch_index, disease in enumerate(batch.get("disease", []) or []):
        labels = disease.get("labels")
        masks = disease.get("masks", {}) or {}
        disease_summary.append(
            {
                "batch_index": batch_index,
                "volume_id": volume_ids[batch_index] if batch_index < len(volume_ids) else "",
                "row_found": bool(disease.get("row_found", False)),
                "labels_shape": list(labels.shape) if labels is not None else None,
                "mask_names": sorted(str(k) for k in masks.keys()),
            }
        )
    payload = {
        "run_name": run_name,
        "step": int(step),
        "volume_ids": volume_ids,
        "ct_shape": list(batch["ct"].shape) if batch.get("ct") is not None else None,
        "reconstruction_weight_shape": (
            list(batch["reconstruction_weight"].shape) if batch.get("reconstruction_weight") is not None else None
        ),
        "debug_timing": batch.get("debug_timing", []),
        "text_pairs": text_pair_debug_records(batch),
        "disease": disease_summary,
    }
    path = out_dir / f"{run_name}_step{step:04d}_loaded_batch_debug.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _render_panel_grid(
    *,
    panels: list[tuple[str, np.ndarray | str, str, float | None, float | None]],
    title: str,
    path: Path,
    max_cols: int = 3,
) -> Path:
    import os

    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = Path(__file__).resolve().parents[3] / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_props = font_manager.FontProperties(fname=str(font_path))
    else:
        font_props = font_manager.FontProperties(family=["DejaVu Sans"])

    cols = min(max_cols, max(1, len(panels)))
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.2 * rows), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for ax, (panel_title, image, cmap, lo, hi) in zip(axes_flat, panels):
        ax.axis("off")
        if cmap == "text":
            ax.text(
                0.02,
                0.98,
                str(image),
                va="top",
                ha="left",
                fontsize=14,
                wrap=True,
                transform=ax.transAxes,
                fontproperties=font_props,
            )
        elif cmap == "rgb":
            ax.imshow(image)
            ax.set_title(panel_title, fontsize=10, fontproperties=font_props)
        else:
            ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(panel_title, fontsize=10, fontproperties=font_props)
    for ax in axes_flat[len(panels) :]:
        ax.axis("off")
    fig.suptitle(title, fontproperties=font_props)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _to_numpy_3d(tensor_dhw: torch.Tensor | np.ndarray) -> np.ndarray:
    array = tensor_dhw.detach().float().cpu().numpy() if isinstance(tensor_dhw, torch.Tensor) else np.asarray(tensor_dhw, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected [D,H,W], got {array.shape}")
    return array.astype(np.float32, copy=False)


def _resize_2d_to_shape(image: np.ndarray, target_shape: tuple[int, int], *, mode: str = "nearest") -> np.ndarray:
    tensor = torch.as_tensor(np.asarray(image, dtype=np.float32))[None, None]
    if mode == "nearest":
        resized = F.interpolate(tensor, size=target_shape, mode="nearest")[0, 0]
    elif mode == "bilinear":
        resized = F.interpolate(tensor, size=target_shape, mode="bilinear", align_corners=False)[0, 0]
    else:
        raise ValueError(f"unsupported resize mode={mode!r}")
    return resized.cpu().numpy().astype(np.float32, copy=False)


def _mask_center_dhw(mask: torch.Tensor | np.ndarray) -> tuple[int, int, int]:
    array = _to_numpy_3d(mask)
    coords = np.argwhere(array > 1e-6)
    if coords.size == 0:
        return tuple(int(dim // 2) for dim in array.shape)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    center = (lo + hi) // 2
    return tuple(int(x) for x in center)


def _map_center_to_ct_grid(
    center_dhw: tuple[int, int, int],
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    mapped = []
    for center, source_dim, target_dim in zip(center_dhw, source_shape, target_shape, strict=True):
        # Map voxel centers between grids rather than mapping edge coordinates.
        value = int(round((float(center) + 0.5) * float(target_dim) / float(source_dim) - 0.5))
        mapped.append(int(np.clip(value, 0, target_dim - 1)))
    return tuple(mapped)


def _view_slice_at(tensor_dhw: torch.Tensor | np.ndarray, view_name: str, center_dhw: tuple[int, int, int]) -> np.ndarray:
    array = _to_numpy_3d(tensor_dhw)
    d_idx = int(np.clip(center_dhw[0], 0, array.shape[0] - 1))
    h_idx = int(np.clip(center_dhw[1], 0, array.shape[1] - 1))
    w_idx = int(np.clip(center_dhw[2], 0, array.shape[2] - 1))
    if view_name == "axial":
        return np.rot90(array[d_idx], k=-1)
    if view_name == "sagittal":
        return np.flipud(array[:, :, w_idx])
    if view_name == "coronal":
        return np.flipud(array[:, h_idx, :])
    raise ValueError(f"unknown view_name={view_name!r}")


def _overlay_mask_on_ct(
    ct_dhw: torch.Tensor | np.ndarray,
    mask_dhw: torch.Tensor | np.ndarray,
    *,
    view_name: str,
    center_dhw: tuple[int, int, int],
    color: tuple[float, float, float] = (1.0, 0.85, 0.0),
    window: float = 1000.0,
    level: float = 0.0,
) -> np.ndarray:
    """Overlay a DataLoader mask-grid slice onto a high-res CT slice for debug only.

    The mask tensor is not changed for training. Only this visualization nearest-
    resizes the 2D mask slice to the CT slice shape so spatial alignment can be
    inspected without inventing smooth high-resolution mask boundaries.
    """
    ct = _to_numpy_3d(ct_dhw)
    mask = _to_numpy_3d(mask_dhw)
    ct_center_dhw = _map_center_to_ct_grid(center_dhw, tuple(int(x) for x in mask.shape), tuple(int(x) for x in ct.shape))
    ct_slice = _view_slice_at(ct, view_name, ct_center_dhw)
    mask_slice = np.clip(_view_slice_at(mask, view_name, center_dhw), 0.0, 1.0)
    mask_slice = np.clip(_resize_2d_to_shape(mask_slice, tuple(int(x) for x in ct_slice.shape), mode="nearest"), 0.0, 1.0)
    ct_hu, vmin, vmax = _window_hu(ct_slice, window=window, level=level)
    gray = np.clip((ct_hu - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    overlay = np.zeros_like(rgb)
    overlay[..., 0] = float(color[0])
    overlay[..., 1] = float(color[1])
    overlay[..., 2] = float(color[2])
    alpha = 0.45 * mask_slice[..., None]
    return (rgb * (1.0 - alpha) + overlay * alpha).astype(np.float32, copy=False)


UNIFIED_MASK_COLORS: dict[str, tuple[float, float, float]] = {
    "airway": (0.0, 0.85, 1.0),
    "aorta": (1.0, 0.45, 0.05),
    "central_vessels": (1.0, 0.0, 0.85),
    "chest_wall_bone": (0.95, 0.95, 0.95),
    "extra_chest_soft_tissue": (1.0, 0.45, 0.75),
    "heart": (1.0, 0.05, 0.05),
    "hiatal": (1.0, 0.85, 0.0),
    "lung": (0.05, 0.75, 0.15),
    "mediastinum": (0.65, 0.25, 1.0),
    "mediastinum_proxy": (0.65, 0.25, 1.0),
    "pleura": (0.15, 0.35, 1.0),
    "pleura_proxy": (0.15, 0.35, 1.0),
    "thoracic_spine": (0.55, 0.45, 0.35),
    "upper_abdomen": (0.65, 0.45, 0.1),
}


FALLBACK_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.90, 0.20, 0.20),
    (0.20, 0.60, 1.00),
    (0.20, 0.80, 0.30),
    (0.95, 0.75, 0.10),
    (0.75, 0.35, 0.95),
    (0.10, 0.85, 0.85),
    (0.95, 0.45, 0.70),
    (0.70, 0.55, 0.30),
)


def _mask_color(key: Any) -> tuple[float, float, float]:
    name = str(key)
    if name in UNIFIED_MASK_COLORS:
        return UNIFIED_MASK_COLORS[name]
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return FALLBACK_COLORS[int(digest[:8], 16) % len(FALLBACK_COLORS)]


def _overlay_masks_on_ct(
    ct_dhw: torch.Tensor | np.ndarray,
    mask_items: list[tuple[str, torch.Tensor | np.ndarray]],
    *,
    view_name: str,
    center_dhw: tuple[int, int, int],
    window: float = 1000.0,
    level: float = 0.0,
) -> np.ndarray:
    ct = _to_numpy_3d(ct_dhw)
    normalized_masks = [(key, _to_numpy_3d(mask)) for key, mask in mask_items]
    mask_shape = tuple(int(x) for x in normalized_masks[0][1].shape)
    ct_center_dhw = _map_center_to_ct_grid(center_dhw, mask_shape, tuple(int(x) for x in ct.shape))
    ct_slice = _view_slice_at(ct, view_name, ct_center_dhw)
    ct_hu, vmin, vmax = _window_hu(ct_slice, window=window, level=level)
    gray = np.clip((ct_hu - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    for key, mask in normalized_masks:
        mask_slice = np.clip(_view_slice_at(mask, view_name, center_dhw), 0.0, 1.0)
        mask_slice = np.clip(_resize_2d_to_shape(mask_slice, tuple(int(x) for x in ct_slice.shape), mode="nearest"), 0.0, 1.0)
        color = np.asarray(_mask_color(key), dtype=np.float32)
        overlay = np.broadcast_to(color, rgb.shape)
        alpha = 0.42 * mask_slice[..., None]
        rgb = rgb * (1.0 - alpha) + overlay * alpha
    return rgb.astype(np.float32, copy=False)


def _union_masks(masks: list[torch.Tensor | np.ndarray]) -> torch.Tensor | None:
    valid = [torch.as_tensor(_to_numpy_3d(mask), dtype=torch.float32) for mask in masks if mask is not None]
    if not valid:
        return None
    return torch.stack(valid, dim=0).amax(dim=0)


def _overlay_row_panels(
    ct_dhw: torch.Tensor | np.ndarray,
    mask_dhw: torch.Tensor | np.ndarray,
    *,
    prefix: str,
    color_key: Any | None = None,
) -> list[tuple[str, np.ndarray | str, str, float | None, float | None]]:
    mask = torch.as_tensor(_to_numpy_3d(mask_dhw), dtype=torch.float32)
    mask_grid = tuple(int(x) for x in mask.shape)
    ct_grid = tuple(int(x) for x in _to_numpy_3d(ct_dhw).shape)
    center_dhw = _mask_center_dhw(mask)
    ct_center_dhw = _map_center_to_ct_grid(center_dhw, mask_grid, ct_grid)
    panels: list[tuple[str, np.ndarray | str, str, float | None, float | None]] = []
    for view_name in ("axial", "sagittal", "coronal"):
        image = _overlay_mask_on_ct(
            ct_dhw,
            mask,
            view_name=view_name,
            center_dhw=center_dhw,
            color=_mask_color(prefix if color_key is None else color_key),
        )
        panels.append(
            (
                f"{view_name}\nmask={center_dhw} ct={ct_center_dhw}",
                image,
                "rgb",
                None,
                None,
            )
        )
    return panels


def _overlay_multi_mask_row_panels(
    ct_dhw: torch.Tensor | np.ndarray,
    mask_items: list[tuple[str, torch.Tensor | np.ndarray]],
) -> list[tuple[str, np.ndarray | str, str, float | None, float | None]]:
    if not mask_items:
        return []
    masks = [torch.as_tensor(_to_numpy_3d(mask), dtype=torch.float32) for _, mask in mask_items]
    mask_shape = tuple(int(x) for x in masks[0].shape)
    for mask in masks:
        if tuple(mask.shape) != mask_shape:
            raise ValueError(f"global mask shapes are inconsistent: {tuple(mask.shape)} != {mask_shape}")
    ct_grid = tuple(int(x) for x in _to_numpy_3d(ct_dhw).shape)
    union = torch.stack(masks, dim=0).amax(dim=0)
    center_dhw = _mask_center_dhw(union)
    ct_center_dhw = _map_center_to_ct_grid(center_dhw, mask_shape, ct_grid)
    normalized_items = [(key, mask) for (key, _), mask in zip(mask_items, masks)]
    panels: list[tuple[str, np.ndarray | str, str, float | None, float | None]] = []
    for view_name in ("axial", "sagittal", "coronal"):
        image = _overlay_masks_on_ct(ct_dhw, normalized_items, view_name=view_name, center_dhw=center_dhw)
        panels.append(
            (
                f"{view_name}\nmask={center_dhw} ct={ct_center_dhw}",
                image,
                "rgb",
                None,
                None,
            )
        )
    return panels


def _ct_view_panels(ct: torch.Tensor | np.ndarray, *, prefix: str = "CT") -> list[tuple[str, np.ndarray | str, str, float | None, float | None]]:
    panels: list[tuple[str, np.ndarray | str, str, float | None, float | None]] = []
    for view_name, image in _three_views(ct):
        ct_hu, vmin, vmax = _window_hu(image, window=1000.0, level=0.0)
        panels.append((f"{prefix} {view_name}", ct_hu, "gray", vmin, vmax))
    return panels


def _mask_row_panels(
    mask: torch.Tensor | np.ndarray,
    *,
    prefix: str,
    cmap: str = "viridis",
) -> list[tuple[str, np.ndarray | str, str, float | None, float | None]]:
    return [(f"{prefix} {view_name}", image, cmap, 0.0, 1.0) for view_name, image in _three_views(mask)]


RADGENOME_MASK_ZH: dict[str, str] = {
    "abdomen": "腹部",
    "bone": "骨",
    "breast": "乳腺/胸壁软组织",
    "bronchie": "支气管",
    "esophagus": "食管",
    "heart": "心脏",
    "lung": "肺",
    "mediastinal tissue": "纵隔组织",
    "mediastinum": "纵隔",
    "pleura": "胸膜",
    "thyroid": "甲状腺",
    "trachea": "气管",
    "trachea and bronchie": "气管和支气管",
}


ANATOMY_KEY_ZH: dict[str, str] = {
    **RADGENOME_MASK_ZH,
    "airway": "气道",
    "aorta": "主动脉",
    "central_vessels": "中央血管",
    "chest_wall_bone": "胸壁/骨",
    "extra_chest_soft_tissue": "胸外软组织",
    "hiatal": "食管裂孔/膈肌附近",
    "mediastinum_proxy": "纵隔代理区域",
    "pleura_proxy": "胸膜代理区域",
    "thoracic_spine": "胸椎",
    "upper_abdomen": "上腹部",
}


def _mask_label_with_translation(mask_name: Any) -> str:
    name = str(mask_name)
    translated = RADGENOME_MASK_ZH.get(name)
    return f"{name} ({translated})" if translated else name


def _anatomy_key_with_translation(key: Any) -> str:
    name = str(key)
    translated = ANATOMY_KEY_ZH.get(name)
    return f"{name} ({translated})" if translated else name


def _text_pair_panel(pair: dict[str, Any], pair_index: int) -> tuple[str, str, str, None, None]:
    text = str(pair.get("text", ""))
    if not text:
        index_row = pair.get("index_row", {}) or {}
        for key in ("text", "sentence", "prompt", "finding", "report_text"):
            if str(index_row.get(key, "")).strip():
                text = str(index_row[key])
                break
    wrapped = textwrap.fill(text or "<missing text>", width=44)
    body = (
        f"text{pair_index}\n\n"
        f"TEXT:\n{wrapped}\n\n"
        "MASK/META:\n"
        f"source={pair.get('source_name', '')}\n"
        f"organ={_anatomy_key_with_translation(pair.get('organ_group', ''))}\n"
        f"mask={_mask_label_with_translation(pair.get('mask_name', ''))}\n"
        f"matched={_anatomy_key_with_translation(pair.get('matched_key', ''))}\n"
        "overlay nearest-resizes mask to CT slice for visualization only\n"
    )
    return ("text", body, "text", None, None)


def _disease_mask_panel(mask_name: str) -> tuple[str, str, str, None, None]:
    body = (
        f"disease mask={_anatomy_key_with_translation(mask_name)}\n"
        "overlay nearest-resizes mask to CT slice for visualization only\n"
        "colors use the shared mask color map"
    )
    return ("disease", body, "text", None, None)


def render_loaded_batch_debug_png(*, out_dir: Path, run_name: str, step: int, batch: dict[str, Any]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    volume_id = str(batch["volume_id"][0])
    ct = batch["ct"][0, 0]
    weight = batch.get("reconstruction_weight")
    text_pairs = batch.get("text", [[]])[0] if batch.get("text") else []
    text_global_masks = batch.get("text_global_masks", [[]])[0] if batch.get("text_global_masks") else []
    disease = batch.get("disease", [{}])[0] if batch.get("disease") else {}
    disease_masks = disease.get("masks", {}) or {}

    paths = []
    windows = [
        ("w1000_l0", 1000.0, 0.0),
        ("w400_l40", 400.0, 40.0),
    ]
    panels: list[tuple[str, np.ndarray, str, float | None, float | None]] = []
    weight_views = dict(_three_views(weight[0])) if weight is not None else {}
    for view_name, image in _three_views(ct):
        for tag, window, level in windows:
            ct_hu, vmin, vmax = _window_hu(image, window=window, level=level)
            panels.append((f"ct {view_name} {tag}", ct_hu, "gray", vmin, vmax))
        if view_name in weight_views:
            panels.append((f"token weight {view_name}", weight_views[view_name], "magma", None, None))
    path = out_dir / f"{run_name}_step{step:04d}_{volume_id}_loaded_input.png"
    paths.append(
        _render_panel_grid(
            panels=panels,
            title=f"loaded input step={step} volume={volume_id}",
            path=path,
            max_cols=3,
        )
    )

    text_global = text_global_masks[0] if text_global_masks else {}
    text_masks = [pair.get("mask") for pair in text_pairs if pair.get("mask") is not None]
    text_global_items: list[tuple[str, torch.Tensor | np.ndarray]] = []
    if text_global and text_global.get("masks") is not None:
        global_masks = text_global["masks"]
        global_groups = [str(x) for x in text_global.get("mask_groups", [])]
        global_names = [str(x) for x in text_global.get("mask_names", [])]
        for mask_index in range(int(global_masks.shape[0])):
            key = global_groups[mask_index] if mask_index < len(global_groups) else global_names[mask_index]
            text_global_items.append((key, global_masks[mask_index]))
    if not text_global_items:
        text_global_items = [
            (str(pair.get("matched_key") or pair.get("organ_group") or pair.get("mask_name")), pair["mask"])
            for pair in text_pairs
            if pair.get("mask") is not None
        ]
    text_panels: list[tuple[str, np.ndarray | str, str, float | None, float | None]] = []
    if text_global_items:
        global_source = str(text_global.get("source_name", "radgenome")) if text_global else "selected text masks"
        global_count = int(text_global.get("mask_count", len(text_masks))) if text_global else len(text_masks)
        text_panels.extend(
            [
                (
                    "global",
                    "global RadGenome masks\n"
                    f"source={global_source}\n"
                    f"num_global_masks={global_count}\n"
                    f"num_selected_text_masks={len(text_masks)}\n"
                    "mask grid is loaded by DataLoader\n"
                    "overlay nearest-resizes mask to CT slice for visualization only\n"
                    "colors by organ/matched key",
                    "text",
                    None,
                    None,
                ),
                *_overlay_multi_mask_row_panels(ct, text_global_items),
            ]
        )
    for pair_index, pair in enumerate(text_pairs):
        mask = pair.get("mask")
        if mask is None:
            continue
        title = _mask_title("text", pair, pair_index)
        text_panels.extend(
            [
                _text_pair_panel(pair, pair_index),
                *_overlay_row_panels(ct, mask, prefix=title, color_key=pair.get("matched_key") or pair.get("organ_group")),
            ]
        )
    if text_panels:
        path = out_dir / f"{run_name}_step{step:04d}_{volume_id}_text_masks_3view.png"
        paths.append(
            _render_panel_grid(
                panels=text_panels,
                title=f"selected text masks step={step} volume={volume_id}",
                path=path,
                max_cols=4,
            )
        )

    disease_items = [(str(mask_name), mask) for mask_name, mask in disease_masks.items() if mask is not None]
    disease_panels: list[tuple[str, np.ndarray | str, str, float | None, float | None]] = []
    if disease_items:
        disease_panels.extend(
            [
                (
                    "global",
                    "global TS masks\n"
                    f"num_disease_masks={len(disease_items)}\n"
                    "mask grid is loaded by DataLoader\n"
                    "overlay nearest-resizes mask to CT slice for visualization only\n"
                    "colors by TS mask name / 中文解剖名",
                    "text",
                    None,
                    None,
                ),
                *_overlay_multi_mask_row_panels(ct, disease_items),
            ]
        )
    for mask_name, mask in disease_items:
        disease_panels.extend(
            [
                _disease_mask_panel(mask_name),
                *_overlay_row_panels(ct, mask, prefix=f"disease {mask_name}", color_key=mask_name),
            ]
        )
    if disease_panels:
        path = out_dir / f"{run_name}_step{step:04d}_{volume_id}_disease_masks_3view.png"
        paths.append(
            _render_panel_grid(
                panels=disease_panels,
                title=f"disease masks step={step} volume={volume_id}",
                path=path,
                max_cols=4,
            )
        )
    return paths


def render_input_conditioning_debug_png(
    *,
    out_dir: Path,
    run_name: str,
    step: int,
    volume_id: str,
    input_data: torch.Tensor,
    model_input: torch.Tensor,
    shaped_importance_weights: torch.Tensor | None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    delta = model_input.float() - input_data.float()
    windows = [
        ("w1000_l0", 1000.0, 0.0),
        ("w400_l40", 400.0, 40.0),
    ]
    panels: list[tuple[str, np.ndarray, str, float | None, float | None]] = []
    input_views = dict(_three_views(input_data[0, 0]))
    model_views = dict(_three_views(model_input[0, 0]))
    delta_views = dict(_three_views(delta[0, 0]))
    weight_views = dict(_three_views(shaped_importance_weights[0, 0])) if shaped_importance_weights is not None else {}
    for view_name in ("axial", "sagittal", "coronal"):
        for tag, window, level in windows:
            image = input_views[view_name]
            hu, vmin, vmax = _window_hu(image, window=window, level=level)
            panels.append((f"input CT {view_name} {tag}", hu, "gray", vmin, vmax))
        for tag, window, level in windows:
            image = model_views[view_name]
            hu, vmin, vmax = _window_hu(image, window=window, level=level)
            panels.append((f"model input {view_name} {tag}", hu, "gray", vmin, vmax))
        delta_hu = delta_views[view_name] * 1000.0
        panels.append((f"input delta {view_name} HU", delta_hu, "coolwarm", -250.0, 250.0))
        if view_name in weight_views:
            panels.append((f"input mask/weight {view_name}", weight_views[view_name], "magma", None, None))
    path = out_dir / f"{run_name}_step{step:04d}_{volume_id}_input_conditioning.png"
    paths.append(
        _render_panel_grid(
            panels=panels,
            title=f"input conditioning step={step} volume={volume_id}",
            path=path,
            max_cols=6,
        )
    )
    return paths

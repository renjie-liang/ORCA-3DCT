"""Materialize report-generation visual embeddings from token artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dtbd3d.core.artifact import load_token_row
from dtbd3d.core.btb3d_model import TOKEN_LAYOUTS
from dtbd3d.core.token_codec import CODEBOOK_DIM, LFQConvention, merge_8x8_reportgen, unpack_lfq_codes

REPORTGEN_CHANNEL_ORDER = "msb_reverse_channels"


@dataclass(frozen=True)
class ReportgenArtifactPaths:
    """Resolved paths from a final_eval reportgen artifact manifest."""

    manifest_path: Path
    reportgen_dir: Path
    codebook_path: Path
    codebook_metadata_path: Path
    token_dir: Path
    ids_path: Path
    tokens_int_path: Path
    manifest: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _resolve_relative(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def resolve_reportgen_artifact_manifest(manifest_path: str | Path, split: str) -> ReportgenArtifactPaths:
    """Resolve codebook and split token paths from `reportgen_artifact/manifest.json`."""

    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    reportgen_dir = manifest_path.parent
    manifest = _read_json(manifest_path)

    token_artifacts = manifest.get("token_artifacts")
    if not isinstance(token_artifacts, dict) or split not in token_artifacts:
        available = sorted(token_artifacts) if isinstance(token_artifacts, dict) else []
        raise KeyError(f"split {split!r} not found in {manifest_path}; available={available}")
    split_entry = token_artifacts[split]
    if not isinstance(split_entry, dict):
        raise TypeError(f"token_artifacts[{split!r}] must be an object")

    codebook_dir_value = manifest.get("codebook_artifact_dir")
    if not codebook_dir_value:
        raise ValueError(f"{manifest_path} does not declare codebook_artifact_dir")
    codebook_dir = _resolve_relative(reportgen_dir, codebook_dir_value)

    token_dir = _resolve_relative(reportgen_dir, split_entry.get("dir", Path(split_entry["tokens_int"]).parent))
    ids_path = _resolve_relative(reportgen_dir, split_entry["ids"])
    tokens_int_path = _resolve_relative(reportgen_dir, split_entry["tokens_int"])
    return ReportgenArtifactPaths(
        manifest_path=manifest_path,
        reportgen_dir=reportgen_dir,
        codebook_path=codebook_dir / "codebook.npy",
        codebook_metadata_path=codebook_dir / "metadata.json",
        token_dir=token_dir,
        ids_path=ids_path,
        tokens_int_path=tokens_int_path,
        manifest=manifest,
    )


def validate_reportgen_codebook_metadata(metadata: dict[str, Any], metadata_path: Path) -> None:
    """Fail fast when a codebook artifact is not in BTB3D reportgen channel order."""

    codebook_order = metadata.get("codebook_channel_order")
    expected_order = metadata.get("reportgen_expected_channel_order")
    requires_reverse = metadata.get("requires_channel_reverse_for_btb3d_pretrained_reportgen")
    if codebook_order != REPORTGEN_CHANNEL_ORDER:
        raise ValueError(
            f"{metadata_path} codebook_channel_order={codebook_order!r}; expected "
            f"{REPORTGEN_CHANNEL_ORDER!r}. Re-run the final eval bundle with the current exporter."
        )
    if expected_order not in (None, REPORTGEN_CHANNEL_ORDER):
        raise ValueError(
            f"{metadata_path} reportgen_expected_channel_order={expected_order!r}; expected "
            f"{REPORTGEN_CHANNEL_ORDER!r}"
        )
    if requires_reverse is not False:
        raise ValueError(
            f"{metadata_path} requires_channel_reverse_for_btb3d_pretrained_reportgen={requires_reverse!r}; "
            "expected false for direct reportgen lookup."
        )


def load_reportgen_codebook(
    codebook_path: str | Path,
    metadata_path: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a reportgen-ready codebook artifact.

    The returned array may be a memory-mapped ndarray. It is safe to index as
    `codebook[token_ids]` when metadata passes the channel-order checks.
    """

    codebook_path = Path(codebook_path)
    metadata_path = Path(metadata_path)
    if not codebook_path.exists():
        raise FileNotFoundError(codebook_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = _read_json(metadata_path)
    validate_reportgen_codebook_metadata(metadata, metadata_path)
    codebook = np.load(codebook_path, mmap_mode="r")
    expected_rows = 1 << CODEBOOK_DIM
    if codebook.shape != (expected_rows, CODEBOOK_DIM):
        raise ValueError(f"{codebook_path} shape {codebook.shape}; expected {(expected_rows, CODEBOOK_DIM)}")
    if not np.issubdtype(codebook.dtype, np.floating):
        raise TypeError(f"{codebook_path} dtype {codebook.dtype}; expected floating")
    return codebook, metadata


def materialize_reportgen_features_from_codes(
    codes: np.ndarray,
    compression: str,
    convention: LFQConvention = REPORTGEN_CHANNEL_ORDER,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Return BTB3D/LLaVA image features with shape `(1, T, H, W, C)`."""

    token_t, token_h, token_w = TOKEN_LAYOUTS[compression]
    expected = token_t * token_h * token_w
    codes = np.asarray(codes, dtype=np.uint32)
    if codes.shape != (expected,):
        raise ValueError(f"token shape {codes.shape}; expected {(expected,)} for {compression}")

    if codebook is None:
        features = unpack_lfq_codes(codes, convention=convention, num_bits=CODEBOOK_DIM)
    else:
        features = np.asarray(codebook[codes.astype(np.int64, copy=False)])
        if features.shape != (expected, CODEBOOK_DIM):
            raise ValueError(f"codebook lookup shape {features.shape}; expected {(expected, CODEBOOK_DIM)}")

    arr = features.reshape(token_t, token_h, token_w, CODEBOOK_DIM)
    arr = arr.transpose(3, 0, 1, 2)[None].astype(np.float32, copy=False)
    if compression == "8x8x8":
        arr = merge_8x8_reportgen(arr).astype(np.float32, copy=False)
    return arr.transpose(0, 2, 3, 4, 1)


def materialize_reportgen_image(
    token_dir: str | Path,
    volume_id: str,
    compression: str,
    convention: LFQConvention,
    id_to_idx: dict[str, int] | None,
    matrix: np.ndarray | None,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Load one token row and materialize its report-generation image features."""

    codes = load_token_row(token_dir, volume_id, id_to_idx, matrix)
    return materialize_reportgen_features_from_codes(codes, compression, convention, codebook)

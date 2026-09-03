#!/usr/bin/env python3
"""
Extract BTB3D tokenizer-index token artifacts from raw CT volumes.

This is the source-of-truth artifact generator for downstream reproduction:

    raw CT -> BTB3D encoder/tokenizer -> quantized_output.indices

The output is task-neutral. Reconstruction can decode these packed integer
tokens directly. Report generation should materialize task-specific `.npz`
features from these same tokens instead of using a separate embedding source.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from dtbd3d.core.artifact import read_ids, save_token_row, update_ids, write_tokens_matrix
    from dtbd3d.core.btb3d_model import CONFIGS, TOKEN_LAYOUTS, expected_token_count, load_btb3d_tokenizer
    from dtbd3d.core.ct_preprocess import center_crop_axis2, preprocess_volume
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dtbd3d.core.artifact import read_ids, save_token_row, update_ids, write_tokens_matrix
    from dtbd3d.core.btb3d_model import CONFIGS, TOKEN_LAYOUTS, expected_token_count, load_btb3d_tokenizer
    from dtbd3d.core.ct_preprocess import center_crop_axis2, preprocess_volume


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    root = project_root()
    data_links = root / "Experiment/core_code/data_links"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compression", choices=sorted(CONFIGS), required=True)
    p.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more .nii.gz files and/or directories containing NIfTI volumes.",
    )
    p.add_argument("--out-dir", required=True, help="Token artifact directory.")
    p.add_argument("--btb3d-repo", default=str(root / "Experiment/core_code/btb3d_baseline/encoder-decoder"))
    p.add_argument("--data-root", default=str(data_links / "ct_rate"))
    p.add_argument("--weights-root", default=str(data_links / "btb3d_weights"))
    p.add_argument(
        "--tokenizer-checkpoint",
        default=None,
        help="Optional tokenizer state checkpoint, e.g. tokenizer_step_2048.safetensors.",
    )
    p.add_argument("--metadata-csv", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--write-matrix",
        action="store_true",
        help="Also build tokens_int.npy from per-volume token files after extraction.",
    )
    p.add_argument("--verbose", action="store_true", help="Print per-volume write/skip details.")
    return p.parse_args()


def volume_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def iter_inputs(input_paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in input_paths:
        if input_path.is_dir():
            files.extend(sorted(input_path.rglob("*.nii.gz")))
            files.extend(sorted(input_path.rglob("*.nii")))
        else:
            if not input_path.exists():
                raise FileNotFoundError(f"Input path does not exist: {input_path}")
            files.append(input_path)

    dedup: list[Path] = []
    seen: set[str] = set()
    for path in files:
        if not (path.name.endswith(".nii.gz") or path.name.endswith(".nii")):
            raise ValueError(f"Input path is not a NIfTI file: {path}")
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(path)
    return dedup


@contextlib.contextmanager
def suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    tokens_dir = out_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    nii_files = iter_inputs([Path(x) for x in args.input])
    if args.limit is not None:
        nii_files = nii_files[: args.limit]
    if not nii_files:
        raise SystemExit("No input NIfTI files found")

    metadata_csv = (
        Path(args.metadata_csv)
        if args.metadata_csv
        else Path(args.data_root) / "dataset/metadata/validation_metadata.csv"
    )
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    metadata_df = pd.read_csv(metadata_csv)
    expected_tokens = expected_token_count(args.compression)

    print(f"Processing {len(nii_files)} NIfTI volume(s)")
    print(f"Compression: {args.compression}; expected tokens per volume: {expected_tokens}")
    print(f"Device: {args.device}")

    model = load_btb3d_tokenizer(
        args.compression,
        args.btb3d_repo,
        args.weights_root,
        args.device,
        tokenizer_checkpoint=args.tokenizer_checkpoint,
    )

    written_ids: list[str] = []
    with torch.no_grad():
        progress = tqdm(nii_files, desc=f"extract {args.compression}", unit="vol")
        for nii_path in progress:
            vol_id = volume_id_from_path(nii_path)
            progress.set_postfix_str(vol_id)
            out_path = tokens_dir / f"{vol_id}.npy"
            if out_path.exists() and not args.overwrite:
                if args.verbose:
                    tqdm.write(f"Skipping existing {out_path}")
                written_ids.append(vol_id)
                continue

            with suppress_output(not args.verbose):
                inp, _ = preprocess_volume(str(nii_path), metadata_df)
                inp = center_crop_axis2(inp).to(args.device).to(torch.bfloat16)
                _, quantized_output, _ = model.tokenizer.encode(
                    inp,
                    entropy_loss_weight=0.0,
                    calculate_quantize_loss=False,
                )
            tokens = quantized_output.indices.detach().cpu().numpy().reshape(-1).astype(np.uint32)
            if tokens.shape != (expected_tokens,):
                raise ValueError(f"{vol_id} token shape {tokens.shape}; expected {(expected_tokens,)}")
            save_token_row(out_dir, vol_id, tokens)
            written_ids.append(vol_id)
            unique_codes = int(np.unique(tokens).size)
            progress.set_postfix_str(f"{vol_id} unique_codes={unique_codes}")
            if args.verbose:
                tqdm.write(f"Wrote {out_path} unique_codes={unique_codes}")

    update_ids(out_dir / "ids.txt", written_ids)
    if args.write_matrix:
        matrix_path = write_tokens_matrix(out_dir, expected_tokens)
        print(f"Wrote {matrix_path} shape=({len(read_ids(out_dir / 'ids.txt'))}, {expected_tokens})")
    print(f"Token artifact directory: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

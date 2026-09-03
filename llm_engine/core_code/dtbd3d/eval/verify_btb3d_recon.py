#!/usr/bin/env python
"""
verify_btb3d_recon.py — V1 reconstruction sanity check.

Loads pre-computed post-VQ tokens, decodes them with the BTB3D decoder,
and compares the reconstruction against the preprocessed input volume.

Adapted from 3DCT_Encoder/adapters/verify_btb3d_recon.py with:
- Hard-coded paths replaced by --paths arguments (defaulting to data_links/)
- Compression rate (8x8x8 vs 16x16x8) selectable via --compression
- Per-volume CSV optionally written to --out (default: stdout summary only)
"""
import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    from dtbd3d.core.artifact import load_token_row, open_matrix, read_ids
    from dtbd3d.core.ct_preprocess import center_crop_axis2, preprocess_volume
    from dtbd3d.core.token_codec import unpack_lfq_codes
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dtbd3d.core.artifact import load_token_row, open_matrix, read_ids
    from dtbd3d.core.ct_preprocess import center_crop_axis2, preprocess_volume
    from dtbd3d.core.token_codec import unpack_lfq_codes

# Token spatial layouts on 512x512x241 input
TOKEN_LAYOUTS = {
    "16x16x8": (31, 32, 32),    #  31 * 32 * 32 = 31744 tokens, 18-dim each
    "8x8x8":   (31, 64, 64),    #  31 * 64 * 64 = 126976 tokens, 18-dim each (or 72 after merge)
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parents[2] / "data_links"
    p.add_argument("--btb3d-repo", default=str(Path(__file__).resolve().parents[2] / "btb3d_baseline" / "encoder-decoder"))
    p.add_argument("--data-root",  default=str(here / "ct_rate"))
    p.add_argument("--weights-root", default=str(here / "btb3d_weights"))
    p.add_argument(
        "--token-dir",
        dest="token_dir",
        default=None,
        help="Token artifact directory containing ids.txt, tokens/*.npy and optionally tokens_int.npy.",
    )
    p.add_argument("--canonical-dir", dest="token_dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--compression", choices=["16x16x8", "8x8x8"], required=True)
    p.add_argument("--split", default="valid", choices=["valid", "train"])
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bit-order", choices=["lsb", "msb"], default="lsb")
    p.add_argument("--reverse-channels", action="store_true")
    p.add_argument("--out", default=None, help="Optional per-volume CSV output path")
    p.add_argument(
        "--save-recon-dir",
        default=None,
        help=(
            "Optional directory for visual inspection artifacts. Saves "
            "preprocessed input/reconstruction NIfTI files and compact NPZ "
            "metadata for each processed volume. Use with a small --n-samples."
        ),
    )
    p.add_argument(
        "--save-recon-full",
        action="store_true",
        help=(
            "Also save full preprocessed input/reconstruction arrays as compressed "
            "float16 NPZ. NIfTI files are already saved when --save-recon-dir is set; "
            "this extra NPZ can be large and is mainly for numpy-based debugging."
        ),
    )
    p.add_argument(
        "--exclude-csv",
        action="append",
        default=[],
        help="CSV(s) with a volume_id column to exclude from sampling, useful for resume.",
    )
    return p.parse_args()


def convention_from_flags(bit_order: str, reverse_channels: bool) -> str:
    return f"{bit_order}_{'reverse_channels' if reverse_channels else 'identity'}"


def save_recon_artifacts(
    out_dir,
    vol_id,
    inp_np,
    recon_np,
    inp_roi,
    recon_roi,
    ssim,
    psnr,
    mse,
    save_full=False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    affine = np.diag([1.5, 0.75, 0.75, 1.0]).astype(np.float32)
    nib.save(
        nib.Nifti1Image(inp_np.astype(np.float32), affine),
        out_dir / f"{vol_id}_input_preprocessed.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(recon_np.astype(np.float32), affine),
        out_dir / f"{vol_id}_recon_preprocessed.nii.gz",
    )
    nib.save(
        nib.Nifti1Image((recon_np - inp_np).astype(np.float32), affine),
        out_dir / f"{vol_id}_diff_preprocessed.nii.gz",
    )

    np.savez_compressed(
        out_dir / f"{vol_id}_recon_metadata.npz",
        volume_id=np.array(vol_id),
        data_range=np.array([-1.0, 1.0], dtype=np.float32),
        shape=np.array(inp_np.shape, dtype=np.int32),
        affine=affine,
        ssim=np.array(ssim, dtype=np.float32),
        psnr=np.array(psnr, dtype=np.float32),
        mse=np.array(mse, dtype=np.float32),
    )

    if save_full:
        np.savez_compressed(
            out_dir / f"{vol_id}_recon_full.npz",
            volume_id=np.array(vol_id),
            input=inp_np.astype(np.float16),
            recon=recon_np.astype(np.float16),
            input_roi=inp_roi.astype(np.float16),
            recon_roi=recon_roi.astype(np.float16),
            data_range=np.array([-1.0, 1.0], dtype=np.float32),
            ssim=np.array(ssim, dtype=np.float32),
            psnr=np.array(psnr, dtype=np.float32),
            mse=np.array(mse, dtype=np.float32),
        )


def main():
    args = parse_args()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.save_recon_dir:
        Path(args.save_recon_dir).mkdir(parents=True, exist_ok=True)

    if args.compression == "8x8x8":
        config_name = "magvit2_3d_model_config_8x8x8.yaml"
    else:
        config_name = "magvit2_3d_model_config.yaml"
    config_file = os.path.join(args.btb3d_repo, "configs", config_name)
    ckpt_file   = os.path.join(args.weights_root, f"encoder-decoder/{args.compression.replace('x','_')}/3rd_stage.ckpt")
    if not args.token_dir:
        raise SystemExit("Missing required argument: --token-dir")
    token_dir = args.token_dir
    volume_dir  = os.path.join(args.data_root, "dataset", f"{args.split}_fixed")
    meta_csv    = os.path.join(args.data_root, "dataset/metadata", f"{args.split}ation_metadata.csv" if args.split == "valid" else "train_metadata.csv")

    token_t, token_h, token_w = TOKEN_LAYOUTS[args.compression]
    expected_tokens = token_t * token_h * token_w

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"BTB3D config not found: {config_file}")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"BTB3D checkpoint not found: {ckpt_file}")
    if not os.path.isdir(volume_dir):
        raise FileNotFoundError(f"Volume directory not found: {volume_dir}")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {meta_csv}")
    metadata_df = pd.read_csv(meta_csv)
    ids = read_ids(Path(token_dir) / "ids.txt")
    if not ids:
        raise FileNotFoundError(f"No ids found in {Path(token_dir) / 'ids.txt'}")

    id_to_idx, tokens_int = open_matrix(token_dir)
    if tokens_int is not None:
        assert tokens_int.shape == (len(ids), expected_tokens), \
            f"tokens_int shape {tokens_int.shape} vs expected ({len(ids)}, {expected_tokens})"
    elif not (Path(token_dir) / "tokens").exists():
        raise FileNotFoundError(
            f"No tokens_int.npy and no tokens/ directory found in token artifact dir: {token_dir}"
        )
    print(f"Token source: {token_dir}")
    print(f"Unpack convention: bit_order={args.bit_order}, reverse_channels={args.reverse_channels}")

    sys.path.insert(0, args.btb3d_repo)
    from modeling.magvit_model import VisionTokenizer
    from src.utils import get_config

    model_config = get_config(config_file)
    model = VisionTokenizer(
        config=model_config,
        commitment_cost=0,
        diversity_gamma=0,
        use_gan=False,
        use_lecam_ema=False,
        use_perceptual=False,
    )
    states = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    model.tokenizer.load_state_dict(states, strict=True)
    model.eval()
    model.to(args.device).to(torch.bfloat16)
    print(f"Model loaded: {ckpt_file}")
    print(f"Token layout: {token_t}x{token_h}x{token_w} = {expected_tokens} tokens")

    resumed_df = None
    resumed_ids = set()
    if args.out and os.path.exists(args.out):
        resumed_df = pd.read_csv(args.out)
        if "volume_id" not in resumed_df.columns:
            raise ValueError(f"--out exists but lacks volume_id column: {args.out}")
        resumed_ids = set(str(v) for v in resumed_df["volume_id"].dropna())
        print(f"Resume: {len(resumed_ids)} volume(s) already in {args.out}, will be skipped")

    excluded_ids = set(resumed_ids)
    for csv_path in args.exclude_csv:
        df = pd.read_csv(csv_path)
        if "volume_id" not in df.columns:
            raise ValueError(f"--exclude-csv requires a volume_id column: {csv_path}")
        excluded_ids.update(str(v) for v in df["volume_id"].dropna())

    candidate_idx = [i for i, vol_id in enumerate(ids) if vol_id not in excluded_ids]
    if excluded_ids:
        print(
            f"Excluding {len(excluded_ids)} volume_id(s) from {len(ids)} total; "
            f"{len(candidate_idx)} candidate(s) remain"
        )

    remaining_target = max(0, args.n_samples - len(resumed_ids))
    sample_size = min(remaining_target, len(candidate_idx))
    rng = np.random.default_rng(args.seed)
    if sample_size > 0:
        sample_idx = rng.choice(
            candidate_idx,
            size=sample_size,
            replace=False,
        ).tolist()
    else:
        sample_idx = []
        print(f"Nothing new to run: target n_samples={args.n_samples} already met by {len(resumed_ids)} resumed rows.")

    results = []

    def write_results() -> None:
        if not args.out:
            return
        df = pd.DataFrame(
            results,
            columns=["volume_id", "bit_order", "reverse_channels", "ssim", "psnr", "mse"],
        )
        if resumed_df is not None and len(resumed_df) > 0:
            df = pd.concat([resumed_df, df], ignore_index=True)
        tmp_out = f"{args.out}.tmp"
        df.to_csv(tmp_out, index=False)
        os.replace(tmp_out, args.out)

    for ix in sample_idx:
        vol_id = ids[ix]
        # Unpack uint32 codes -> {-1, +1}^18 LFQ representation
        codes = load_token_row(token_dir, vol_id, id_to_idx, tokens_int)
        tokens = unpack_lfq_codes(
            codes,
            convention=convention_from_flags(args.bit_order, args.reverse_channels),
        )

        z_q = torch.tensor(tokens, dtype=torch.bfloat16)
        z_q = z_q.reshape(token_t, token_h, token_w, 18)
        z_q = z_q.permute(3, 0, 1, 2).unsqueeze(0).to(args.device)

        with torch.no_grad():
            recon = model.tokenizer.decode(z_q)
        recon_np = recon[0, 0].float().cpu().numpy()

        nii_path = None
        for dirpath, _, fnames in os.walk(volume_dir):
            if f"{vol_id}.nii.gz" in fnames:
                nii_path = os.path.join(dirpath, f"{vol_id}.nii.gz")
                break
        assert nii_path is not None, f"nii.gz not found for {vol_id}"

        inp, valid_slices = preprocess_volume(nii_path, metadata_df)
        inp = center_crop_axis2(inp)
        inp_np = inp[0, 0].numpy()

        min_d = min(inp_np.shape[0], recon_np.shape[0])
        min_h = min(inp_np.shape[1], recon_np.shape[1])
        min_w = min(inp_np.shape[2], recon_np.shape[2])
        inp_np   = inp_np[:min_d, :min_h, :min_w]
        recon_np = recon_np[:min_d, :min_h, :min_w]
        recon_np = np.clip(recon_np, -1.0, 1.0)

        sd, sh, sw = valid_slices
        sd = slice(sd.start, min(sd.stop, min_d))
        sh = slice(sh.start, min(sh.stop, min_h))
        sw = slice(sw.start, min(sw.stop, min_w))
        inp_roi   = inp_np[sd, sh, sw]
        recon_roi = recon_np[sd, sh, sw]

        ssim = structural_similarity(inp_roi, recon_roi, data_range=2.0)
        psnr = peak_signal_noise_ratio(inp_roi, recon_roi, data_range=2.0)
        mse  = float(np.mean((inp_roi - recon_roi) ** 2))

        if args.save_recon_dir:
            save_recon_artifacts(
                args.save_recon_dir,
                vol_id,
                inp_np,
                recon_np,
                inp_roi,
                recon_roi,
                ssim,
                psnr,
                mse,
                save_full=args.save_recon_full,
            )

        results.append((vol_id, args.bit_order, bool(args.reverse_channels), ssim, psnr, mse))
        print(f"  {vol_id:30s}  SSIM={ssim:.4f}  PSNR={psnr:.2f} dB  MSE={mse:.5f}", flush=True)
        write_results()

    ssims = [r[3] for r in results]
    psnrs = [r[4] for r in results]
    mses  = [r[5] for r in results]
    if not results:
        print(f"\n{'='*70}")
        if resumed_df is not None and len(resumed_df) > 0:
            print(f"No new volumes processed. Existing metrics rows: {len(resumed_df)}")
            if args.out:
                print(f"Per-volume metrics already available at {args.out}")
        else:
            print("No volumes processed.")
        return

    print(f"\n{'='*70}")
    print(f"N={len(results)}  "
          f"SSIM={np.mean(ssims):.4f}±{np.std(ssims):.4f}  "
          f"PSNR={np.mean(psnrs):.2f}±{np.std(psnrs):.2f} dB  "
          f"MSE={np.mean(mses):.5f}±{np.std(mses):.5f}")
    print(f"BTB3D paper Table 1 Stage 3 reference  "
          f"(16x16x8: PSNR=26.75 SSIM=0.749 MSE=0.002 / "
          f"8x8x8: PSNR=28.17 SSIM=0.760 MSE=0.001)")

    if args.out:
        write_results()
        total_rows = len(results) + (0 if resumed_df is None else len(resumed_df))
        print(f"\nSaved per-volume metrics to {args.out} (total rows: {total_rows})")


if __name__ == "__main__":
    main()

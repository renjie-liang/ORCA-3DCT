#!/usr/bin/env python3
"""
Run BTB3D report-generation inference and emit prediction JSONL.

This wrapper keeps `btb3d_baseline/` frozen. It runs a local LLaVA inference
entrypoint that reads BTB3D token artifacts directly, captures its
raw BTB3D-style JSONL in a work directory, and then converts that raw output
into the prediction format accepted by `eval_fast.py`.

Prediction output format:
    {"volume_id": "...",
     "generated_text": "...",
     "reference_text": "...",
     "question": "...",
     "source_image": "...",
     "model_name": "...",
     "variant": "..."}
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

CORE_CODE = Path(__file__).resolve().parents[2]
if str(CORE_CODE) not in sys.path:
    sys.path.insert(0, str(CORE_CODE))

from dtbd3d.core.artifact import read_ids, token_path
from dtbd3d.core.btb3d_model import TOKEN_LAYOUTS
from dtbd3d.core.visual_embedding import resolve_reportgen_artifact_manifest

try:
    from .btb3d_to_metrics_jsonl import convert_btb3d_jsonl
except ImportError:
    from btb3d_to_metrics_jsonl import convert_btb3d_jsonl


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


DEFAULT_CONFIG = project_root() / "Experiment/core_code/dtbd3d/configs/repro_16x16x8.yaml"


def load_config_defaults(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths", {})
    llava = cfg.get("llava_eval_args", {})
    return {
        "btb3d_repo": paths.get("btb3d_repo"),
        "ctclip_repo": paths.get("ctclip_repo"),
        "model_path": paths.get("model_path"),
        "model_base": paths.get("model_base"),
        "token_dir": paths.get("token_dir") or paths.get("canonical_dir"),
        "json_path": paths.get("vqa_json"),
        "reports_csv": paths.get("reports_csv"),
        "conv_mode": llava.get("conv_mode"),
        "temperature": llava.get("temperature"),
        "repetition_penalty": llava.get("repetition_penalty"),
        "max_new_tokens": llava.get("max_new_tokens"),
        "type_filter": llava.get("type_filter"),
        "num_gpus": llava.get("num_gpus"),
    }


def resolve_default_path(value: str | None) -> str | None:
    if not value:
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((project_root() / p).resolve())


def prediction_default_out(workdir: Path, variant: str, model_name: str, json_path: str) -> Path:
    stem = Path(json_path).stem
    tag = variant or model_name or "btb3d"
    safe_tag = tag.replace("/", "_").replace(" ", "_")
    return workdir / f"{stem}_{safe_tag}_predictions.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="YAML config to source inference defaults from")
    p.add_argument("--model-path", default=None)
    p.add_argument("--model-base", default=None)
    p.add_argument("--token-dir", dest="token_dir", default=None,
                   help="Token artifact directory containing ids.txt and tokens/*.npy or tokens_int.npy")
    p.add_argument("--canonical-dir", dest="token_dir", default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--reportgen-artifact-manifest",
        default=None,
        help="final_eval/.../reportgen_artifact/manifest.json; resolves token-dir and codebook automatically.",
    )
    p.add_argument("--split", default="valid", help="Token split to read when using --reportgen-artifact-manifest")
    p.add_argument("--codebook-path", default=None, help="Optional reportgen-ready codebook.npy for learned table artifacts.")
    p.add_argument("--codebook-metadata", default=None, help="metadata.json for --codebook-path.")
    p.add_argument("--compression", choices=["16x16x8", "8x8x8"], default="16x16x8")
    p.add_argument("--bit-order", choices=["msb", "lsb"], default="msb")
    p.add_argument("--reverse-channels", action="store_true",
                   help="Reverse LFQ bit channels when materializing reportgen image tensors")
    p.add_argument("--json-path", default=None)
    p.add_argument("--reports-csv", default=None)
    p.add_argument("--out", default=None, help="Prediction evaluation-format JSONL")
    p.add_argument("--raw-out", default=None, help="Optional path to keep BTB3D raw JSONL")
    p.add_argument("--variant", default="", help="Tag written into prediction JSONL")
    p.add_argument("--model-name", default="", help="Tag written into prediction JSONL")
    p.add_argument("--btb3d-repo", default=None)
    p.add_argument("--ctclip-repo", default=None)
    p.add_argument("--workdir", default=None, help="Directory where BTB3D raw output is produced")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--conv-mode", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--type-filter", default=None)
    p.add_argument("--num-gpus", type=int, default=None)
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Use single-GPU batch inference when >1; keeps the same BTB3D raw output schema.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--fresh", action="store_true",
                   help="Delete existing BTB3D raw output file before running to avoid resume on stale partial runs")
    p.add_argument("--reference-fields", default="Findings_EN,Impressions_EN")
    p.add_argument("--keep-empty-answers", action="store_true")
    args = p.parse_args()

    cfg = load_config_defaults(args.config)
    for key, value in cfg.items():
        current = getattr(args, key)
        if current is None:
            setattr(args, key, value)

    for key in (
        "btb3d_repo",
        "ctclip_repo",
        "model_path",
        "token_dir",
        "json_path",
        "reports_csv",
        "reportgen_artifact_manifest",
        "codebook_path",
        "codebook_metadata",
    ):
        setattr(args, key, resolve_default_path(getattr(args, key)))

    if args.model_base:
        args.model_base = resolve_default_path(args.model_base)

    if args.reportgen_artifact_manifest:
        artifact = resolve_reportgen_artifact_manifest(args.reportgen_artifact_manifest, args.split)
        if args.token_dir and Path(args.token_dir).resolve() != artifact.token_dir.resolve():
            raise SystemExit(
                f"--token-dir {args.token_dir} conflicts with manifest split {args.split}: {artifact.token_dir}"
            )
        if args.codebook_path and Path(args.codebook_path).resolve() != artifact.codebook_path.resolve():
            raise SystemExit(
                f"--codebook-path {args.codebook_path} conflicts with manifest codebook: {artifact.codebook_path}"
            )
        if args.codebook_metadata and Path(args.codebook_metadata).resolve() != artifact.codebook_metadata_path.resolve():
            raise SystemExit(
                f"--codebook-metadata {args.codebook_metadata} conflicts with manifest metadata: "
                f"{artifact.codebook_metadata_path}"
            )
        args.token_dir = str(artifact.token_dir)
        args.codebook_path = str(artifact.codebook_path)
        args.codebook_metadata = str(artifact.codebook_metadata_path)

    if args.conv_mode is None:
        args.conv_mode = "llama3"
    if args.temperature is None:
        args.temperature = 0.0
    if args.repetition_penalty is None:
        args.repetition_penalty = 1.3
    if args.max_new_tokens is None:
        args.max_new_tokens = 512
    if args.type_filter is None:
        args.type_filter = "report_generation"
    if args.num_gpus is None:
        args.num_gpus = 1
    if args.num_gpus <= 0:
        raise SystemExit(f"--num-gpus must be positive, got {args.num_gpus}")
    if args.batch_size <= 0:
        raise SystemExit(f"--batch-size must be positive, got {args.batch_size}")
    if bool(args.codebook_path) != bool(args.codebook_metadata):
        raise SystemExit("--codebook-path and --codebook-metadata must be provided together")

    missing = [name for name in ("model_path", "model_base", "token_dir", "json_path", "reports_csv") if not getattr(args, name)]
    if missing:
        raise SystemExit(f"Missing required arguments after config defaults: {', '.join(missing)}")

    return args


def expected_raw_name(model_path: str, type_filter: str) -> str:
    save_it = Path(model_path).resolve().parent.name
    type_tag = f"_{type_filter}" if type_filter else ""
    return f"16_preprocessed_encoded_attnpool_1node_{save_it}it_node0{type_tag}_vqa.jsonl"


def load_expected_image_files(json_path: str, type_filter: str) -> list[str]:
    with open(json_path) as f:
        data = json.load(f)
    if type_filter:
        data = [d for d in data if d["id"].startswith(type_filter)]
    out = []
    for d in data:
        image = d["image"]
        if image.endswith(".nii_embedded.npz"):
            out.append(image)
        elif image.endswith(".nii.gz"):
            out.append(image.replace(".nii.gz", ".nii_embedded.npz"))
        elif image.endswith(".npz"):
            out.append(image)
        else:
            out.append(f"{image}.nii_embedded.npz")
    return out


def volume_id_from_image(image: str) -> str:
    name = Path(image).name
    for suffix in [".nii_embedded.npz", ".nii.gz", ".nii", ".npz"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def ensure_token_rows_exist(token_dir: str, image_files: list[str], compression: str) -> None:
    missing = []
    base = Path(token_dir)
    ids = set(read_ids(base / "ids.txt"))
    matrix_path = base / "tokens_int.npy"
    matrix_exists = matrix_path.exists()
    token_t, token_h, token_w = TOKEN_LAYOUTS[compression]
    expected_tokens = token_t * token_h * token_w
    if matrix_exists:
        matrix = np.load(matrix_path, mmap_mode="r")
        if matrix.ndim != 2 or matrix.shape[1] != expected_tokens:
            raise ValueError(f"{matrix_path} shape {matrix.shape}; expected (*, {expected_tokens})")
        if ids and matrix.shape[0] != len(ids):
            raise ValueError(f"{matrix_path} row count {matrix.shape[0]} does not match ids.txt count {len(ids)}")
    for image_file in image_files:
        volume_id = volume_id_from_image(image_file)
        row_path = token_path(base, volume_id)
        has_per_volume = row_path.exists()
        has_matrix_row = matrix_exists and volume_id in ids
        if has_per_volume:
            row = np.load(row_path, mmap_mode="r")
            if row.shape != (expected_tokens,):
                raise ValueError(f"{row_path} shape {row.shape}; expected {(expected_tokens,)}")
            if row.dtype != np.uint32:
                raise TypeError(f"{row_path} dtype {row.dtype}; expected uint32")
        if not (has_per_volume or has_matrix_row):
            missing.append(str(row_path))
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} token rows under {token_dir} before inference.\n"
            f"First missing paths:\n{preview}"
        )


def count_raw_records(path: Path) -> int:
    n = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def enrich_prediction_jsonl(path: Path, model_name: str, variant: str) -> None:
    rows = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if model_name:
                rec["model_name"] = model_name
            if variant:
                rec["variant"] = variant
            rows.append(rec)
    with path.open("w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")


def main():
    args = parse_args()
    if args.out:
        out_path = Path(args.out).resolve()
        default_workdir = out_path.parent
    else:
        default_workdir = Path(args.token_dir).resolve()
        out_path = prediction_default_out(default_workdir, args.variant, args.model_name, args.json_path).resolve()
    workdir = Path(args.workdir).resolve() if args.workdir else default_workdir
    repo = Path(args.btb3d_repo).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_batch_inference = args.batch_size > 1
    raw_name = "batch_token_artifact_report_generation_vqa.jsonl" if use_batch_inference else expected_raw_name(args.model_path, args.type_filter)
    raw_generated = workdir / raw_name
    if args.fresh and raw_generated.exists():
        raw_generated.unlink()
        print(f"Removed stale raw output before rerun: {raw_generated}")

    expected_images = load_expected_image_files(args.json_path, args.type_filter)
    if not expected_images:
        raise ValueError(f"No expected report-generation images found in {args.json_path} with type_filter={args.type_filter!r}")
    print(f"Preflight: expecting {len(expected_images)} token row(s)")
    ensure_token_rows_exist(args.token_dir, expected_images, args.compression)

    env = os.environ.copy()
    py_paths = [str(CORE_CODE), str(repo)]
    if args.ctclip_repo:
        py_paths.append(str(Path(args.ctclip_repo).resolve()))
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(py_paths)

    if use_batch_inference:
        if args.num_gpus != 1:
            raise SystemExit("--batch-size > 1 currently supports single-GPU inference only; set --num-gpus 1")
        cmd = [
            args.python_bin,
            str(project_root() / "Experiment/core_code/dtbd3d/eval/batch_llava_eval_token_artifact.py"),
            "--model-path", args.model_path,
            "--model-base", args.model_base,
            "--token_dir", args.token_dir,
            "--compression", args.compression,
            "--bit-order", args.bit_order,
            "--json_path", args.json_path,
            "--output", str(raw_generated),
            "--device", args.device,
            "--conv-mode", args.conv_mode,
            "--temperature", str(args.temperature),
            "--repetition_penalty", str(args.repetition_penalty),
            "--max-new-tokens", str(args.max_new_tokens),
            "--type_filter", args.type_filter,
            "--batch-size", str(args.batch_size),
        ]
    else:
        cmd = [
            args.python_bin,
            str(project_root() / "Experiment/core_code/dtbd3d/eval/multigpu_llava_eval_token_artifact.py"),
            "--model-path", args.model_path,
            "--model-base", args.model_base,
            "--token_dir", args.token_dir,
            "--compression", args.compression,
            "--bit-order", args.bit_order,
            "--json_path", args.json_path,
            "--device", args.device,
            "--conv-mode", args.conv_mode,
            "--temperature", str(args.temperature),
            "--repetition_penalty", str(args.repetition_penalty),
            "--max-new-tokens", str(args.max_new_tokens),
            "--type_filter", args.type_filter,
            "--num_gpus", str(args.num_gpus),
        ]
    if args.reverse_channels:
        cmd.append("--reverse-channels")
    if args.codebook_path:
        cmd.extend(["--codebook-path", args.codebook_path, "--codebook-metadata", args.codebook_metadata])
    if args.debug:
        cmd.append("--debug")

    print("Running BTB3D inference...")
    print("Workdir:", workdir)
    subprocess.run(cmd, cwd=workdir, env=env, check=True)

    if not raw_generated.exists():
        raise FileNotFoundError(f"Expected BTB3D raw output not found: {raw_generated}")

    raw_count = count_raw_records(raw_generated)
    if raw_count != len(expected_images):
        raise RuntimeError(
            f"Unexpected BTB3D raw output row count: expected {len(expected_images)} records from {args.json_path}, "
            f"but raw file contains {raw_count}. Use --fresh for a clean run and inspect {raw_generated}."
        )

    convert_btb3d_jsonl(
        btb3d_jsonl=str(raw_generated),
        reports_csv=args.reports_csv,
        out_jsonl=str(out_path),
        reference_fields=[s.strip() for s in args.reference_fields.split(",") if s.strip()],
        report_task_only=True,
        keep_empty_answers=args.keep_empty_answers,
    )
    enrich_prediction_jsonl(out_path, model_name=args.model_name, variant=args.variant)

    if args.raw_out:
        raw_out = Path(args.raw_out).resolve()
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_generated, raw_out)
        print(f"Copied raw BTB3D output to {raw_out}")

    print(f"Predictions written to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print the BTB3D token-artifact full-validation command manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    root = project_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-root",
        default=str(root / "Experiment/runs/inference_repro/full_validation_matrix_token_artifact"),
        help="Output root for token-artifact full-validation matrix artifacts.",
    )
    p.add_argument(
        "--valid-fixed",
        default=str(root / "./data/ct_rate/dataset/valid_fixed"),
    )
    p.add_argument(
        "--valid-vqa",
        default=str(root / "./data/ct_rate/dataset/vqa/valid_vqa.json"),
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--manifest-json", default=None)
    p.add_argument("--manifest-md", default=None)
    p.add_argument("--write-manifest", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    run_root = Path(args.run_root)
    core_code = root / "Experiment/core_code"
    eval_dir = core_code / "dtbd3d/eval"
    labels_csv = core_code / "data_links/ct_rate/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
    radbert = core_code / "data_links/ctclip_weights/RadBertClassifier.pth"
    t16 = run_root / "token_artifacts/16x16x8/valid"
    t8 = run_root / "token_artifacts/8x8x8/valid"

    commands = [
        {
            "id": "extract_16x16x8_tokens",
            "title": "extract 16x16x8 token artifact",
            "task": "token_artifact_extraction",
            "resolution": "16x16x8",
            "requires_gpu": True,
            "outputs": [str(t16 / "ids.txt"), str(t16 / "tokens_int.npy"), str(t16 / "tokens")],
            "command": (
                f"PYTHONPATH={core_code} python {eval_dir / 'extract_token_artifact.py'} "
                f"--compression 16x16x8 --input {args.valid_fixed} --out-dir {t16} "
                f"--device {args.device} --write-matrix"
            ),
        },
        {
            "id": "extract_8x8x8_tokens",
            "title": "extract 8x8x8 token artifact",
            "task": "token_artifact_extraction",
            "resolution": "8x8x8",
            "requires_gpu": True,
            "outputs": [str(t8 / "ids.txt"), str(t8 / "tokens_int.npy"), str(t8 / "tokens")],
            "command": (
                f"PYTHONPATH={core_code} python {eval_dir / 'extract_token_artifact.py'} "
                f"--compression 8x8x8 --input {args.valid_fixed} --out-dir {t8} "
                f"--device {args.device} --write-matrix"
            ),
        },
        {
            "id": "inspect_16x16x8_tokens",
            "title": "inspect 16x16x8 token artifact",
            "task": "token_artifact_inspection",
            "resolution": "16x16x8",
            "requires_gpu": False,
            "depends_on": ["extract_16x16x8_tokens"],
            "outputs": [str(run_root / "token_artifacts/16x16x8/valid_inspect.json")],
            "command": (
                f"PYTHONPATH={core_code} python {eval_dir / 'inspect_token_artifact.py'} "
                f"--compression 16x16x8 --token-dir {t16} --check-per-volume "
                f"--out {run_root / 'token_artifacts/16x16x8/valid_inspect.json'}"
            ),
        },
        {
            "id": "inspect_8x8x8_tokens",
            "title": "inspect 8x8x8 token artifact",
            "task": "token_artifact_inspection",
            "resolution": "8x8x8",
            "requires_gpu": False,
            "depends_on": ["extract_8x8x8_tokens"],
            "outputs": [str(run_root / "token_artifacts/8x8x8/valid_inspect.json")],
            "command": (
                f"PYTHONPATH={core_code} python {eval_dir / 'inspect_token_artifact.py'} "
                f"--compression 8x8x8 --token-dir {t8} --check-per-volume "
                f"--out {run_root / 'token_artifacts/8x8x8/valid_inspect.json'}"
            ),
        },
        {
            "id": "run_16x16x8_reportgen",
            "title": "run 16x16x8 report generation",
            "task": "report_generation",
            "resolution": "16x16x8",
            "requires_gpu": True,
            "depends_on": ["inspect_16x16x8_tokens"],
            "outputs": [
                str(run_root / "reportgen_16x16x8/full_valid_predictions.jsonl"),
                str(run_root / "reportgen_16x16x8/full_valid_raw_btb3d.jsonl"),
            ],
            "command": (
                f"PYTHONPATH={core_code} python {eval_dir / 'run_report_generation.py'} "
                f"--config {core_code / 'dtbd3d/configs/repro_16x16x8.yaml'} "
                f"--token-dir {t16} --compression 16x16x8 --bit-order msb --reverse-channels "
                f"--json-path {args.valid_vqa} "
                f"--variant token_artifact_16x16x8_msb_reverse_channels "
                f"--model-name btb3d-16-token-artifact-msb-reverse "
                f"--out {run_root / 'reportgen_16x16x8/full_valid_predictions.jsonl'} "
                f"--raw-out {run_root / 'reportgen_16x16x8/full_valid_raw_btb3d.jsonl'} --fresh"
            ),
        },
        {
            "id": "eval_16x16x8_reportgen",
            "title": "evaluate 16x16x8 report generation",
            "task": "report_generation_eval",
            "resolution": "16x16x8",
            "requires_gpu": True,
            "depends_on": ["run_16x16x8_reportgen"],
            "outputs": [str(run_root / "reportgen_16x16x8/full_valid_metrics_fast.json")],
            "command": (
                f"python {eval_dir / 'eval_fast.py'} "
                f"--pred {run_root / 'reportgen_16x16x8/full_valid_predictions.jsonl'} "
                f"--labels-csv {labels_csv} --radbert {radbert} --device {args.device} "
                f"--out {run_root / 'reportgen_16x16x8/full_valid_metrics_fast.json'}"
            ),
        },
        {
            "id": "run_8x8x8_reportgen",
            "title": "run 8x8x8 report generation",
            "task": "report_generation",
            "resolution": "8x8x8",
            "requires_gpu": True,
            "depends_on": ["inspect_8x8x8_tokens"],
            "outputs": [
                str(run_root / "reportgen_8x8x8/full_valid_predictions.jsonl"),
                str(run_root / "reportgen_8x8x8/full_valid_raw_btb3d.jsonl"),
            ],
            "command": (
                f"PYTHONPATH={core_code} python {eval_dir / 'run_report_generation.py'} "
                f"--config {core_code / 'dtbd3d/configs/repro_8x8x8.yaml'} "
                f"--token-dir {t8} --compression 8x8x8 --bit-order msb "
                f"--json-path {args.valid_vqa} "
                f"--variant token_artifact_8x8x8_merged72 "
                f"--model-name btb3d-8-token-artifact-merged72 "
                f"--out {run_root / 'reportgen_8x8x8/full_valid_predictions.jsonl'} "
                f"--raw-out {run_root / 'reportgen_8x8x8/full_valid_raw_btb3d.jsonl'} --fresh"
            ),
        },
        {
            "id": "eval_8x8x8_reportgen",
            "title": "evaluate 8x8x8 report generation",
            "task": "report_generation_eval",
            "resolution": "8x8x8",
            "requires_gpu": True,
            "depends_on": ["run_8x8x8_reportgen"],
            "outputs": [str(run_root / "reportgen_8x8x8/full_valid_metrics_fast.json")],
            "command": (
                f"python {eval_dir / 'eval_fast.py'} "
                f"--pred {run_root / 'reportgen_8x8x8/full_valid_predictions.jsonl'} "
                f"--labels-csv {labels_csv} --radbert {radbert} --device {args.device} "
                f"--out {run_root / 'reportgen_8x8x8/full_valid_metrics_fast.json'}"
            ),
        },
        {
            "id": "run_16x16x8_reconstruction",
            "title": "run 16x16x8 reconstruction",
            "task": "reconstruction",
            "resolution": "16x16x8",
            "requires_gpu": True,
            "depends_on": ["extract_16x16x8_tokens"],
            "outputs": [str(run_root / "reconstruction_16x16x8/full_valid_msb_identity.csv")],
            "command": (
                f"PYTHONPATH={core_code} python {eval_dir / 'verify_btb3d_recon.py'} "
                f"--compression 16x16x8 --token-dir {t16} --n-samples 3039 "
                f"--device {args.device} --bit-order msb "
                f"--out {run_root / 'reconstruction_16x16x8/full_valid_msb_identity.csv'}"
            ),
        },
        {
            "id": "run_8x8x8_reconstruction",
            "title": "run 8x8x8 reconstruction",
            "task": "reconstruction",
            "resolution": "8x8x8",
            "requires_gpu": True,
            "depends_on": ["extract_8x8x8_tokens"],
            "outputs": [str(run_root / "reconstruction_8x8x8/full_valid_msb_identity.csv")],
            "command": (
                f"PYTHONPATH={core_code} python {eval_dir / 'verify_btb3d_recon.py'} "
                f"--compression 8x8x8 --token-dir {t8} --n-samples 3039 "
                f"--device {args.device} --bit-order msb "
                f"--out {run_root / 'reconstruction_8x8x8/full_valid_msb_identity.csv'}"
            ),
        },
    ]

    print(f"Run root: {run_root}\n")
    for i, item in enumerate(commands, 1):
        print(f"## {i}. {item['title']}")
        if item.get("depends_on"):
            print(f"depends_on: {', '.join(item['depends_on'])}")
        print(item["command"])
        print()
    print("Note: report-generation and reconstruction commands both read the shared token artifact.")

    manifest = {
        "run_root": str(run_root),
        "valid_fixed": str(args.valid_fixed),
        "valid_vqa": str(args.valid_vqa),
        "device": args.device,
        "commands": commands,
    }
    if args.write_manifest or args.manifest_json or args.manifest_md:
        json_path = Path(args.manifest_json) if args.manifest_json else run_root / "full_validation_matrix_manifest.json"
        md_path = Path(args.manifest_md) if args.manifest_md else run_root / "full_validation_matrix_manifest.md"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(manifest, indent=2))
        lines = [
            "# BTB3D Full Validation Matrix Manifest",
            "",
            f"Run root: `{run_root}`",
            f"Device: `{args.device}`",
            "",
        ]
        for i, item in enumerate(commands, 1):
            lines.extend(
                [
                    f"## {i}. {item['title']}",
                    "",
                    f"- id: `{item['id']}`",
                    f"- task: `{item['task']}`",
                    f"- resolution: `{item['resolution']}`",
                    f"- requires_gpu: `{item['requires_gpu']}`",
                ]
            )
            if item.get("depends_on"):
                lines.append(f"- depends_on: `{', '.join(item['depends_on'])}`")
            if item.get("outputs"):
                lines.append("- outputs:")
                for output in item["outputs"]:
                    lines.append(f"  - `{output}`")
            lines.extend(["", "```bash", item["command"], "```", ""])
        md_path.write_text("\n".join(lines))
        print(f"Wrote JSON manifest: {json_path}")
        print(f"Wrote Markdown manifest: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

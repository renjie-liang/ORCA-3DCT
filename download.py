#!/usr/bin/env python3
"""Fetch the released assets and put them where the code expects them.

    python download.py --list                          # what exists, and what it costs in disk
    python download.py --bundle reportgen --budget 216 # the smallest thing that trains a full cell
    python download.py --bundle uncompressed --encoder colipri

Nothing here is clever: it is `snapshot_download` with the right allow-patterns, plus the manifests the
trainer reads, written against the paths that actually landed on this machine. Manifests are generated
rather than shipped so a moved or partial download fails at download time instead of three hours into a
training run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = "LiangRenjie/ORCA-3DCT"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"

# Grid shape and token dim per (method, budget). ORCA regions have no spatial layout, hence [B,1,1];
# Grid average keeps the cube it pooled over. Both feed the decoder the same NUMBER of tokens, which is
# what makes the comparison budget-matched.
ARMS = {
    ("ORCA", 8):      {"grid": [8, 1, 1],   "dim": 792},
    ("ORCA", 27):     {"grid": [27, 1, 1],  "dim": 792},
    ("ORCA", 64):     {"grid": [64, 1, 1],  "dim": 792},
    ("ORCA", 216):    {"grid": [216, 1, 1], "dim": 792},
    ("GridAvg", 8):   {"grid": [2, 2, 2],   "dim": 768},
    ("GridAvg", 27):  {"grid": [3, 3, 3],   "dim": 768},
    ("GridAvg", 64):  {"grid": [4, 4, 4],   "dim": 768},
    ("GridAvg", 216): {"grid": [6, 6, 6],   "dim": 768},
}

# name -> (patterns, destination, approx GB, one-line description)
BUNDLES = {
    "compressed": (
        ["compressed/colipri/**"], DATA / "embeddings", 25.3,
        "ORCA and Grid-average tokens, COLIPRI, all four budgets",
    ),
    "uncompressed": (
        ["uncompressed/**"], DATA / "embeddings" / "uncompressed", 546.0,
        "the raw encoder grids — 545 GB for COLIPRI, ~1 GB for the pooled encoders",
    ),
    "organ_masks": (
        ["organ_masks/**"], DATA, 0.3,
        "TotalSegmentator organ occupancy on the COLIPRI grid — needed to run ORCA yourself",
    ),
    "checkpoints": (
        ["checkpoints/**"], CKPT, 11.2,
        "our four report-generation checkpoints (best epoch per arm, by clinical F1)",
    ),
}

# Everything a report-generation cell needs, and nothing else.
BUNDLE_SETS = {
    "reportgen": ["compressed"],
    "all": list(BUNDLES),
}

# Reports, labels and the question set are CT-RATE's own files, not ours, so they are not mirrored here.
# CT-RATE is gated; accept its terms once and these three downloads land where the code expects them.
CTRATE = [
    ("dataset/vqa/train_vqa.json",                          "data/vqa/train_reportgen.json",       "1.2 GB"),
    ("dataset/vqa/valid_vqa.json",                          "data/vqa/valid_reportgen.json",       "37 MB"),
    ("dataset/radiology_text_reports/validation_reports.csv", "data/reports/validation_reports.csv", "5 MB"),
    ("dataset/multi_abnormality_labels/valid_predicted_labels.csv", "data/labels/valid_predicted_labels.csv", "0.2 MB"),
]


def fetch_ctrate() -> None:
    """Pull the four CT-RATE files the trainer needs, straight from the CT-RATE repo."""
    from huggingface_hub import hf_hub_download

    print("\nfetching annotations from ibrahimhamamci/CT-RATE (gated — accept its terms on the Hub first)")
    for remote, local, size in CTRATE:
        dst = ROOT / local
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"  have {local}")
            continue
        print(f"  {remote}  ({size})")
        src = hf_hub_download("ibrahimhamamci/CT-RATE", remote, repo_type="dataset")
        shutil.copy2(src, dst)
        print(f"    -> {local}")


def human(gb: float) -> str:
    return f"{gb * 1024:.0f} MB" if gb < 1 else f"{gb:.1f} GB"


def do_list() -> None:
    print(f"\n  https://huggingface.co/datasets/{REPO}\n")
    print(f"  {'bundle':<14} {'size':>9}   description")
    print(f"  {'-' * 14} {'-' * 9}   {'-' * 60}")
    for name, (_, _, gb, desc) in BUNDLES.items():
        print(f"  {name:<14} {human(gb):>9}   {desc}")
    print(f"\n  shortcuts: {', '.join(f'{k} = {" + ".join(v)}' for k, v in BUNDLE_SETS.items())}")
    print("\n  --budget / --encoder narrow a bundle; see --help.\n")
    print("  Only what we computed is mirrored here. The reports, the 18 abnormality labels and the")
    print("  question set are CT-RATE's own files -- `--annotations` fetches them from the CT-RATE repo,")
    print("  which is gated. Llama-3.1-8B-Instruct comes from meta-llama via `--base-weights`, and")
    print("  RadBertClassifier.pth must come from the CT-CLIP release (see README).\n")


def fetch(patterns: list[str], dest: Path, budget: int | None, encoder: str | None) -> Path:
    from huggingface_hub import snapshot_download

    if budget is not None:
        patterns = [p.replace("colipri/**", f"colipri/*_b{budget}/*") for p in patterns]
    if encoder is not None:
        patterns = [p.replace("uncompressed/**", f"uncompressed/{encoder}/**") for p in patterns]
    print(f"  patterns: {patterns}")
    local = snapshot_download(REPO, repo_type="dataset", allow_patterns=patterns)
    dest.mkdir(parents=True, exist_ok=True)
    return Path(local)


def write_manifests(store: Path, budget: int | None) -> None:
    """One manifest per arm actually on disk. Absent splits are an error, not a warning."""
    out_dir = DATA / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for (method, bud), spec in ARMS.items():
        if budget is not None and bud != budget:
            continue
        arm_dir = store / "compressed" / "colipri" / f"{method}_b{bud}"
        if not arm_dir.is_dir():
            continue
        splits = {}
        for split in ("train", "valid"):
            array, ids = arm_dir / f"{split}.npy", arm_dir / f"{split}_ids.txt"
            if not array.exists() or not ids.exists():
                sys.exit(f"FATAL: {arm_dir.name} is missing {split} — re-run the download")
            splits[split] = {"array": str(array), "ids": str(ids)}
        manifest = out_dir / f"manifest_{method}_b{bud}.json"
        manifest.write_text(json.dumps({
            "artifact_type": "npy_stacked",
            "encoder": "colipri",
            "method": method,
            "budget": bud,
            "visual_dim": spec["dim"],
            "grid_shape": spec["grid"],
            "splits": splits,
        }, indent=1) + "\n")
        written += 1
        print(f"  manifest: {manifest.relative_to(ROOT)}")
    if written == 0:
        sys.exit("FATAL: no token arms found on disk — nothing to write a manifest for")


def fetch_base_weights() -> None:
    """Llama comes from Meta's own repo. We do not mirror an 16 GB model that is already on the Hub."""
    from huggingface_hub import snapshot_download

    dest = CKPT / "Llama-3.1-8B-Instruct"
    if dest.exists():
        print(f"  base weights already at {dest.relative_to(ROOT)}")
        return
    CKPT.mkdir(parents=True, exist_ok=True)
    print("  fetching meta-llama/Llama-3.1-8B-Instruct (gated — accept the license on the Hub first)")
    snapshot_download("meta-llama/Llama-3.1-8B-Instruct", local_dir=dest,
                      allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.jinja"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show the bundles and their sizes, then exit")
    ap.add_argument("--bundle", help=f"one of {', '.join(list(BUNDLES) + list(BUNDLE_SETS))}")
    ap.add_argument("--budget", type=int, choices=[8, 27, 64, 216], help="restrict token bundles to one budget")
    ap.add_argument("--encoder", help="restrict the uncompressed bundle to one encoder, e.g. colipri")
    ap.add_argument("--annotations", action="store_true",
                    help="also fetch the reports, labels and question set from the CT-RATE repo")
    ap.add_argument("--base-weights", action="store_true", help="also fetch Llama-3.1-8B-Instruct")
    args = ap.parse_args()

    if args.list or args.bundle is None:
        do_list()
        return

    names = BUNDLE_SETS.get(args.bundle, [args.bundle])
    unknown = [n for n in names if n not in BUNDLES]
    if unknown:
        sys.exit(f"FATAL: unknown bundle {unknown[0]!r}; run --list")

    total = sum(BUNDLES[n][2] for n in names)
    print(f"\nfetching {', '.join(names)} (~{human(total)}) from {REPO}\n")

    store = None
    for name in names:
        patterns, dest, gb, desc = BUNDLES[name]
        print(f"[{name}] {desc}  (~{human(gb)})")
        store = fetch(patterns, dest, args.budget, args.encoder)

    if any(n == "compressed" for n in names):
        print("\nwriting manifests")
        write_manifests(store, args.budget)

    if args.annotations:
        fetch_ctrate()

    if args.base_weights:
        print("\nbase weights")
        fetch_base_weights()

    print("\ndone. RadBertClassifier.pth is not redistributed here — see README for where to get it.")


if __name__ == "__main__":
    main()

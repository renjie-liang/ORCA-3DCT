#!/usr/bin/env python3
"""Deprecated wrapper for multigpu_llava_eval_token_artifact.py."""

from __future__ import annotations

import argparse
from pathlib import Path
import multiprocessing
import sys

CORE_CODE = Path(__file__).resolve().parents[2]
if str(CORE_CODE) not in sys.path:
    sys.path.insert(0, str(CORE_CODE))

from dtbd3d.eval.multigpu_llava_eval_token_artifact import main


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Deprecated canonical-token LLaVA wrapper")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--token_dir", type=str, default=None)
    parser.add_argument("--canonical_dir", dest="token_dir", type=str, default=None)
    parser.add_argument("--compression", choices=["16x16x8", "8x8x8"], required=True)
    parser.add_argument("--bit-order", choices=["msb", "lsb"], default="msb")
    parser.add_argument("--reverse-channels", action="store_true")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--type_filter", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()
    if not args.token_dir:
        raise SystemExit("Missing required argument: --token_dir")
    main(args)

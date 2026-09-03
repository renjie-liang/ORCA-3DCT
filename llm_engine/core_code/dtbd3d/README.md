# DTBD3D Baseline Code

This package contains the cleaned baseline code used to reproduce the BTB3D
inference and evaluation pipeline on CT-RATE validation data.

The baseline is intentionally minimal. The main goal is to keep one shared
token artifact that can be consumed by both downstream tasks:

- report generation
- 3D reconstruction

## Core Data Contract

The shared artifact is stored under directories named `token_artifacts/`:

```text
<token_artifact_dir>/
  ids.txt
  tokens_int.npy
  tokens/
    valid_1_a_1.npy
    valid_1_a_2.npy
    ...
```

Each token is the direct integer index from the BTB3D tokenizer
`quantized_output.indices`. This is the source of truth. We do not reconstruct
tokens from report-generation `.npz` files.

Expected layouts:

| Compression | Token layout | Tokens per volume |
|---|---:|---:|
| `16x16x8` | `31 x 32 x 32` | `31744` |
| `8x8x8` | `31 x 64 x 64` | `126976` |

The reconstruction path unpacks these integer tokens directly into the LFQ
representation expected by the BTB3D decoder.

For `8x8x8` report generation only, the code performs the BTB3D-required
runtime merge from direct 18-bit tokens to the 72-channel report-generation
input. The stored token artifact remains direct tokenizer indices.

## Main Entry Points

Token artifact extraction:

```bash
python Experiment/core_code/dtbd3d/eval/extract_token_artifact.py
```

Token artifact inspection:

```bash
python Experiment/core_code/dtbd3d/eval/inspect_token_artifact.py
```

Report generation from token artifacts:

```bash
python Experiment/core_code/dtbd3d/eval/run_report_generation.py
```

Reconstruction evaluation from token artifacts:

```bash
python Experiment/core_code/dtbd3d/eval/verify_btb3d_recon.py
```

Report-generation metrics:

```bash
python Experiment/core_code/dtbd3d/eval/eval_fast.py
```

One-off investigation scripts from sub task 1 were moved out of this package:

```text
Experiment/archived_code/sub_task1_inference_repro/eval_debug_scripts/
```

Use those archived scripts only for provenance/debugging. They are not part of
the maintained baseline API.

## Resume Behavior

Extraction resumes from per-volume files:

```text
<token_artifact_dir>/tokens/<volume_id>.npy
```

If a token file already exists and `--overwrite` is not passed, extraction skips
that volume. `tokens_int.npy` is the compact matrix artifact; the per-volume
`tokens/` directory is the safest resume boundary.

Report generation resumes from the raw BTB3D JSONL written in the reportgen
work directory:

```text
16_preprocessed_encoded_attnpool_1node_<compression>it_node0_report_generation_vqa.jsonl
```

Do not pass `--fresh` when resuming report generation. `--fresh` deletes the raw
BTB3D JSONL and starts from zero.

Reconstruction resumes from the output CSV:

```text
full_valid_msb_identity.csv
```

Existing `volume_id` rows are skipped.

## Output Types

There are three different output concepts:

| Concept | Current name/path pattern | Purpose |
|---|---|---|
| Token artifact | `token_artifacts/...` | Shared tokenizer-index artifact consumed by reportgen and reconstruction |
| Raw reportgen output | `*_report_generation_vqa.jsonl` | Incremental BTB3D/LLaVA output used for reportgen resume |
| Prediction JSONL | `full_valid_predictions.jsonl` | Evaluation input consumed by `eval_fast.py` |

Legacy compatibility:

- Older completed runs may still contain paths named `canonical_tokens/` and
  files named `full_valid_canonical.jsonl`.
- Deprecated wrapper scripts and `--canonical-dir` aliases are kept so recorded
  commands remain runnable.
- New experiments should use `token_artifacts/`, `--token-dir`, and
  `full_valid_predictions.jsonl`.

## Current Validated Assumptions

- Direct tokenizer indices are the correct source of truth.
- `8x8x8` reconstruction works from the direct token artifact.
- `16x16x8` reconstruction uses `bit_order=msb` and no channel reversal.
- `8x8x8` report generation must apply the 72-channel merge at runtime.
- The old path that tried to invert merged 72-channel reportgen `.npz` files
  back into 18-bit tokens is not reliable and should not be used as the baseline.

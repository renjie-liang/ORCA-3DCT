# Recorded Run Outputs

All ReportGen experiment outputs are stored under this `runs/` directory.

## Top-level layout

```text
runs/
├── logs/
├── reportgen_orca_b8/
├── reportgen_orca_b27/
├── reportgen_orca_b64/
├── reportgen_orca_b216/
└── reportgen_orcarand_b8/
```

## Logs

SLURM stdout/stderr logs are stored in:

```text
runs/logs/
```

Log filenames include the job name and SLURM job id, for example:

```text
runs/logs/rg_orca_b8_s2_2of4_h10_37560344.out
runs/logs/rg_orcarand_b8_s1_1of2_37572906.out
```

Use these files to inspect runtime progress, resume behavior, errors, and generation progress.

## Real ORCA ReportGen runs

The real-token ORCA ReportGen runs are stored in:

```text
runs/reportgen_orca_b8/
runs/reportgen_orca_b27/
runs/reportgen_orca_b64/
runs/reportgen_orca_b216/
```

Each cell has two stages:

```text
reportgen_orca_b*/reportgen_orca_b*__s1/   # projector-only stage
reportgen_orca_b*/reportgen_orca_b*__s2/   # LoRA + projector stage
```

Important result files inside each stage:

```text
evaluations/step_XXXXXX/metrics_fast.json      # main evaluation metrics
evaluations/step_XXXXXX/predictions.jsonl      # generated reports aligned for metrics
evaluations/step_XXXXXX/raw_btb3d_output.jsonl # raw generation output
evaluations/step_XXXXXX/clinical_labels.jsonl  # RadBERT clinical label outputs
evaluations/step_XXXXXX/valid_loss.json        # validation loss / perplexity
checkpoints/step_XXXXXX/training_state.pt      # resumable training state
checkpoints/training_state_latest.pt           # symlink/pointer to latest training state
run.log                                        # stage-level training log
run_config.json                                # resolved run configuration
```

Step-to-epoch mapping:

```text
1508 steps = 1 epoch
3016 steps = 2 epochs
4524 steps = 3 epochs
6032 steps = 4 epochs
```

Current key completed comparison outputs:

```text
runs/reportgen_orca_b8/reportgen_orca_b8__s2/evaluations/step_001508/metrics_fast.json
runs/reportgen_orca_b8/reportgen_orca_b8__s2/evaluations/step_003016/metrics_fast.json
runs/reportgen_orca_b8/reportgen_orca_b8__s2/evaluations/step_004524/metrics_fast.json
runs/reportgen_orca_b8/reportgen_orca_b8__s2/evaluations/step_006032/metrics_fast.json

runs/reportgen_orca_b216/reportgen_orca_b216__s2/evaluations/step_001508/metrics_fast.json
runs/reportgen_orca_b216/reportgen_orca_b216__s2/evaluations/step_003016/metrics_fast.json
runs/reportgen_orca_b216/reportgen_orca_b216__s2/evaluations/step_004524/metrics_fast.json
runs/reportgen_orca_b216/reportgen_orca_b216__s2/evaluations/step_006032/metrics_fast.json
```

These are the completed real-token `b8` and `b216` LoRA-stage epoch 1--4 metrics used for fair best-epoch comparison.

## Random-token control

The random-token control run is stored in:

```text
runs/reportgen_orcarand_b8/
```

Stages:

```text
runs/reportgen_orcarand_b8/reportgen_orcarand_b8__s1/
runs/reportgen_orcarand_b8/reportgen_orcarand_b8__s2/
```

The random-token input cache and manifest are outside `runs/`:

```text
data/tokens/colipri_orcarand_b8/
data/tokens/manifest_orcarand_b8.json
```

The random control uses the same tensor shape and dtype as the real `orca b8` tokens, with random values sampled using matched mean/std estimated from the real `colipri_orca_b8` tokens.

Completed random-control outputs currently include:

```text
runs/reportgen_orcarand_b8/reportgen_orcarand_b8__s1/evaluations/step_001508/metrics_fast.json
runs/reportgen_orcarand_b8/reportgen_orcarand_b8__s1/evaluations/step_003016/metrics_fast.json
runs/reportgen_orcarand_b8/reportgen_orcarand_b8__s1/evaluations/step_004524/metrics_fast.json
runs/reportgen_orcarand_b8/reportgen_orcarand_b8__s1/evaluations/step_006032/metrics_fast.json
```

The random-control `s2` stage writes to:

```text
runs/reportgen_orcarand_b8/reportgen_orcarand_b8__s2/evaluations/step_XXXXXX/metrics_fast.json
```

## Stopped / intermediate cells

The `b27` and `b64` real-token cells were intentionally stopped after the available evidence showed the middle token budgets did not add useful information for the current comparison.

Their existing outputs remain available in:

```text
runs/reportgen_orca_b27/
runs/reportgen_orca_b64/
```

Some timed-out jobs may have checkpoint files without complete `metrics_fast.json` files. A checkpoint without a corresponding `metrics_fast.json` should be treated as a resumable training state, not as a completed evaluation result.


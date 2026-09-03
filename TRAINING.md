# Training and evaluation

Every command here is plain `python` / `deepspeed` and runs on one GPU. We ran everything through SLURM,
but the submission wrappers were specific to our cluster, so what they wrapped is written out below
instead — adapt the resources to your own scheduler.

Paths are relative to the repository root, and every heavy asset resolves under `data/` and
`checkpoints/` (see [README](README.md) for the download).

---

## 1. Probing read-outs

A probing run is fully described by one YAML under `probe/experiments/`:

```yaml
exp_id: exp_bc3_orca_colipri_b216
encoder: colipri
compression: {method: agglo_organ, budget: 216,
              params: {lam: 0.5, centroid: true, centroid_encoding: sinusoidal,
                       centroid_freqs: 4, centroid_scale: 2.0}}
probe: {families: [disease, size, density, location, radiomics],
        epochs: 10, bs: 128, seeds: [2026, 2027, 2028]}
data: {limit: 0, num_workers: 8}
```

```bash
python probe/run.py         probe/experiments/<exp_id>.yaml   # scores -> results/experiments/<exp_id>/
python probe/run_heldout.py probe/experiments/<exp_id>.yaml   # held-out split variant
python probe/summarize_sweep.py                               # aggregate a sweep into one table
```

`run.py` copies the config next to its results, so `results/experiments/<exp_id>/config.yaml` always
records exactly what produced those numbers.

**Cache the compressor for Ward-based methods.** `agglo_organ` (ORCA) re-derives its merge tree per
volume; without a cache a full-train run recomputes it 24k times and takes hours instead of minutes.

```bash
PROBE_CACHE_ROOT=$PWD/cache python probe/run.py probe/experiments/<exp_id>.yaml
```

`PROBE_CACHE_ROOT` **must be absolute**. A relative value resolves against the working directory and
silently misses the cache — the run still succeeds, just far slower and with no warning.

Our resources: 1× L4, 8 CPU, 64 GB, ≤12 h per experiment.

---

## 2. Report generation

Two stages per cell. Stage 2 warm-starts from stage 1's best epoch, so stage 1 must finish first.

| stage | trains | LR | epochs | steps/epoch |
|---|---|---|---|---|
| `s1` | projector only (`--projector-only`, LoRA frozen) | 5e-4 | 8 | 1,508 |
| `s2` | LoRA **+** projector, warm-started from s1's best epoch | 2e-5 | 8 | 1,508 |

Both evaluate on the **full 1,564-volume validation set after every epoch**, which is where the time
goes: ~0.2 h training vs ~2 h evaluation per epoch, so ~18 h per stage and ~36 h per cell.

```bash
bash llm_engine/run_reportgen.sh --smoke                        # ~15 min wiring check — do this first
bash llm_engine/run_reportgen.sh --method ORCA --budget 216     # both stages, in order
bash llm_engine/run_reportgen.sh --method ORCA --budget 216 --stage s1
```

Under the hood each stage is one `deepspeed` call:

```bash
deepspeed --num_gpus=1 llm_engine/vqa_train.py \
  --reportgen-artifact-manifest data/embeddings/manifest_ORCA_b216.json \
  --token-selection none --token-budget 0 \
  --out-dir <run_dir> --run-name <label> --steps 12064 \
  --train-limit 0 --valid-limit 0 --batch-size 16 --eval-batch-size 1 \
  --gradient-accumulation-steps 1 --num-workers 8 \
  --lr 5e-4 --mm-projector-lr 5e-4 --weight-decay 0.0 --warmup-ratio 0.03 \
  --save-every 1508 --eval-every 1508 --device cuda:0 --seed 2026 \
  --max-new-tokens 512 \
  --model-path llm_engine/base/llava_config --init-from-scratch --projector-only \
  --deepspeed-config llm_engine/zero1_author_reportgen.dsconfig
```

Stage 2 drops `--projector-only`, sets `--lr 2e-5`, and adds
`--init-weights-from-checkpoint <s1_run>/checkpoints/<best_step>`, where the best step comes from

```bash
python llm_engine/pick_best_epoch.py --run_dir <s1_run>   # writes <s1_run>/best.json, by clinical F1
```

Selection is by **clinical F1**, not by a text metric: BLEU and ROUGE reward copying the reference's
phrasing, and only the RadBERT labels speak to whether the compressed tokens kept the findings.

`--steps` is a **cumulative** total and the runner auto-resumes from
`checkpoints/training_state_latest.pt`, so a job that hits a wall-clock limit loses at most one epoch and
re-running the same command picks up where it stopped. That is also how a long cell can be split into
short jobs on a busy queue: ask for `k × 2 × 1508` steps in link *k*.

Our resources: 1× B200, 8 CPU, 120 GB, 24 h per job. Peak GPU was ~40 GB (s1) / ~52 GB (s2), so a 24 GB
card cannot run this cell.

### Do not raise `--eval-batch-size`

It stays at 1. Evaluation is ~10× the training cost, so batching it is the obvious speed-up, and it is
deliberately not taken: generation runs with `padding_side="left"`, and batched left-padded generation
does not fail loudly — it quietly produces slightly worse reports, which lands directly in the clinical
F1 the study is trying to measure.

---

## 3. Scoring

```bash
# clinical F1 + BLEU/ROUGE-L/METEOR/CIDEr + CRG, per epoch (written by the trainer as metrics_fast.json)
python llm_engine/pick_best_epoch.py --run_dir <run_dir>

# GREEN clinical score — one prediction file per job, ~7 h for 1,564 examples on an L4
python -m eval.green.score_green --pred <run_dir>/evaluations/step_XXXXXX/predictions.jsonl \
                                 --out  results/green/<arm>_<budget>_s2_step_XXXXXX
```

GREEN needs `checkpoints/GREEN-RadLlama2-7b` **and** `paraphrase-mpnet-base-v2` present locally, or it
crashes at the very end after doing all the work.

---

## 4. Inference cost profile

Reproduces the prefill / KV-cache / latency numbers:

```bash
python llm_engine/profile_inference.py --budgets 8 27 64 216 1728 13824 --out results_llm/inference_cost.json
```

Multi-pass alternating sweep, median over passes, so a warm-up pass cannot bias one budget.

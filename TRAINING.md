# Training and evaluation

Every command runs on one GPU. Paths are relative to the repository root; heavy assets resolve under `data/` and `checkpoints/` (see [README](README.md) for the download).

---

## 1. Probing

A run is one YAML under `probe/experiments/`:

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
python probe/run.py         probe/experiments/<exp_id>.yaml   # -> results/experiments/<exp_id>/
python probe/run_heldout.py probe/experiments/<exp_id>.yaml   # held-out split variant
python probe/summarize_sweep.py                               # aggregate a sweep into one table
```

**Cache the compressor.** ORCA (`agglo_organ`) re-derives its merge tree per volume, so without a cache a full run recomputes it 24k times — hours instead of minutes.

```bash
PROBE_CACHE_ROOT=$PWD/cache python probe/run.py probe/experiments/<exp_id>.yaml
```

`PROBE_CACHE_ROOT` must be **absolute**. A relative path silently misses the cache: the run still succeeds, just far slower, with no warning.

We used 1× L4, 8 CPU, 64 GB, ≤12 h per experiment.

---

## 2. Report generation

Two stages per cell, stage 2 warm-started from stage 1, so stage 1 must finish first.

| stage | trains | LR | epochs | steps/epoch |
|---|---|---|---|---|
| `s1` | projector only (LoRA frozen) | 5e-4 | 8 | 1,508 |
| `s2` | LoRA + projector, from s1's best epoch | 2e-5 | 8 | 1,508 |

```bash
bash llm_engine/run_reportgen.sh --smoke                        # ~15 min wiring check — do this first
bash llm_engine/run_reportgen.sh --method ORCA --budget 216     # both stages, in order
bash llm_engine/run_reportgen.sh --method ORCA --budget 216 --stage s1
```

Both stages evaluate on the full 1,564-volume validation set after every epoch: ~0.2 h training and ~2 h evaluation per epoch, so ~36 h per cell.

Each stage is one `deepspeed` call:

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

Stage 2 drops `--projector-only`, sets `--lr 2e-5`, and adds `--init-weights-from-checkpoint <s1_run>/checkpoints/<best_step>`, where the best step is chosen by clinical F1:

```bash
python llm_engine/pick_best_epoch.py --run_dir <s1_run>   # writes <s1_run>/best.json
```

`--steps` is a **cumulative** total and the runner auto-resumes from `checkpoints/training_state_latest.pt`. A job that hits its wall-clock limit loses at most one epoch, and a long cell can be split across short jobs by asking for `k × 2 × 1508` steps in link *k*.

**Do not raise `--eval-batch-size`.** Evaluation is ~10× the training cost, so batching it is the obvious speed-up and is deliberately not taken: generation runs with `padding_side="left"`, and batched left-padded generation does not fail loudly — it quietly produces slightly worse reports, which lands directly in the clinical F1 being measured.

We used 1× B200, 8 CPU, 120 GB, 24 h per job. Peak GPU was ~40 GB (s1) and ~52 GB (s2), so a 24 GB card cannot run this cell.

---

## 3. Scoring

[CheapCT](https://github.com/renjie-liang/CheapCT) provides vLLM inference and GREEN. We recommend using it for inference and scoring. The implementations below are what produced the published numbers.

```bash
# clinical F1 + BLEU/ROUGE-L/METEOR/CIDEr + CRG, per epoch (the trainer writes metrics_fast.json)
python llm_engine/pick_best_epoch.py --run_dir <run_dir>

# GREEN clinical score — one prediction file per job, ~7 h for 1,564 examples on an L4
python -m eval.green.score_green --pred <run_dir>/evaluations/step_XXXXXX/predictions.jsonl \
                                 --out  results/green/<arm>_<budget>_s2_step_XXXXXX
```

GREEN needs `checkpoints/GREEN-RadLlama2-7b` **and** `paraphrase-mpnet-base-v2` present locally, or it crashes at the very end after doing all the work.

---

## 4. Inference cost

```bash
python llm_engine/profile_inference.py --budgets 8 27 64 216 1728 13824 --out results_llm/inference_cost.json
```

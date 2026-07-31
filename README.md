# ORCA: ORgan-Centroid Aggregation for Training-Free 3D CT Visual Token Compression

Reference implementation and results for the paper *ORCA: ORgan-Centroid Aggregation for
Training-Free 3D CT Visual Token Compression*.

![ORCA overview](overview.png)

A 3D CT scan produces thousands to tens of thousands of visual tokens, and they must be compressed
before a language model can consume them. **ORCA** is a training-free compressor: it merges neighbouring
tokens into connected, organ-guided regions and writes each region's centroid position back into the
token, preserving the anatomical evidence that grid pooling blends away. It is plug-and-play, giving an
adjustable token budget with no model change, attention hook, extra supervision, or text query.

## Main results

Grid average vs. ORCA at $B{=}216$ across all three evaluation surfaces (CT-RATE / COLIPRI; higher is better):

| task | measure | Grid average | ORCA |
|---|---|---|---|
| Probing (AUROC / $R^2$) | disease  | 0.851 | **0.852** |
|                         | size     | 0.681 | **0.720** |
|                         | density  | 0.865 | **0.913** |
|                         | location | 0.247 | **0.677** |
|                         | texture  | 0.760 | **0.816** |
| Measurement VQA (acc.)  | size     | 0.766 | **0.778** |
|                         | density  | 0.807 | **0.847** |
|                         | location | 0.649 | **0.721** |
|                         | texture  | 0.795 | **0.838** |
| Report generation       | CE-F1    | **0.475** | 0.470 |
|                         | CRG      | **0.438** | 0.436 |
|                         | GREEN    | 0.319 | **0.329** |

ORCA's advantage is largest where localized detail matters: location probing (0.68 vs 0.25) and VQA
(0.72 vs 0.65). Report-generation metrics are near-saturated across compressors; ORCA leads on GREEN,
the strongest clinical score. ORCA also shrinks the LLM visual context $64\times$, its KV-cache $50\times$,
and per-volume latency $31\times$. Numbers for every encoder, attribute, and budget are under `results/`.

## Code and results

- **Core method code** — Grid average and ORCA (`probe/compressors/`; ORCA is `branch_c/agglo_organ.py`).
- **Downstream training and evaluation** — the probing read-outs, the LLaVA-style VQA / report-generation
  pipeline, and the GREEN clinical-report metric (`probe/`, `llm_engine/`, `eval/green/`).
- **All results** — the metrics behind every table and figure, for every compressor (`results/`, `results_llm/`).

Everything needed to reproduce and verify the numbers is included.

## Data and checkpoints

Heavy assets are not committed. Place two folders at the repository root (a download bundle will be
released on Hugging Face), and the code resolves all paths relatively:

```
data/          encoder token embeddings, organ masks, CT volumes, VQA question sets
checkpoints/   encoder weights, the LLM base checkpoint, GREEN grader weights
```

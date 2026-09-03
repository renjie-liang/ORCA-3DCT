# ORCA: ORgan-Centroid Aggregation for Training-Free 3D CT Visual Token Compression

Reference implementation, released embeddings, and results for
[*ORCA: ORgan-Centroid Aggregation for Training-Free 3D CT Visual Token Compression*](https://arxiv.org/abs/2608.00345).

![ORCA overview](overview.png)

A 3D CT scan produces thousands to tens of thousands of visual tokens, and they must be compressed before a
language model can consume them. **ORCA** is a training-free compressor: it merges neighbouring tokens into
connected, organ-guided regions and writes each region's centroid position back into the token, preserving
the anatomical evidence that grid pooling blends away. It is plug-and-play, giving an adjustable token budget
with no model change, attention hook, extra supervision, or text query.

---

## 📦 Released embeddings

**The uncompressed encoder outputs are the main thing we are contributing here.** Encoding all of CT-RATE
with a 3D vision–language encoder costs several hundred GPU-hours, and nobody publishes the result — so
every group that wants to work on 3D CT token compression, pooling, retrieval, or probing pays that cost
again from scratch. These are those outputs, released so you do not have to.

The ORCA and Grid-average token bundles are a derived layer on top: useful if you want our exact inputs,
but the uncompressed grids are what let you run **your own** method.

All arrays are `float16`, one row per volume, aligned to a sibling `*_ids.txt`. Hosted at
**[🤗 LiangRenjie/ORCA-3DCT](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT)**.

### CT-RATE

| Encoder | Uncompressed | ORCA `B=8` | `B=27` | `B=64` | `B=216` | Grid average |
|---|---|---|---|---|---|---|
| **CoLiPri** <br><sub>[Wald et al. 2025](https://arxiv.org/abs/2510.15042)</sub> | 24×24×24×768<br>**13,824 tok** · 545 GB<br>[⬇](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/uncompressed/colipri) | 792-d · 0.33 GB<br>[⬇](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/ORCA_b8) | 1.10 GB<br>[⬇](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/ORCA_b27) | 2.60 GB<br>[⬇](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/ORCA_b64) | 8.79 GB<br>[⬇](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/ORCA_b216) | 768-d, all 4 budgets<br>[⬇](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri) |
| **CT-CLIP** <br><sub>[Hamamcı et al. 2026](https://doi.org/10.1038/s41551-025-01599-y)</sub> | 24×24×24×512<br>**13,824 tok** · 710 GB<br><sub>batch 2</sub> | 536-d · 0.43 GB<br><sub>batch 2</sub> | 1.45 GB | 3.44 GB | 13.03 GB | 512-d, all 4 budgets |
| **HLIP** <br><sub>[Towards Scalable Language-Image Pre-training for 3D Medical Imaging](https://arxiv.org/abs/2505.21862)</sub> | 8×14×14×1024<br>**1,568 tok** · 161 GB<br><sub>batch 2</sub> | — | — | — | — | — |
| **FVLM** <br><sub>[Shui et al., ICLR 2025](https://arxiv.org/abs/2501.14548)</sub> | pooled 256-d<br>global + 4 organs · 0.3 GB<br>[⬇](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/uncompressed/fvlm) | — | — | — | — | — |
| **ViSD-Boost** <br><sub>[Cao et al. 2025](https://arxiv.org/abs/2508.03742)</sub> | pooled 256-d<br>global + 4 organs · 0.4 GB<br>[⬇](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/uncompressed/visd_boost) | — | — | — | — | — |

**Coverage.** CT-CLIP and HLIP cover all of CT-RATE (47,149 train / 3,039 valid). CoLiPri covers 24,128 /
1,564 — that is the encoder's own coverage of the corpus, not a subsetting choice of ours.

**Also released:** the TotalSegmentator-derived organ occupancy maps on the CoLiPri grid (0.3 GB), which is
all you need to run ORCA yourself on the uncompressed grids, plus the CT-RATE reports, the 18 abnormality
labels, and the report-generation question set.

### Loading

These are raw arrays, not a tabular dataset — `load_dataset()` and the dataset viewer do not apply. Use
`hf download` and memory-map:

```python
import numpy as np
from huggingface_hub import snapshot_download

d = snapshot_download("LiangRenjie/ORCA-3DCT", repo_type="dataset",
                      allow_patterns="compressed/colipri/ORCA_b216/*")
tokens = np.load(f"{d}/compressed/colipri/ORCA_b216/valid.npy", mmap_mode="r")  # (1564, 216, 792) fp16
ids    = open(f"{d}/compressed/colipri/ORCA_b216/valid_ids.txt").read().split()
tokens[ids.index("valid_1000_a_2")]        # -> (216, 792), ready for a projector
```

Row *i* of the array is line *i* of the id file. The uncompressed grids are one `.npy` per volume instead,
named by volume id.

---

## 📊 Main results

Grid average vs. ORCA at $B{=}216$ (CT-RATE / CoLiPri; higher is better):

| task | measure | Grid average | ORCA |
|---|---|---|---|
| Probing (AUROC / $R^2$) | disease  | 0.851 | **0.852** |
|                         | size     | 0.681 | **0.720** |
|                         | density  | 0.865 | **0.913** |
|                         | location | 0.247 | **0.677** |
|                         | texture  | 0.760 | **0.816** |
| Report generation       | CE-F1    | **0.475** | 0.470 |
|                         | CRG      | **0.438** | 0.436 |
|                         | GREEN    | 0.319 | **0.329** |

ORCA's advantage is largest where localized detail matters: location probing (0.68 vs 0.25). Report-generation
metrics are near-saturated across compressors; ORCA leads on GREEN, the strongest clinical score. ORCA also
shrinks the LLM visual context $64\times$, its KV-cache $50\times$, and per-volume latency $31\times$. Numbers
for every encoder, attribute, and budget are under `results/` and `results_llm/`.

The measurement-VQA surface reported in the paper is maintained separately, in
[**CheapCT**](https://github.com/renjie-liang/CheapCT) ([paper](https://arxiv.org/abs/2607.22771)); the raw
per-run metrics for those cells are kept here under `results_llm/` for completeness.

---

## 🚀 Quickstart

```bash
git clone https://github.com/renjie-liang/ORCA-3DCT.git && cd ORCA-3DCT

conda env create -f environment.yml && conda activate orca3dct   # python 3.12

python download.py --bundle reportgen --budget 216               # ~35 GB: tokens + labels + base weights
bash llm_engine/run_reportgen.sh --smoke                         # ~15 min wiring check
bash llm_engine/run_reportgen.sh --method ORCA --budget 216      # the real cell (~36 h on one B200)
```

`download.py --list` shows every bundle and its size before you commit the disk. See
**[TRAINING.md](TRAINING.md)** for the full recipe, costs, and the traps worth knowing.

### Environment

Only three pins matter; take whatever else your CUDA stack prefers.

| pinned | why |
|---|---|
| `python=3.12` | the engine uses 3.10+ syntax |
| `transformers>=4.50,<5` | `LlamaAttention.forward` changed shape across 4.x; we ran 4.57 |
| `deepspeed>=0.14` | the ZeRO-1 config uses post-0.14 keys |

If `flash-attn` will not build, drop it — the engine falls back to sdpa and only speed is lost.

---

## 🗂 Repository layout

```
probe/            compressors + probing read-outs
                  compressors/branch_c/agglo_organ.py   <- ORCA itself
                  experiments/<exp_id>.yaml             <- the 228 configs behind results/
llm_engine/       LLaVA-style report-generation trainer
                  run_reportgen.sh   <- the entry point
                  base/llava_config/ <- architecture config (no weights)
                  llava/             <- the vendored LLaVA decoder
eval/green/       the GREEN clinical-report metric
figures/          figure code + rendered PDFs
results/          probing scores, GREEN scores, significance tests
results_llm/      report-generation and VQA per-run metrics
data/             downloaded — embeddings, labels, reports, organ masks
checkpoints/      downloaded — base weights and our trained checkpoints
```

Heavy assets are not committed; `download.py` places them where the code expects.

---

## ⚖️ License, attribution, and terms

The **code** in this repository is released under [Apache-2.0](LICENSE). `llm_engine/llava/` is the LLaVA
decoder (Apache-2.0, Copyright 2023 Haotian Liu), reached here via the BTB3D fork at commit `0eeb6e6`; the
files carry their original headers and our changes are marked in place.

The **released embeddings are derived from [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)**
and inherit its terms: **CC-BY-NC-SA-4.0**, academic and research use only, no commercial use, no
re-identification. The Hugging Face dataset is gated for the same reason CT-RATE is. If you use them, cite
CT-RATE and the encoder whose outputs you used, alongside ORCA.

```bibtex
@article{orca2026,
  title   = {ORCA: ORgan-Centroid Aggregation for Training-Free 3D CT Visual Token Compression},
  journal = {arXiv preprint arXiv:2608.00345},
  year    = {2026}
}
```

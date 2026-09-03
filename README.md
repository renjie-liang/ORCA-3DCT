# ORCA: ORgan-Centroid Aggregation for Training-Free 3D CT Visual Token Compression

[📄 arXiv](https://arxiv.org/abs/2608.00345) · [💻 GitHub](https://github.com/renjie-liang/ORCA-3DCT) · [🤗 Models & Data](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT)

![ORCA overview](overview.png)

A 3D CT scan produces thousands to tens of thousands of visual tokens, and they must be compressed before a language model can consume them. **ORCA** is a training-free compressor: it merges neighbouring tokens into connected, organ-guided regions and writes each region's centroid position back into the token, preserving the anatomical evidence that grid pooling blends away. It is plug-and-play, giving an adjustable token budget with no model change, attention hook, extra supervision, or text query.

---

## Released embeddings

We provide both compressed and uncompressed embeddings at **[ORCA-3DCT](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT)**. Each encoder's outputs were obtained either by running its authors' released checkpoint or by reproducing the encoder from its paper.

We also release the **organ segmentation** resampled onto each encoder's token grid: for every volume, an `(11, *grid)` float16 array giving the fraction of each token occupied by each of eleven thoracic structures — lung, airway, heart, aorta, central vessels, mediastinum, hiatus, pleura, chest-wall bone, thoracic spine, upper abdomen. Soft occupancy in [0, 1], not a hard label. Match the set to the encoder you are using: encoders resample and crop the volume differently, so a segmentation built for one grid does not align with another.

All arrays are `float16`, one row per volume, aligned to a sibling `*_ids.txt`.

### CT-RATE

| Encoder | ORCA `B=216` | ORCA `B=64` | ORCA `B=27` | ORCA `B=8` | Grid avg `B=216` | Grid avg `B=64` | Grid avg `B=27` | Grid avg `B=8` | Uncompressed | Organ segmentation |
|---|---|---|---|---|---|---|---|---|---|---|
| [CoLiPri](https://arxiv.org/abs/2510.15042) | [216×792 · 8.79 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/ORCA_b216) | [64×792 · 2.60 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/ORCA_b64) | [27×792 · 1.10 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/ORCA_b27) | [8×792 · 0.33 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/ORCA_b8) | [216×768 · 8.53 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/GridAvg_b216) | [64×768 · 2.53 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/GridAvg_b64) | [27×768 · 1.07 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/GridAvg_b27) | [8×768 · 0.32 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/GridAvg_b8) | [24×24×24×768 · 545 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/uncompressed/colipri) | [11×24×24×24 · 0.30 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/organ_masks/colipri) |
| [CT-CLIP](https://doi.org/10.1038/s41551-025-01599-y) | 216×536 · 13.0 GB | 64×536 · 3.44 GB | 27×536 · 1.45 GB | 8×536 · 0.43 GB | 216×512 · 12.4 GB | 64×512 · 3.29 GB | 27×512 · 1.39 GB | 8×512 · 0.41 GB | 24×24×24×512 · 710 GB | [11×24×24×24 · 0.83 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/organ_masks/ctclip) |
| [FVLM](https://arxiv.org/abs/2501.14548) | — | — | — | — | — | — | — | — | [5×256 · 0.3 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/uncompressed/fvlm) | — |
| [ViSD-Boost](https://arxiv.org/abs/2508.03742) | — | — | — | — | — | — | — | — | [6×256 · 0.4 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/uncompressed/visd_boost) | — |

Cells without a link are computed but not uploaded yet — the CoLiPri rows are complete.

**Coverage.** CT-CLIP covers all of CT-RATE (47,149 train / 3,039 valid). CoLiPri covers 24,128 / 1,564 — that is the encoder's own coverage of the corpus, not a subsetting choice of ours.

The segmentation comes from TotalSegmentator; with the uncompressed grids it is all you need to run ORCA yourself.

**What we do not mirror.** Only the arrays we computed are hosted here. The reports, the 18 abnormality labels and the report-generation question set are CT-RATE's own files, so take them from [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) directly — accept its terms once, then `python download.py --annotations` puts these four where the code expects them:

| CT-RATE path | lands at | |
|---|---|---|
| `dataset/vqa/{train,valid}_vqa.json` | `data/vqa/{train,valid}_reportgen.json` | 1.2 GB / 37 MB |
| `dataset/radiology_text_reports/validation_reports.csv` | `data/reports/` | 5 MB |
| `dataset/multi_abnormality_labels/valid_predicted_labels.csv` | `data/labels/` | 0.2 MB |

The question set is CT-CHAT's multi-task VQA; the trainer keeps only its `report_generation` records, so no preprocessing is needed.

### Loading

These are raw arrays, not a tabular dataset — `load_dataset()` and the dataset viewer do not apply. Use `hf download` and memory-map:

```python
import numpy as np
from huggingface_hub import snapshot_download

d = snapshot_download("LiangRenjie/ORCA-3DCT", repo_type="dataset",
                      allow_patterns="compressed/colipri/ORCA_b216/*")
tokens = np.load(f"{d}/compressed/colipri/ORCA_b216/valid.npy", mmap_mode="r")  # (1564, 216, 792) fp16
ids    = open(f"{d}/compressed/colipri/ORCA_b216/valid_ids.txt").read().split()
tokens[ids.index("valid_1000_a_2")]        # -> (216, 792), ready for a projector
```

Row *i* of the array is line *i* of the id file. The uncompressed grids are one `.npy` per volume instead, named by volume id.

---

## Main results

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

ORCA's advantage is largest where localized detail matters: location probing (0.68 vs 0.25). Report-generation metrics are near-saturated across compressors; ORCA leads on GREEN, the strongest clinical score. ORCA also shrinks the LLM visual context $64\times$, its KV-cache $50\times$, and per-volume latency $31\times$. Numbers for every encoder, attribute, and budget are under `results/` and `results_llm/`.

The measurement-VQA surface reported in the paper is maintained separately, in [**CheapCT**](https://github.com/renjie-liang/CheapCT) ([paper](https://arxiv.org/abs/2607.22771)); the raw per-run metrics for those cells are kept here under `results_llm/` for completeness.

---

## Quickstart

```bash
git clone https://github.com/renjie-liang/ORCA-3DCT.git && cd ORCA-3DCT

conda env create -f environment.yml && conda activate orca3dct   # python 3.12

python download.py --bundle reportgen --budget 216               # 8.8 GB of ORCA tokens
python download.py --annotations --base-weights                  # CT-RATE text + Llama-3.1-8B
bash llm_engine/run_reportgen.sh --smoke                         # ~15 min wiring check
bash llm_engine/run_reportgen.sh --method ORCA --budget 216      # the real cell (~36 h on one B200)
```

`download.py --list` shows every bundle and its size before you commit the disk. See **[TRAINING.md](TRAINING.md)** for the full recipe, costs, and the traps worth knowing.

**For anything beyond a single cell, use [CheapCT](https://github.com/renjie-liang/CheapCT)'s vLLM inference and GREEN.** Evaluation, not training, is what this study costs — about ten times the training it follows — and the vLLM path is far faster than the reference implementations kept here.

### Environment

Only three pins matter; take whatever else your CUDA stack prefers.

| pinned | why |
|---|---|
| `python=3.12` | the engine uses 3.10+ syntax |
| `transformers>=4.50,<5` | `LlamaAttention.forward` changed shape across 4.x; we ran 4.57 |
| `deepspeed>=0.14` | the ZeRO-1 config uses post-0.14 keys |

If `flash-attn` will not build, drop it — the engine falls back to sdpa and only speed is lost.

---

## Repository layout

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

## License, attribution, and terms

The **code** in this repository is released under [Apache-2.0](LICENSE). `llm_engine/llava/` is the LLaVA decoder (Apache-2.0, Copyright 2023 Haotian Liu), reached here via the BTB3D fork at commit `0eeb6e6`; the files carry their original headers and our changes are marked in place.

The **released embeddings are derived from [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)** and inherit its terms: **CC-BY-NC-SA-4.0**, academic and research use only, no commercial use, no re-identification. Downloading them means accepting CT-RATE's terms as well as this license. If you use them, cite CT-RATE and the encoder whose outputs you used, alongside ORCA.

```bibtex
@article{orca2026,
  title   = {ORCA: ORgan-Centroid Aggregation for Training-Free 3D CT Visual Token Compression},
  journal = {arXiv preprint arXiv:2608.00345},
  year    = {2026}
}
```

# ORCA: ORgan-Centroid Aggregation for Training-Free 3D CT Visual Token Compression

[📄 arXiv](https://arxiv.org/abs/2608.00345) · [💻 GitHub](https://github.com/renjie-liang/ORCA-3DCT) · [🤗 Models & Data](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT)

![ORCA overview](overview.png)

A 3D CT scan produces thousands to tens of thousands of visual tokens, and they must be compressed before a language model can consume them. **ORCA** is a training-free compressor: it merges neighbouring tokens into connected, organ-guided regions and writes each region's centroid position back into the token, preserving the anatomical evidence that grid pooling blends away. It is plug-and-play, giving an adjustable token budget with no model change, attention hook, extra supervision, or text query.

---

## Released embeddings

We provide both compressed and uncompressed embeddings at **[ORCA-3DCT](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT)**. Each encoder's outputs were obtained either by running its authors' released checkpoint or by reproducing the encoder from its paper.

We also release the organ segmentation resampled from TotalSegmentator onto each encoder's token grid. Use the matched set: encoders resample and crop the volume differently, so a segmentation built for one grid does not align with another.

### CT-RATE

| Encoder | ORCA `B=216` | ORCA `B=64` | ORCA `B=27` | ORCA `B=8` | Grid avg `B=216` | Grid avg `B=64` | Grid avg `B=27` | Grid avg `B=8` | Uncompressed | Organ segmentation |
|---|---|---|---|---|---|---|---|---|---|---|
| [COLIPRI](https://arxiv.org/abs/2510.15042) | [216×792 · 8.79 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/colipri_ORCA_b216_d792_lam0p5) | [64×792 · 2.60 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/colipri_ORCA_b64_d792_lam0p5) | [27×792 · 1.10 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/colipri_ORCA_b27_d792_lam0p5) | [8×792 · 0.33 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/colipri_ORCA_b8_d792_lam0p5) | [216×768 · 8.53 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/colipri_GridAvg_b216_d768) | [64×768 · 2.53 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/colipri_GridAvg_b64_d768) | [27×768 · 1.07 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/colipri_GridAvg_b27_d768) | [8×768 · 0.32 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/compressed/colipri/colipri_GridAvg_b8_d768) | [24×24×24×768 · 545 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/uncompressed/colipri) | [11×24×24×24 · 0.30 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/organ_masks/colipri) |
| [CT-CLIP](https://doi.org/10.1038/s41551-025-01599-y) | 216×536 · 13.0 GB | 64×536 · 3.44 GB | 27×536 · 1.45 GB | 8×536 · 0.43 GB | 216×512 · 12.4 GB | 64×512 · 3.29 GB | 27×512 · 1.39 GB | 8×512 · 0.41 GB | 24×24×24×512 · 710 GB | [11×24×24×24 · 0.83 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/organ_masks/ctclip) |
| [ViSD-Boost + CT-CLIP](https://arxiv.org/abs/2508.03742) | — | — | — | — | — | — | — | — | [5×256 · 0.13 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/uncompressed/visd_boost_ctclip/visd_boost_ctclip_native_b5_d256) | — |
| [ViSD-Boost](https://arxiv.org/abs/2508.03742) | — | — | — | — | — | — | — | — | [4×256 · 0.10 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/uncompressed/visd_boost/visd_boost_native_b4_d256) | — |
| [FVLM](https://arxiv.org/abs/2501.14548) | — | — | — | — | — | — | — | — | [4×256 · 0.10 GB](https://huggingface.co/datasets/LiangRenjie/ORCA-3DCT/tree/main/uncompressed/fvlm/fvlm_native_b4_d256) | — |

Cells without a link are computed but not uploaded yet — the COLIPRI rows are complete.

**Coverage.** CT-CLIP covers all of CT-RATE (47,149 train / 3,039 valid). COLIPRI covers 24,128 / 1,564, because its authors take one reconstruction per scan to be sufficient; CT-RATE ships several reconstructions of the same study.

### Loading

```python
import numpy as np
from huggingface_hub import snapshot_download

d = snapshot_download("LiangRenjie/ORCA-3DCT", repo_type="dataset",
                      allow_patterns="compressed/colipri/colipri_ORCA_b216_d792_lam0p5/*")
tokens = np.load(f"{d}/compressed/colipri/colipri_ORCA_b216_d792_lam0p5/valid.npy", mmap_mode="r")  # (1564, 216, 792) fp16
ids    = open(f"{d}/compressed/colipri/colipri_ORCA_b216_d792_lam0p5/valid_ids.txt").read().split()
tokens[ids.index("valid_1000_a_2")]        # -> (216, 792), ready for a projector
```

---

## Main results

Grid average vs. ORCA at $B{=}216$ (CT-RATE / COLIPRI; higher is better):

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

ORCA wins every probing attribute and leads on GREEN, the strongest clinical effecacy; the report-generation text metrics are near-saturated across compressors. It also shrinks the LLM visual context $64\times$. Full results are under `results/` and `results_llm/`.

---

## Quickstart

```bash
git clone https://github.com/renjie-liang/ORCA-3DCT.git && cd ORCA-3DCT

conda env create -f environment.yml && conda activate orca3dct   # python 3.12

python download.py --bundle reportgen --budget 216   # 8.8 GB of ORCA tokens

# both are gated — accept the terms on the Hub first
#   https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
#   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
python download.py --annotations                    # CT-RATE reports and labels
python download.py --base-weights                   # Llama-3.1-8B

bash llm_engine/run_reportgen.sh --smoke                    # ~15 min wiring check
bash llm_engine/run_reportgen.sh --method ORCA --budget 216 # the real cell (~36 h on one B200)
```

See **[TRAINING.md](TRAINING.md)** for the full pipeline and the traps worth knowing.

[CheapCT](https://github.com/renjie-liang/CheapCT) provides vLLM inference and GREEN. We recommend using it for inference and scoring.

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

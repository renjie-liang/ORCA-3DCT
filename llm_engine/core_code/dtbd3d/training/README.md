# DTBD3D BTB3D-Style Training Pipeline

This package trains the report-generation `mm_projector` plus LLaMA LoRA adapters from exported DTBD3D report-generation artifacts.

## Layout

- `data/`: memory-mapped `tokens_int.npy` plus `codebook.npy` loading, report-generation prompt encoding, and collation.
- `models/`: LLaMA + LoRA + BTB3D visual projector wrapper.
- `train/`: manual file-logged training loop and periodic evaluation callback.
- `configs/sub_task9_reportgen_smoke_*.yaml`: short single-GPU smoke configs.
- `configs/sub_task9_reportgen_subset9571_4epoch_*_nozero_eval128.yaml`: current comparison configs.

## Launch

```bash
cd .
export PYTHONPATH="$PWD/Experiment/core_code:${PYTHONPATH:-}"
SETTING=c025 bash Experiment/scripts/sub_task9_reportgen_smoke_train.sh
```

The run directory is created under:

```text
Experiment/runs/btb3d_repro_generator/{phase}_{date}/{full_hp_string}_{seed}_{date}_{HHMM}/
```

## Notes

- No MLflow, wandb, Hydra, or TensorBoard are used.
- Checkpoints are saved in adapter/HF-style safetensors files, not `.pth`.
- `max_text_length=8192` limits tokenized text. The 31,744 visual LFQ vectors are inserted as `inputs_embeds`, matching the released BTB3D inference path instead of expanding them into tokenizer ids.
- `max_visual_tokens` subsamples the 31,744 projected visual tokens before LLaMA training to keep the LLaMA context and activation memory bounded.
- Training consumes `reportgen_artifact/manifest.json`; channel order is validated from the exported codebook metadata.

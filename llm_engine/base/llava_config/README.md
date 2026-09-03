# `llava_config/` — architecture config only, no weights

`vqa_train.py` is always launched with `--init-from-scratch`, so **every weight comes from stock
Llama-3.1-8B-Instruct**; nothing is loaded from this directory. The two JSON files here supply only
the architecture description that the stock Llama config does not carry:

| field | value | why the stock Llama config cannot be used |
|---|---|---|
| `model_type` | `llava_llama` | selects `LlavaLlamaForCausalLM`, not `LlamaForCausalLM` |
| `mm_projector_type` | `attn_pool+mlp2x_gelu` | the visual projector this study trains |
| `mm_context_size` | 18 | attention-pool query count |
| `vocab_size` / `pad_token_id` | 128261 / 128256 | the 5 added visual placeholder tokens |
| `use_mm_proj`, `mm_patch_merge_type`, `image_aspect_ratio` | — | required by the LLaVA forward path |

`mm_hidden_size` is a **placeholder**. The real per-token visual dimension is taken from the token
manifest at run time (`visual_dim`: 768 for Grid average, 792 for ORCA) and passed as
`mm_hidden_size_override`, so one config serves every arm and budget.

`adapter_config.json` is the LoRA template (r=128, alpha=256, dropout=0.05 over the seven
q/k/v/o/gate/up/down projections). Stage 1 (`--projector-only`) freezes it; stage 2 trains it.

Provenance: extracted from the report-generation ORCA b=216 stage-2 checkpoint; the original
`checkpoint-38000` directory this study was launched from held nothing else that was ever read.

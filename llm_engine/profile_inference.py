#!/usr/bin/env python3
"""T6b downstream cost: LLM inference speed/memory as a function of visual-token budget B.

The point of token compression is a cheaper downstream LLM. This measures exactly that: for the paper's
LLaVA backbone (Llama-3.1-8B-Instruct: hidden 4096, 32 layers, 32 heads, 8 KV heads/GQA, head_dim 128), feed
B visual tokens + a fixed prompt and time prefill + decode, record peak memory, and report the KV-cache size.

Cost depends ONLY on sequence length, not on token content or the LoRA/projector weights, so we time the exact
Llama-3.1-8B architecture with random init (no download, no auth, fully reproducible). Numbers are latency and
memory, not text quality. attn = flash_attention_2 if available (production), else sdpa.

    python profile_inference.py            # writes results_llm/t6b_inference_cost.json + prints a table
"""
import json, time, argparse, statistics
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--budgets", default="8,27,64,216,13824")
ap.add_argument("--prompt", type=int, default=64)       # text prompt tokens accompanying the visual tokens
ap.add_argument("--gen", type=int, default=128)         # decoded tokens (report-length)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--passes", type=int, default=4)          # total sweep passes; direction alternates each pass
ap.add_argument("--warmup-passes", type=int, default=1)   # leading passes discarded (global cold-start / autotune)
ap.add_argument("--tag", default="")                      # freeform label recorded in meta (e.g. GPU name)
ap.add_argument("--out", default="./results_llm/t6b_inference_cost.json")
a = ap.parse_args()

DEV = "cuda"
DT = torch.bfloat16
# Llama-3.1-8B-Instruct architecture (from llm_engine/base/checkpoint-38000/config.json)
cfg = LlamaConfig(hidden_size=4096, intermediate_size=14336, num_hidden_layers=32, num_attention_heads=32,
                  num_key_value_heads=8, vocab_size=128256, max_position_embeddings=131072, rope_theta=500000.0)
KV_BYTES_PER_TOKEN = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2  # K+V, bf16

ATTN = "sdpa"
try:
    cfg._attn_implementation = ATTN
except Exception:
    pass
model = LlamaForCausalLM(cfg).to(DEV, DT).eval()
print(f"[model] Llama-3.1-8B ({sum(p.numel() for p in model.parameters())/1e9:.1f}B params) attn={ATTN} dtype={DT}", flush=True)


def sync(): torch.cuda.synchronize()


@torch.no_grad()
def measure(B):
    seq = B + a.prompt
    emb = torch.randn(1, seq, cfg.hidden_size, device=DEV, dtype=DT)
    # -------- prefill (time to first token: forward over the full visual+prompt sequence) --------
    for _ in range(2):  # warmup
        model(inputs_embeds=emb, use_cache=True)
    sync()
    pref = []
    for _ in range(a.reps):
        torch.cuda.reset_peak_memory_stats(); sync(); t0 = time.perf_counter()
        out = model(inputs_embeds=emb, use_cache=True)
        sync(); pref.append((time.perf_counter() - t0) * 1e3)
    prefill_ms = sorted(pref)[len(pref) // 2]
    peak_prefill = torch.cuda.max_memory_allocated() / 1e9
    # -------- decode (autoregressive, KV-cached): median ms/token over `gen` steps --------
    past = out.past_key_values
    tok = torch.randn(1, 1, cfg.hidden_size, device=DEV, dtype=DT)
    for _ in range(2):
        model(inputs_embeds=tok, past_key_values=past, use_cache=True)
    sync(); t0 = time.perf_counter()
    p = past
    for _ in range(a.gen):
        o = model(inputs_embeds=tok, past_key_values=p, use_cache=True); p = o.past_key_values
    sync(); decode_ms_tok = (time.perf_counter() - t0) * 1e3 / a.gen
    peak_total = torch.cuda.max_memory_allocated() / 1e9
    kv_mb = KV_BYTES_PER_TOKEN * seq / 1e6
    total_s = (prefill_ms + decode_ms_tok * a.gen) / 1e3
    return dict(B=B, seq=seq, prefill_ms=round(prefill_ms, 1), decode_ms_per_tok=round(decode_ms_tok, 2),
                kv_cache_MB=round(kv_mb, 1), peak_mem_GB=round(peak_total, 2),
                total_s_per_report=round(total_s, 2))


# Multi-pass sweep: each pass walks the budgets in ALTERNATING order (lo->hi, hi->lo, ...) so every B is
# measured both early (cold) and late (warm) across passes; leading warmup passes are discarded, and each
# reported cell is the MEDIAN across the effective passes. This cancels global cold-start / autotune / thermal
# ordering effects that a single fixed-order sweep bakes into whichever B happens to run first.
budgets = [int(b) for b in a.budgets.split(",")]
by_B, oom_B = {}, set()
for p in range(a.passes):
    order = budgets if p % 2 == 0 else budgets[::-1]      # alternate direction each pass
    for _B in order:
        if _B in oom_B:
            continue
        try:
            r = measure(_B)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); oom_B.add(_B)
            print(f"[OOM] B={_B} did not fit on this GPU (peak > device memory)", flush=True); continue
        if p >= a.warmup_passes:                          # discard leading warmup pass(es)
            by_B.setdefault(_B, []).append(r)
    print(f"[pass {p+1}/{a.passes} {'WARMUP-discard' if p < a.warmup_passes else 'effective'} "
          f"dir={'lo->hi' if p % 2 == 0 else 'hi->lo'}] done", flush=True)

rows = []
for _B in budgets:                                        # report in budget order, not sweep order
    if _B in oom_B:
        rows.append(dict(B=_B, seq=_B + a.prompt, oom=True)); continue
    rs = by_B[_B]
    pf = [x["prefill_ms"] for x in rs]; dm = [x["decode_ms_per_tok"] for x in rs]
    pk = [x["peak_mem_GB"] for x in rs]; ts = [x["total_s_per_report"] for x in rs]
    rows.append(dict(B=_B, seq=_B + a.prompt,
        prefill_ms=round(statistics.median(pf), 1), decode_ms_per_tok=round(statistics.median(dm), 2),
        kv_cache_MB=rs[0]["kv_cache_MB"], peak_mem_GB=round(statistics.median(pk), 2),
        total_s_per_report=round(statistics.median(ts), 2),
        n_eff=len(rs), prefill_ms_passes=[round(x, 1) for x in pf], total_s_passes=[round(x, 2) for x in ts]))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
meta = dict(model="Llama-3.1-8B (arch-exact, random init for timing)", attn=ATTN, prompt_tokens=a.prompt,
            gen_tokens=a.gen, kv_bytes_per_token=KV_BYTES_PER_TOKEN,
            passes=a.passes, warmup_passes=a.warmup_passes, tag=a.tag, rows=rows)
json.dump(meta, open(a.out, "w"), indent=2)
print(f"\n{'B':>6} {'seq':>6} {'prefill_ms':>11} {'ms/tok':>7} {'KV_MB':>8} {'peak_GB':>8} {'s/report':>9}")
for r in rows:
    if r.get("oom"):
        print(f"{r['B']:>6} {r['seq']:>6} {'OOM (did not fit)':>40}"); continue
    print(f"{r['B']:>6} {r['seq']:>6} {r['prefill_ms']:>11} {r['decode_ms_per_tok']:>7} "
          f"{r['kv_cache_MB']:>8} {r['peak_mem_GB']:>8} {r['total_s_per_report']:>9}")
b0 = rows[-1]  # largest budget (uncompressed if present)
bl = next((r for r in rows if r["B"] == 216 and not r.get("oom")), None)  # B=216 reference for the ratio line
if bl is None or b0 is bl:
    print("wrote", a.out); import sys; sys.exit(0)
if b0.get("oom"):
    print(f"\nuncompressed (13824) did NOT fit this GPU; compressed B={bl['B']} runs at "
          f"{bl['peak_mem_GB']:.1f} GB peak -- compression ENABLES this GPU.")
else:
    print(f"\ncompression 13824->216: prefill {b0['prefill_ms']/bl['prefill_ms']:.1f}x faster, "
          f"KV {b0['kv_cache_MB']/bl['kv_cache_MB']:.0f}x smaller, peak {b0['peak_mem_GB']/bl['peak_mem_GB']:.1f}x less")
print("wrote", a.out)

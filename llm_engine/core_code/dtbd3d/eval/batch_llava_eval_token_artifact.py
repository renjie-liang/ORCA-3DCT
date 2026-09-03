#!/usr/bin/env python3
"""Batch BTB3D/LLaVA inference from DTBD3D token artifacts.

This is the batch-generation counterpart to
`multigpu_llava_eval_token_artifact.py`. It keeps the same raw BTB3D JSONL
output schema, but processes multiple report-generation prompts in one
`model.generate()` call on a single GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    from dtbd3d.core.artifact import open_matrix
    from dtbd3d.core.visual_embedding import load_reportgen_codebook, materialize_reportgen_image
    from dtbd3d.eval.multigpu_llava_eval_token_artifact import (
        convention_from_flags,
        image_output_name,
        volume_id_from_image,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dtbd3d.core.artifact import open_matrix
    from dtbd3d.core.visual_embedding import load_reportgen_codebook, materialize_reportgen_image
    from dtbd3d.eval.multigpu_llava_eval_token_artifact import (
        convention_from_flags,
        image_output_name,
        volume_id_from_image,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--token_dir", required=True, help="Token artifact directory.")
    parser.add_argument("--compression", choices=["16x16x8", "8x8x8"], required=True)
    parser.add_argument("--bit-order", choices=["msb", "lsb"], default="msb")
    parser.add_argument("--reverse-channels", action="store_true")
    parser.add_argument("--codebook-path", default=None, help="Optional reportgen-ready codebook.npy.")
    parser.add_argument("--codebook-metadata", default=None, help="metadata.json for --codebook-path.")
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--output", required=True, help="Raw BTB3D-style JSONL output.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conv-mode", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--type_filter", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_reportgen_records(json_path: str, type_filter: str | None) -> list[dict]:
    with open(json_path) as f:
        records = json.load(f)
    if type_filter:
        records = [record for record in records if record["id"].startswith(type_filter)]
    if not records:
        raise ValueError(f"No samples found in {json_path} with type_filter={type_filter!r}")
    return records


def load_done_images(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open() as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["image"])
    return done


def infer_conv_mode(model_name: str, requested: str | None) -> str:
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llama3"
    if requested is not None and requested != conv_mode:
        print(f"[WARNING] Auto-inferred conversation mode is {conv_mode}, using requested {requested}")
        conv_mode = requested
    return conv_mode


def build_prompt(conv_templates, conv_mode: str, user_text: str) -> str:
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def stack_equal_length_input_ids(input_ids_list: list[torch.Tensor], device: torch.device):
    lengths = {ids.numel() for ids in input_ids_list}
    if len(lengths) != 1:
        raise ValueError(f"Batch contains mixed prompt lengths: {sorted(lengths)}")
    input_ids = torch.stack([ids.reshape(-1) for ids in input_ids_list]).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask, next(iter(lengths))


def clean_decoded_text(text: str, pad_token: str | None) -> str:
    text = text.replace("<|eot_id|>", "").strip()
    if pad_token:
        text = text.replace(pad_token, "").strip()
    return text


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if bool(args.codebook_path) != bool(args.codebook_metadata):
        raise ValueError("--codebook-path and --codebook-metadata must be provided together")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and output_path.exists():
        output_path.unlink()
        print(f"Removed stale raw output before rerun: {output_path}")

    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    from llava.utils import disable_torch_init

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    if "llava" not in model_name.lower():
        model_name = "llava-lora-" + model_name
    conv_mode = infer_conv_mode(model_name, args.conv_mode)

    print(f"Loading model: {args.model_path}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        False,
        False,
        device=device,
    )
    if not isinstance(model, LlavaLlamaForCausalLM):
        model.__class__ = LlavaLlamaForCausalLM
    model.to(device)
    model.eval()

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")

    records = load_reportgen_records(args.json_path, args.type_filter)
    done_images = load_done_images(output_path)
    pending_records = [record for record in records if image_output_name(record["image"]) not in done_images]
    print(f"Samples: total={len(records)} done={len(done_images)} pending={len(pending_records)} batch_size={args.batch_size}")

    pending_items = []
    for record in pending_records:
        human_turns = [conv for conv in record["conversations"] if conv["from"] == "human"]
        if len(human_turns) != 1:
            raise ValueError(f"Expected exactly one human turn for {record['id']}, got {len(human_turns)}")
        question = human_turns[0]["value"]
        prompt = build_prompt(conv_templates, conv_mode, question)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        pending_items.append({
            "record": record,
            "question": question,
            "prompt": prompt,
            "input_ids": input_ids,
            "token_len": int(input_ids.numel()),
        })
    # The BTB3D multimodal prepare path does not remove padded text tokens.
    # Bucket by exact prompt length so batch inference does not change context.
    pending_items.sort(key=lambda item: item["token_len"])

    id_to_idx, matrix = open_matrix(args.token_dir)
    convention = convention_from_flags(args.bit_order, args.reverse_channels)
    codebook = None
    if args.codebook_path:
        codebook, codebook_metadata = load_reportgen_codebook(args.codebook_path, args.codebook_metadata)
        print(
            f"Loaded reportgen codebook {args.codebook_path} "
            f"mode={codebook_metadata.get('codebook_mode')} dtype={codebook.dtype}"
        )
    first_debug_done = False

    with output_path.open("a") as out_f:
        start = 0
        progress = tqdm(total=len(pending_items), desc="batch_token_reportgen", unit="sample")
        while start < len(pending_items):
            token_len = pending_items[start]["token_len"]
            end = start
            while end < len(pending_items) and pending_items[end]["token_len"] == token_len and end - start < args.batch_size:
                end += 1
            batch_items = pending_items[start:end]
            image_arrays: list[np.ndarray] = []
            image_sizes: list[int] = []
            image_names: list[str] = []
            sample_ids: list[str] = []
            questions: list[str] = []
            input_ids_list: list[torch.Tensor] = []

            for item in batch_items:
                record = item["record"]
                image_name = image_output_name(record["image"])
                volume_id = volume_id_from_image(record["image"])
                image = materialize_reportgen_image(
                    args.token_dir,
                    volume_id,
                    args.compression,
                    convention,
                    id_to_idx,
                    matrix,
                    codebook,
                )
                image_arrays.append(image[0])
                image_sizes.append(image.size)
                image_names.append(image_name)

                question = item["question"]
                sample_ids.append(record["id"])
                questions.append(question)
                input_ids_list.append(item["input_ids"])

            image_tensor = torch.tensor(np.stack(image_arrays, axis=0)).to(device, dtype=torch.float16)
            input_ids, attention_mask, prompt_width = stack_equal_length_input_ids(input_ids_list, device)

            if args.debug and not first_debug_done:
                print(f"[DEBUG] image_tensor shape={tuple(image_tensor.shape)} dtype={image_tensor.dtype}")
                print(f"[DEBUG] input_ids shape={tuple(input_ids.shape)} prompt_width={prompt_width}")
                print(f"[DEBUG] contains IMAGE_TOKEN_INDEX={(input_ids == IMAGE_TOKEN_INDEX).any().item()}")

            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

            conversations_out = []
            for i, image_name in enumerate(image_names):
                answer = clean_decoded_text(tokenizer.decode(output_ids[i]), tokenizer.pad_token)
                conversations_out.append({
                    "id": sample_ids[i],
                    "question": questions[i],
                    "answer": answer,
                })
                out_f.write(json.dumps({
                    "image": image_name,
                    "conversations_out": [conversations_out[-1]],
                }) + "\n")

            out_f.flush()
            if args.debug and not first_debug_done:
                print(f"\n=== First Batch Sample: {image_names[0]} ===")
                print(f"[Question]\n{questions[0]}")
                print(f"[Generated]\n{conversations_out[0]['answer']}")
                print("=" * 60)
                first_debug_done = True
            progress.update(len(batch_items))
            start = end

    print(f"Inference completed. Results saved to {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

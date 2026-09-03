#!/usr/bin/env python3
"""Run frozen BTB3D/LLaVA inference from BTB3D token artifacts.

This mirrors the author's `multigpu_llava_eval.py` output format, but replaces
the `.nii_embedded.npz` read with runtime materialization from
`quantized_output.indices`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import torch
import tqdm

try:
    from dtbd3d.core.artifact import open_matrix
    from dtbd3d.core.token_codec import LFQConvention
    from dtbd3d.core.visual_embedding import load_reportgen_codebook, materialize_reportgen_image
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dtbd3d.core.artifact import open_matrix
    from dtbd3d.core.token_codec import LFQConvention
    from dtbd3d.core.visual_embedding import load_reportgen_codebook, materialize_reportgen_image


def volume_id_from_image(image: str) -> str:
    name = Path(image).name
    for suffix in [".nii_embedded.npz", ".nii.gz", ".nii", ".npz"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def image_output_name(image: str) -> str:
    return f"{volume_id_from_image(image)}.nii_embedded.npz"


def convention_from_flags(bit_order: str, reverse_channels: bool) -> LFQConvention:
    return f"{bit_order}_{'reverse_channels' if reverse_channels else 'identity'}"  # type: ignore[return-value]


def process_subset(args, data_subset, gpu_index, output_list, output_filename, done_ids):
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    from llava.utils import disable_torch_init

    device = torch.device(f"cuda:{gpu_index}")
    torch.cuda.set_device(device)
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    if "llava" not in model_name.lower():
        model_name = "llava-lora-" + model_name
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

    id_to_idx, matrix = open_matrix(args.token_dir)
    convention = convention_from_flags(args.bit_order, args.reverse_channels)
    codebook = None
    if args.codebook_path:
        codebook, codebook_metadata = load_reportgen_codebook(args.codebook_path, args.codebook_metadata)
        print(
            f"GPU {gpu_index}: loaded reportgen codebook {args.codebook_path} "
            f"mode={codebook_metadata.get('codebook_mode')} dtype={codebook.dtype}"
        )
    output_save = []
    first_sample_done = False

    for element in tqdm.tqdm(data_subset, desc=f"GPU {gpu_index} Processing"):
        image_file = image_output_name(element["image"])
        if image_file in done_ids:
            continue

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
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(f"[WARNING] Auto-inferred conversation mode is {conv_mode}, but --conv-mode is {args.conv_mode}, using {args.conv_mode}")
            conv_mode = args.conv_mode

        conv = conv_templates[conv_mode].copy()
        volume_id = volume_id_from_image(element["image"])
        image = materialize_reportgen_image(args.token_dir, volume_id, args.compression, convention, id_to_idx, matrix, codebook)
        image_size = image.size
        image_tensor = torch.tensor(image)

        if args.debug and not first_sample_done:
            print(f"\n[DEBUG] image shape: {image.shape}, dtype: {image.dtype}")
            print(f"[DEBUG] image min: {image.min():.4f}, max: {image.max():.4f}, mean: {image.mean():.4f}")
            zero_image_tensor = torch.zeros_like(image_tensor).to(device, dtype=torch.float16)

        image_tensor = image_tensor.to(device, dtype=torch.float16)
        conversations_save = []
        for conversation in element["conversations"]:
            if conversation["from"] != "human":
                continue
            sample_id = element["id"]
            inp = conversation["value"]
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

            if args.debug and not first_sample_done:
                has_img_token = (input_ids == IMAGE_TOKEN_INDEX).any().item()
                print(f"[DEBUG] IMAGE_TOKEN_INDEX={IMAGE_TOKEN_INDEX}, input_ids contains it: {has_img_token}")
                print(f"[DEBUG] input_ids shape: {input_ids.shape}")

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    use_cache=True,
                )
            outputs = tokenizer.decode(output_ids[0]).strip().replace("<|eot_id|>", "").strip()
            conv.messages[-1][-1] = outputs

            if args.debug and not first_sample_done:
                gt = next((c["value"] for c in element["conversations"] if c["from"] == "gpt"), "N/A")
                with torch.inference_mode():
                    zero_output_ids = model.generate(
                        input_ids,
                        images=zero_image_tensor,
                        image_sizes=[image_size],
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )
                zero_outputs = tokenizer.decode(zero_output_ids[0]).strip().replace("<|eot_id|>", "").strip()
                print(f"\n[DEBUG] prompt (last 300 chars): ...{prompt[-300:]}")
                print(f"\n=== First Sample: {image_file} ===")
                print(f"[Ground Truth]\n{gt}")
                print(f"[Generated (real image)]\n{outputs}")
                print(f"[Generated (zero image)]\n{zero_outputs}")
                print("=" * 60 + "\n")
                first_sample_done = True

            conversations_save.append({"id": sample_id, "question": inp, "answer": outputs})
            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        record = {"image": image_file, "conversations_out": conversations_save}
        output_save.append(record)
        with open(output_filename, "a") as f:
            f.write(json.dumps(record) + "\n")

    output_list += output_save


def main(args):
    save_it = args.model_path.split("/")[-2].split("-")[-1]
    print(save_it)
    with open(args.json_path) as f:
        data_val = json.load(f)
    if args.type_filter:
        data_val = [d for d in data_val if d["id"].startswith(args.type_filter)]
        print(f"Filtered to {len(data_val)} samples with type '{args.type_filter}'")
    if not data_val:
        raise ValueError(f"No samples found in {args.json_path} with type_filter={args.type_filter!r}")
    if args.num_gpus <= 0:
        raise ValueError(f"--num_gpus must be positive, got {args.num_gpus}")
    data_subset = data_val

    split_size = len(data_subset) // args.num_gpus
    data_splits = [data_subset[i * split_size : (i + 1) * split_size] for i in range(args.num_gpus)]
    for i in range(len(data_subset) % args.num_gpus):
        data_splits[i].append(data_subset[-(i + 1)])

    type_tag = f"_{args.type_filter}" if args.type_filter else ""
    output_filename = f"16_preprocessed_encoded_attnpool_1node_{save_it}it_node0{type_tag}_vqa.jsonl"
    done_ids = set()
    if Path(output_filename).exists():
        with open(output_filename) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add(rec["image"])
        print(f"Resuming: {len(done_ids)} already done")

    manager = multiprocessing.Manager()
    output_list = manager.list()
    processes = []
    for gpu_idx in range(args.num_gpus):
        p = multiprocessing.Process(
            target=process_subset,
            args=(args, data_splits[gpu_idx], gpu_idx, output_list, output_filename, done_ids),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    failed = [p.exitcode for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} worker process(es) failed with exit codes: {failed}")
    print(f"Inference completed. Results saved to {output_filename}.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Token-artifact LLaVA Processing Script")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--token_dir", type=str, default=None, help="BTB3D token artifact directory.")
    parser.add_argument("--canonical_dir", dest="token_dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--compression", choices=["16x16x8", "8x8x8"], required=True)
    parser.add_argument("--bit-order", choices=["msb", "lsb"], default="msb")
    parser.add_argument("--reverse-channels", action="store_true")
    parser.add_argument("--codebook-path", type=str, default=None, help="Optional reportgen-ready codebook.npy.")
    parser.add_argument("--codebook-metadata", type=str, default=None, help="metadata.json for --codebook-path.")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--type_filter", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    parsed = parser.parse_args()
    if not parsed.token_dir:
        raise SystemExit("Missing required argument: --token_dir")
    if bool(parsed.codebook_path) != bool(parsed.codebook_metadata):
        raise SystemExit("--codebook-path and --codebook-metadata must be provided together")
    main(parsed)

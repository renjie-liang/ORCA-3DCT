import argparse
import torch
import json
import os
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
import numpy as np
import tqdm
import nibabel as nib
import pandas as pd
import multiprocessing
from functools import partial

def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
        array (torch.Tensor): Input array to be resized.
        current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
        target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
        np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def process_subset(args, data_subset, gpu_index, output_list, output_filename, done_ids):
    """
    Process a subset of data on a specific GPU.

    Args:
        args: Parsed command-line arguments.
        data_subset (list): Subset of data to process.
        gpu_index (int): GPU index to use.
        output_list (multiprocessing.Manager().list): Shared list to store outputs.
        output_filename (str): Path to output file for incremental saving.
        done_ids (set): Already-processed element ids to skip.
    """
    # Assign the specific GPU
    device = torch.device(f"cuda:{gpu_index}")
    torch.cuda.set_device(device)
    disable_torch_init()

    # Load Model on the assigned GPU
    model_name = get_model_name_from_path(args.model_path)
    if 'llava' not in model_name.lower():
        model_name = 'llava-lora-' + model_name
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=device
    )

    # Ensure model is LlavaLlamaForCausalLM type for custom generate() support
    if not isinstance(model, LlavaLlamaForCausalLM):
        model.__class__ = LlavaLlamaForCausalLM

    model.to(device)

    # Prepare to collect outputs
    output_save = []
    first_sample_done = False

    for element in tqdm.tqdm(data_subset, desc=f"GPU {gpu_index} Processing"):
        image_file = element["image"].replace(".nii.gz",".npz").replace(".npz", ".nii_embedded.npz")
        if image_file in done_ids:
            continue
        # Determine conversation mode based on model name
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

        # Override with command-line argument if provided
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] Auto-inferred conversation mode is {}, but `--conv-mode` is {}, using {}'.format(
                conv_mode, args.conv_mode, args.conv_mode))
            conv_mode = args.conv_mode
        conv = conv_templates[conv_mode].copy()
        roles = ('user', 'assistant') if "mpt" in model_name.lower() else conv.roles

        image_path = args.embedding_path + image_file
        image = np.load(image_path)["arr"]  # (1, C, T, H, W)
        if args.merge_tokens:
            from einops import rearrange
            image = rearrange(torch.tensor(image), 'b c t (h p1) (w p2) -> b (c p1 p2) t h w', p1=2, p2=2).numpy()
        image = image.transpose(0,2,3,4,1)
        image_size = image.size
        image_tensor = torch.tensor(image)

        if not first_sample_done:
            print(f"\n[DEBUG] image shape: {image.shape}, dtype: {image.dtype}")
            print(f"[DEBUG] image min: {image.min():.4f}, max: {image.max():.4f}, mean: {image.mean():.4f}")
            # ablation: also run with zeros to check if model uses visual input
            zero_image_tensor = torch.zeros_like(image_tensor).to(device, dtype=torch.float16)

        if isinstance(image_tensor, list):
            image_tensor = [img.to(device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(device, dtype=torch.float16)  # Add batch dimension if needed
        conversations_save = []

        for conversation in element["conversations"]:
            i = 0
            id = element["id"]
            if conversation["from"] == "human":
                inp = conversation["value"]
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]

                if not first_sample_done:
                    img_tok_id = tokenizer.convert_tokens_to_ids("<image>")
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
                        use_cache=True)

                outputs = tokenizer.decode(output_ids[0]).strip().replace("<|eot_id|>", "").strip()
                conv.messages[-1][-1] = outputs

                if not first_sample_done:
                    gt = next((c["value"] for c in element["conversations"] if c["from"] == "gpt"), "N/A")
                    print(f"\n[DEBUG] prompt (last 300 chars): ...{prompt[-300:]}")
                    # ablation: run same prompt with zero image
                    with torch.inference_mode():
                        zero_output_ids = model.generate(
                            input_ids,
                            images=zero_image_tensor,
                            image_sizes=[image_size],
                            do_sample=False,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True)
                    zero_outputs = tokenizer.decode(zero_output_ids[0]).strip().replace("<|eot_id|>", "").strip()
                    print(f"\n=== First Sample: {image_file} ===")
                    print(f"[Ground Truth]\n{gt}")
                    print(f"[Generated (real image)]\n{outputs}")
                    print(f"[Generated (zero image)]\n{zero_outputs}")
                    print("=" * 60 + "\n")
                    first_sample_done = True

                conversations_save.append({"id":id, "question": inp, "answer":outputs})

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        record = {"image": image_file, "conversations_out": conversations_save}
        output_save.append(record)
        with open(output_filename, "a") as f:
            f.write(json.dumps(record) + "\n")

    # Append the results to the shared list
    output_list += output_save

    #except Exception as e:
    #    print(f"An error occurred on GPU {gpu_index}: {e}")

def main(args):
    # Load and Split Data
    save_it = args.model_path.split("/")[-2].split("-")[-1]
    print(save_it)

    with open(args.json_path, 'r') as file:
        data_val = json.load(file)

    if args.type_filter:
        data_val = [d for d in data_val if d["id"].startswith(args.type_filter)]
        print(f"Filtered to {len(data_val)} samples with type '{args.type_filter}'")

    total_nodes = args.total_nodes
    node_index = args.node_index

    # Calculate the subset of data for this node
    per_node = len(data_val) // total_nodes
    start_idx = node_index * per_node
    # Ensure the last node picks up any remaining data
    end_idx = start_idx + per_node if node_index != total_nodes - 1 else len(data_val)
    #start_idx = 0
    #end_idx = 12
    data_subset = data_val[start_idx:end_idx]

    # Number of GPUs per node
    num_gpus = args.num_gpus  # Adjust if needed

    # Split data_subset into num_gpus parts
    split_size = len(data_subset) // num_gpus
    data_splits = [data_subset[i*split_size : (i+1)*split_size] for i in range(num_gpus)]
    # Handle any remaining data
    for i in range(len(data_subset) % num_gpus):
        data_splits[i].append(data_subset[-(i+1)])

    # Determine output filename and load already-done ids for resume support
    type_tag = f"_{args.type_filter}" if args.type_filter else ""
    output_filename = f"16_preprocessed_encoded_attnpool_1node_{save_it}it_node{node_index}{type_tag}_vqa.jsonl"
    done_ids = set()
    if os.path.exists(output_filename):
        with open(output_filename) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add(rec["image"])
        print(f"Resuming: {len(done_ids)} already done")

    # Initialize multiprocessing Manager
    manager = multiprocessing.Manager()
    output_list = manager.list()

    # Create a list to hold processes
    processes = []

    for gpu_idx in range(num_gpus):
        p = multiprocessing.Process(target=process_subset, args=(args, data_splits[gpu_idx], gpu_idx, output_list, output_filename, done_ids))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print(f"Node {node_index} has completed processing. Results saved to {output_filename}.")

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn' to avoid CUDA re-initialization issues
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Distributed LLaVA Processing Script")

    # Model and Processing Arguments
    parser.add_argument("--model-path", type=str, default="path_to_checkpoint")
    parser.add_argument("--model-base", type=str, default="path_to_checkpoint")
    parser.add_argument("--embedding_path", type=str, default="path_to_encoded_volumes")
    parser.add_argument("--json_path", type=str, default="path_to_vqa_reports.json")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cuda').")
    parser.add_argument("--conv-mode", type=str, default=None, help="Conversation mode override.")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty (1.0=off, 1.3 recommended).")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit precision.")
    parser.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit precision.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--type_filter", type=str, default=None, help="Only run samples whose id starts with this prefix (e.g. 'report_generation').")
    parser.add_argument("--merge_tokens", action="store_true", help="Merge 2x2 spatial tokens (18-dim -> 72-dim) for 8x8x8 config.")

    # Distributed Arguments
    parser.add_argument("--node_index", type=int, default=0, help="Index of the current node (0).")
    parser.add_argument("--total_nodes", type=int, default=1, help="Total number of nodes (default: 1).")
    parser.add_argument("--num_gpus", type=int, default=1, help="Total number of gpus (default: 1).")

    args = parser.parse_args()
    main(args)

"""Shared constants for the BTB3D report-generation training pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(".")
CORE_CODE_ROOT = PROJECT_ROOT / "Experiment" / "core_code"
RUN_ROOT = PROJECT_ROOT / "Experiment" / "runs" / "btb3d_repro_generator"

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

PAD_TOKEN = "<pad>"
TOKEN_FOR_MULTIPLE_CHOICE = "<multiple_choice>"
TOKEN_FOR_LONG_ANSWER = "<long_answer>"
TOKEN_FOR_SHORT_ANSWER = "<short_answe>"
TOKEN_FOR_REPORT_GENERATION = "<report_generation>"
BTB3D_SPECIAL_TOKENS = [
    TOKEN_FOR_MULTIPLE_CHOICE,
    TOKEN_FOR_LONG_ANSWER,
    TOKEN_FOR_SHORT_ANSWER,
    TOKEN_FOR_REPORT_GENERATION,
]

LLAMA3_SYSTEM_PROMPT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    ". A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "<|eot_id|>"
)
LLAMA3_USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA3_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_EOT = "<|eot_id|>"

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


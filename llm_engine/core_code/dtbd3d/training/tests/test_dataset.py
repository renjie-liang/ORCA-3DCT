from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dtbd3d.core.token_codec import CODEBOOK_DIM
from dtbd3d.training.data import BTB3DDataCollator, BTB3DReportGenDataset
from dtbd3d.training.models import build_tiny_btb3d_model_for_tests


TOKEN_SHAPE = (31, 32, 32)
N_TOKENS = 31 * 32 * 32


def write_reportgen_codebook(tmp_path: Path) -> tuple[Path, Path]:
    codebook_dir = tmp_path / "codebook_artifact"
    codebook_dir.mkdir()
    codebook_path = codebook_dir / "codebook.npy"
    metadata_path = codebook_dir / "metadata.json"
    codebook = np.zeros((1 << CODEBOOK_DIM, CODEBOOK_DIM), dtype=np.float16)
    np.save(codebook_path, codebook)
    metadata_path.write_text(
        json.dumps(
            {
                "codebook_channel_order": "msb_reverse_channels",
                "reportgen_expected_channel_order": "msb_reverse_channels",
                "requires_channel_reverse_for_btb3d_pretrained_reportgen": False,
            }
        )
    )
    return codebook_path, metadata_path


def test_dataset_and_collator_shapes(tmp_path: Path):
    tokenizer, _ = build_tiny_btb3d_model_for_tests(vocab_size=128, hidden_size=32)
    tokens = np.zeros((4, N_TOKENS), dtype=np.uint32)
    tokens_path = tmp_path / "tokens_int.npy"
    np.save(tokens_path, tokens)
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("train_1_a_1\ntrain_1_a_2\ntrain_2_a_1\ntrain_2_a_2\n")
    codebook_path, metadata_path = write_reportgen_codebook(tmp_path)
    vqa_path = tmp_path / "train_vqa.json"
    records = [
        {
            "id": f"report_generation_{idx}",
            "image": f"train_{1 + idx // 2}_a_{1 + idx % 2}.nii.gz",
            "conversations": [
                {"from": "human", "value": "<image>\nGenerate report.<report_generation>", "type": "report_generation"},
                {"from": "gpt", "value": "Findings: clear. Impression: clear."},
            ],
        }
        for idx in range(4)
    ]
    vqa_path.write_text(json.dumps(records))
    dataset = BTB3DReportGenDataset(
        tokens_path=tokens_path,
        ids_path=ids_path,
        vqa_json_path=vqa_path,
        tokenizer=tokenizer,
        token_shape=TOKEN_SHAPE,
        token_dim=18,
        max_text_length=512,
        max_samples=4,
        compression="16x16x8",
        codebook_path=codebook_path,
        codebook_metadata_path=metadata_path,
    )
    collator = BTB3DDataCollator(pad_token_id=tokenizer.pad_token_id)
    batch = collator([dataset[0], dataset[1]])

    assert batch["image_features"].shape == (2, *TOKEN_SHAPE, 18)
    assert batch["input_ids"].ndim == 2
    assert batch["labels"].shape == batch["input_ids"].shape
    assert (batch["labels"] == -100).any()
    assert len(batch["metadata"]) == 2


def test_dataset_max_samples_after_id_filter(tmp_path: Path):
    tokenizer, _ = build_tiny_btb3d_model_for_tests(vocab_size=128, hidden_size=32)
    tokens_path = tmp_path / "tokens_int.npy"
    np.save(tokens_path, np.zeros((2, N_TOKENS), dtype=np.uint32))
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("train_10000_a_1\ntrain_10001_a_1\n")
    codebook_path, metadata_path = write_reportgen_codebook(tmp_path)

    vqa_path = tmp_path / "train_vqa.json"
    records = [
        {
            "id": "report_generation_unmatched",
            "image": "train_1_a_1.nii.gz",
            "conversations": [
                {"from": "human", "value": "<image>\nGenerate report.<report_generation>", "type": "report_generation"},
                {"from": "gpt", "value": "Findings: clear. Impression: clear."},
            ],
        },
        {
            "id": "report_generation_matched_0",
            "image": "train_10000_a_1.nii.gz",
            "conversations": [
                {"from": "human", "value": "<image>\nGenerate report.<report_generation>", "type": "report_generation"},
                {"from": "gpt", "value": "Findings: clear. Impression: clear."},
            ],
        },
        {
            "id": "report_generation_matched_1",
            "image": "train_10001_a_1.nii.gz",
            "conversations": [
                {"from": "human", "value": "<image>\nGenerate report.<report_generation>", "type": "report_generation"},
                {"from": "gpt", "value": "Findings: clear. Impression: clear."},
            ],
        },
    ]
    vqa_path.write_text(json.dumps(records))
    dataset = BTB3DReportGenDataset(
        tokens_path=tokens_path,
        ids_path=ids_path,
        vqa_json_path=vqa_path,
        tokenizer=tokenizer,
        token_shape=TOKEN_SHAPE,
        token_dim=18,
        max_text_length=512,
        max_samples=1,
        compression="16x16x8",
        codebook_path=codebook_path,
        codebook_metadata_path=metadata_path,
    )

    assert len(dataset) == 1
    assert dataset[0]["volume_id"] == "train_10000_a_1"

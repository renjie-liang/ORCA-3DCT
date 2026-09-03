"""Sampling helpers for region/text alignment pairs."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Sequence, TypeVar

T = TypeVar("T")

_NORMAL_FINDING_TEXTS = {
    "no finding",
    "no findings",
}


def normalize_finding_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .;:,")
    return text


def finding_text_from_pair(pair: Any) -> str:
    if isinstance(pair, dict):
        row = pair.get("index_row", pair.get("row", {}))
        if isinstance(row, dict):
            for key in ("raw_text", "finding", "label_name"):
                value = str(row.get(key, "")).strip()
                if value:
                    return value
        value = str(pair.get("text", pair.get("target_text", ""))).strip()
        marker = "Finding:"
        if marker in value:
            value = value.split(marker, 1)[1]
        return value

    row = getattr(pair, "row", None)
    if isinstance(row, dict):
        for key in ("raw_text", "finding", "label_name"):
            value = str(row.get(key, "")).strip()
            if value:
                return value
        value = str(row.get("text", row.get("target_text", ""))).strip()
    else:
        value = str(getattr(pair, "text", "")).strip()
    marker = "Finding:"
    if marker in value:
        value = value.split(marker, 1)[1]
    return value


def is_normal_no_finding_pair(pair: Any) -> bool:
    return normalize_finding_text(finding_text_from_pair(pair)) in _NORMAL_FINDING_TEXTS


def _stable_pair_key(pair: Any, *, volume_id: str, source_name: str, bucket: str) -> str:
    text_hash = str(getattr(pair, "text_hash", ""))
    embedding_row = str(getattr(pair, "embedding_row", ""))
    organ_group = str(getattr(pair, "organ_group", ""))
    mask_name = str(getattr(pair, "mask_name", ""))
    payload = f"{volume_id}:{source_name}:{bucket}:{embedding_row}:{organ_group}:{mask_name}:{text_hash}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def select_abnormal_first_pairs(pairs: Sequence[T], *, max_pairs: int, volume_id: str, source_name: str) -> list[T]:
    """Select text pairs with abnormal findings before capped no-finding pairs.

    If abnormal pairs exceed ``max_pairs``, choose a deterministic pseudo-random
    abnormal subset. If there is remaining capacity, fill it with deterministic
    pseudo-random no-finding pairs.
    """
    if max_pairs <= 0 or len(pairs) <= max_pairs:
        return list(pairs)

    abnormal = [pair for pair in pairs if not is_normal_no_finding_pair(pair)]
    normal = [pair for pair in pairs if is_normal_no_finding_pair(pair)]
    abnormal = sorted(abnormal, key=lambda pair: _stable_pair_key(pair, volume_id=volume_id, source_name=source_name, bucket="abnormal"))
    normal = sorted(normal, key=lambda pair: _stable_pair_key(pair, volume_id=volume_id, source_name=source_name, bucket="normal"))

    selected = abnormal[:max_pairs]
    if len(selected) < max_pairs:
        selected.extend(normal[: max_pairs - len(selected)])
    return selected

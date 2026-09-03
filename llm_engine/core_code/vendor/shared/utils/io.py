"""
IO utilities for JSON and JSONL files

Simple, fail-fast utilities for common file operations.
"""
import json
from pathlib import Path
from typing import Any, List, Dict


def load_jsonl(filepath: str) -> List[Dict]:
    """
    Load JSONL file (one JSON object per line)

    Args:
        filepath: Path to JSONL file

    Returns:
        List of dictionaries

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def save_jsonl(data: List[Dict], filepath: str) -> None:
    """
    Save list of dictionaries to JSONL file

    Args:
        data: List of dictionaries
        filepath: Path to output JSONL file

    Raises:
        TypeError: If data is not a list
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_json(filepath: str) -> Any:
    """
    Load JSON file

    Args:
        filepath: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to JSON file with pretty formatting

    Args:
        data: Data to save
        filepath: Path to output JSON file
        indent: JSON indentation (default: 2)
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

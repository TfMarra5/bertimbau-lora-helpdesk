
"""
Tests for the IT Helpdesk classification dataset.

Validates the original and expanded datasets without
loading the BERT model or running training.
"""

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_PATH = ROOT / "data" / "helpdesk_dataset.json"
ADDITIONS_PATH = ROOT / "data" / "helpdesk_additions_v0.2.json"

LABELS = {
    "PASSWORD_RESET",
    "ACCESS_ISSUE",
    "HARDWARE_FAILURE",
    "SOFTWARE_INSTALL",
    "NETWORK_LATENCY",
    "GENERAL_QUERY",
}


def load_data(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert isinstance(data, list), "Dataset must be a JSON array."

    return data


def test_dataset_sizes():
    original = load_data(ORIGINAL_PATH)
    additions = load_data(ADDITIONS_PATH)

    assert len(original) == 90
    assert len(additions) == 90
    assert len(original) + len(additions) == 180


def test_dataset_labels():
    original = load_data(ORIGINAL_PATH)
    additions = load_data(ADDITIONS_PATH)

    for item in original + additions:
        assert isinstance(item, dict)
        assert item.get("label") in LABELS


def test_dataset_texts():
    original = load_data(ORIGINAL_PATH)
    additions = load_data(ADDITIONS_PATH)

    for item in original + additions:
        assert isinstance(item.get("text"), str)
        assert item["text"].strip()


def test_dataset_balance():
    original = load_data(ORIGINAL_PATH)
    additions = load_data(ADDITIONS_PATH)

    original_counts = Counter(item["label"] for item in original)
    additional_counts = Counter(item["label"] for item in additions)

    for label in LABELS:
        assert original_counts[label] == 15
        assert additional_counts[label] == 15


def test_no_duplicate_tickets():
    original = load_data(ORIGINAL_PATH)
    additions = load_data(ADDITIONS_PATH)

    all_data = original + additions

    normalized_texts = [
        " ".join(item["text"].casefold().split())
        for item in all_data
    ]

    assert len(normalized_texts) == len(set(normalized_texts))

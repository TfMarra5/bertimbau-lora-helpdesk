
"""
Tests for reproducible helpdesk dataset splitting.

Ensures additional tickets enter only the training split
and that validation and test sets remain unchanged.
"""

import json
from pathlib import Path

from src.data_split import split_helpdesk_data


ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_PATH = ROOT / "data" / "helpdesk_dataset.json"
ADDITIONS_PATH = ROOT / "data" / "helpdesk_additions_v0.2.json"

LABELS = [
    "PASSWORD_RESET",
    "ACCESS_ISSUE",
    "HARDWARE_FAILURE",
    "SOFTWARE_INSTALL",
    "NETWORK_LATENCY",
    "GENERAL_QUERY",
]

LABEL2ID = {label: index for index, label in enumerate(LABELS)}


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_datasets():
    original = load_json(ORIGINAL_PATH)
    additional = load_json(ADDITIONS_PATH)
    return original, additional


def normalize_texts(records):
    return {
        " ".join(record["text"].casefold().split())
        for record in records
    }


def test_original_split_sizes():
    original, _ = get_datasets()

    train, val, test = split_helpdesk_data(
        original_data=original,
        additional_data=[],
        label_to_id=LABEL2ID,
        seed=42
    )

    assert len(train) == 63
    assert len(val) == 13
    assert len(test) == 14


def test_expanded_split_sizes():
    original, additional = get_datasets()

    train, val, test = split_helpdesk_data(
        original_data=original,
        additional_data=additional,
        label_to_id=LABEL2ID,
        seed=42
    )

    assert len(train) == 153
    assert len(val) == 13
    assert len(test) == 14


def test_evaluation_splits_remain_identical():
    original, additional = get_datasets()

    _, val_original, test_original = split_helpdesk_data(
        original, [], LABEL2ID, seed=42
    )

    _, val_expanded, test_expanded = split_helpdesk_data(
        original, additional, LABEL2ID, seed=42
    )

    assert val_original == val_expanded
    assert test_original == test_expanded


def test_no_data_leakage():
    original, additional = get_datasets()

    train, val, test = split_helpdesk_data(
        original, additional, LABEL2ID, seed=42
    )

    train_texts = normalize_texts(train)
    val_texts = normalize_texts(val)
    test_texts = normalize_texts(test)
    additional_texts = normalize_texts(additional)

    # All additional examples must be in training
    assert additional_texts <= train_texts

    # No example can appear in multiple splits
    assert train_texts.isdisjoint(val_texts)
    assert train_texts.isdisjoint(test_texts)
    assert val_texts.isdisjoint(test_texts)

    # Additional examples must never enter evaluation
    assert additional_texts.isdisjoint(val_texts)
    assert additional_texts.isdisjoint(test_texts)


def test_split_reproducibility():
    original, additional = get_datasets()

    first = split_helpdesk_data(
        original, additional, LABEL2ID, seed=42
    )

    second = split_helpdesk_data(
        original, additional, LABEL2ID, seed=42
    )

    assert first == second

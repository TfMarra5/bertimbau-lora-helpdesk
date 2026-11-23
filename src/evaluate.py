
"""
Detailed evaluation of the BERT + LoRA helpdesk classifier.

Loads a saved LoRA adapter and evaluates it using the
same fixed test split as the controlled experiments.
"""

import argparse
import json

import numpy as np
import torch

from pathlib import Path
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from data_split import split_helpdesk_data


LABELS = [
    "PASSWORD_RESET",
    "ACCESS_ISSUE",
    "HARDWARE_FAILURE",
    "SOFTWARE_INSTALL",
    "NETWORK_LATENCY",
    "GENERAL_QUERY",
]

LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LoRA helpdesk classifier."
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/controlled-expanded-180",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/helpdesk_dataset.json",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="bert-base-uncased",
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}"
        )

    # Reconstruct the original fixed test split
    with open(args.data_path, "r", encoding="utf-8") as file:
        original_data = json.load(file)

    _, _, test_data = split_helpdesk_data(
        original_data=original_data,
        additional_data=[],
        label_to_id=LABEL2ID,
        seed=42,
    )

    texts = [item["text"] for item in test_data]
    true_labels = [LABEL2ID[item["label"]] for item in test_data]

    print("=" * 60)
    print("DETAILED MODEL EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Test examples: {len(test_data)}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Load tokenizer and original BERT architecture
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir)
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Restore fine-tuned LoRA weights
    model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_dir),
    )

    model = model.to(device)
    model.eval()

    # Tokenize all test tickets
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Run inference
    with torch.inference_mode():
        outputs = model(**inputs)
        predictions = (
            torch.argmax(outputs.logits, dim=-1)
            .cpu()
            .numpy()
        )

    true_labels = np.array(true_labels)

    # Overall accuracy
    accuracy = accuracy_score(true_labels, predictions)

    print("\n" + "=" * 60)
    print("OVERALL ACCURACY")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(
        f"Correct predictions: "
        f"{int((true_labels == predictions).sum())}"
        f"/{len(test_data)}"
    )

    # Metrics for each class
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            true_labels,
            predictions,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    # Confusion matrix
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=list(range(len(LABELS))),
    )

    print("Rows = actual classes")
    print("Columns = predicted classes\n")

    for index, label in enumerate(LABELS):
        print(f"{index}: {label}")

    print("\n      " + " ".join(f"{i:>4}" for i in range(len(LABELS))))

    for index, row in enumerate(matrix):
        values = " ".join(f"{value:>4}" for value in row)
        print(f"{index:>4}  {values}")

    # Show incorrectly classified tickets
    print("\n" + "=" * 60)
    print("MISCLASSIFIED TICKETS")
    print("=" * 60)

    mistakes = 0

    for index, (true_id, predicted_id) in enumerate(
        zip(true_labels, predictions),
        start=1,
    ):
        if true_id != predicted_id:
            mistakes += 1

            print(f"\nTicket #{index}")
            print(f"Text: {texts[index - 1]}")
            print(f"Actual label: {ID2LABEL[int(true_id)]}")
            print(f"Predicted label: {ID2LABEL[int(predicted_id)]}")

    if mistakes == 0:
        print("No misclassified tickets!")

    print(f"\nTotal mistakes: {mistakes}")


if __name__ == "__main__":
    main()

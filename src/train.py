"""
train.py
========
Fine-Tuning pipeline with LoRA (PEFT) on BERT-Base (110M parameters)
for IT Helpdesk ticket intent classification in English.
"""

import os
import json
import argparse
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Official classification labels
LABELS = [
    "PASSWORD_RESET",
    "ACCESS_ISSUE",
    "HARDWARE_FAILURE",
    "SOFTWARE_INSTALL",
    "NETWORK_LATENCY",
    "GENERAL_QUERY"
]

LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABELS)}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(LABELS))),
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1)
    }


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning on BERT for IT Helpdesk")
    parser.add_argument("--base_model", type=str, default="bert-base-uncased")
    parser.add_argument("--data_path", type=str, default="data/helpdesk_dataset.json")
    parser.add_argument(
        "--additional_data_path",
        type=str,
        default=None,
        help="Optional JSON file containing additional labeled tickets"
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints/bert-lora-helpdesk")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    args = parser.parse_args()
    # Reproducibility
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Active device: {device}")


    # 1. Load original dataset
    print(f"\n📂 Loading original dataset from: {args.data_path}")

    with open(args.data_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    if not isinstance(original_data, list):
        raise ValueError("The original dataset must be a JSON array.")

    # Load optional additional training data
    additional_data = []

    if args.additional_data_path:
        print(f"📂 Loading additional training data from: {args.additional_data_path}")

        with open(args.additional_data_path, "r", encoding="utf-8") as f:
            additional_data = json.load(f)

        if not isinstance(additional_data, list):
            raise ValueError("The additional dataset must be a JSON array.")

    # Validate both datasets
    all_data = original_data + additional_data

    for index, item in enumerate(all_data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid record at position {index}.")

        if not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError(f"Missing or empty text at position {index}.")

        if item.get("label") not in LABEL2ID:
            raise ValueError(
                f"Invalid label at position {index}: {item.get('label')}"
            )

    normalized_texts = [
        " ".join(item["text"].casefold().split())
        for item in all_data
    ]

    if len(normalized_texts) != len(set(normalized_texts)):
        raise ValueError("Duplicate ticket texts detected.")

    # Create dataset from ORIGINAL examples only
    texts = [item["text"] for item in original_data]
    labels = [LABEL2ID[item["label"]] for item in original_data]

    dataset = Dataset.from_dict({
        "text": texts,
        "label": labels
    })

    # Split original dataset into training and holdout sets
    train_indices, holdout_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=0.30,
        random_state=42,
        stratify=labels
    )

    # Keep validation and test sets fixed across experiments
    val_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=0.50,
        random_state=42,
        stratify=[labels[i] for i in holdout_indices]
    )

    train_ds = dataset.select(train_indices.tolist())
    eval_ds = dataset.select(val_indices.tolist())
    test_ds = dataset.select(test_indices.tolist())

    # Add new examples exclusively to the training split
    if additional_data:
        additional_ds = Dataset.from_dict({
            "text": [item["text"] for item in additional_data],
            "label": [
                LABEL2ID[item["label"]]
                for item in additional_data
            ]
        })

        train_ds = concatenate_datasets([train_ds, additional_ds])

    print(f"\n✅ Original tickets: {len(original_data)}")
    print(f"✅ Additional training tickets: {len(additional_data)}")
    print(f"✅ Total available tickets: {len(all_data)}")

    print(
        f"   Train: {len(train_ds)} | "
        f"Validation: {len(eval_ds)} | "
        f"Test: {len(test_ds)}"
    )

    # 2. Load Tokenizer
    print(f"\n🔤 Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize_func(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_tokenized = train_ds.map(tokenize_func, batched=True)
    eval_tokenized = eval_ds.map(tokenize_func, batched=True)
    test_tokenized = test_ds.map(tokenize_func, batched=True)

    # 3. Load 110M Base Model
    print(f"\n🧠 Loading base model ({args.base_model} - ~110M parameters)...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # 4. Configure LoRA (PEFT)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save=["classifier"]
    )

    model = get_peft_model(base_model, lora_config)
    print("\n" + "=" * 60)
    print("LoRA PARAMETER BREAKDOWN:")
    model.print_trainable_parameters()
    print("=" * 60)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=1,
        report_to="none"
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_tokenized,
        "eval_dataset": eval_tokenized,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics
    }
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

    # 6. Train
    print("\n🔥 Starting LoRA fine-tuning...")
    trainer.train()

    # 7. Final evaluation on unseen test data
    print("\n🧪 Evaluating on held-out test data...")

    test_metrics = trainer.evaluate(
        eval_dataset=test_tokenized,
        metric_key_prefix="test"
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy:        {test_metrics['test_accuracy']:.4f}")
    print(f"Precision Macro: {test_metrics['test_precision_macro']:.4f}")
    print(f"Recall Macro:    {test_metrics['test_recall_macro']:.4f}")
    print(f"F1-Score Macro:  {test_metrics['test_f1_macro']:.4f}")

    print("=" * 60)

    # 8. Save Adapter & Tokenizer
    print(f"\n💾 Saving LoRA adapter checkpoint to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 9. Measure output adapter size on disk
    adapter_file = os.path.join(args.output_dir, "adapter_model.safetensors")
    if os.path.exists(adapter_file):
        file_size_bytes = os.path.getsize(adapter_file)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print("=" * 60)
        print("🎉 SUCCESS! LoRA ADAPTER GENERATED:")
        print(f"📦 File: {adapter_file}")
        print(f"⚖️ Size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
        print(f"🎯 Full model weighs ~440 MB, while LoRA adapter weighs ~{file_size_mb:.2f} MB!")
        print("=" * 60)


if __name__ == "__main__":
    main()

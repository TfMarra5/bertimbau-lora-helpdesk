"""
inference.py
============
Interactive CLI tool to test the LoRA-adapted BERT model
on enterprise IT Helpdesk ticket intent classification in English.
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

LABELS = [
    "PASSWORD_RESET",
    "ACCESS_ISSUE",
    "HARDWARE_FAILURE",
    "SOFTWARE_INSTALL",
    "NETWORK_LATENCY",
    "GENERAL_QUERY"
]

LABEL_DESCRIPTIONS = {
    "PASSWORD_RESET": "🔑 Password Reset / Credential Unlock",
    "ACCESS_ISSUE": "🔒 Permission / Shared Folder Access / VPN",
    "HARDWARE_FAILURE": "🖥️ Physical Hardware / Monitor / Keyboard / Cables",
    "SOFTWARE_INSTALL": "📦 Software Installation & License Provisioning",
    "NETWORK_LATENCY": "🌐 Network Connectivity / Wi-Fi / Packet Loss",
    "GENERAL_QUERY": "❓ General Inquiries & Corporate IT Policies"
}


def load_model(adapter_path: str, base_model_id: str = "bert-base-uncased"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Loading model and tokenizer on device: {device}...")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path if os.path.exists(adapter_path) else base_model_id)

    # Load frozen 110M base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=len(LABELS)
    ).to(device)

    # Attach ~1.2 MB LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    model.eval()

    return model, tokenizer, device


def predict_intent(text: str, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
        best_class = torch.argmax(probs).item()
        confidence = probs[best_class].item() * 100

    return LABELS[best_class], confidence, probs


def interactive_chat(model, tokenizer, device):
    print("\n" + "=" * 65)
    print("🤖 IT HELPDESK AI (BERT-Base + LoRA 1.2 MB)")
    print("Enter ticket issue description or type 'exit' to quit.")
    print("=" * 65)

    while True:
        try:
            user_input = input("\n👤 User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break

            intent, conf, all_probs = predict_intent(user_input, model, tokenizer, device)
            desc = LABEL_DESCRIPTIONS.get(intent, intent)

            print(f"🎯 Identified Intent: {intent} ({conf:.2f}%)")
            print(f"📌 Category: {desc}")

            # Top-2 alternatives
            top_probs, top_indices = torch.topk(all_probs, k=min(2, len(LABELS)))
            if top_probs[1].item() > 0.15:
                alt_intent = LABELS[top_indices[1].item()]
                print(f"   (Secondary probable category: {alt_intent} with {top_probs[1].item()*100:.1f}%)")

        except KeyboardInterrupt:
            print("\n👋 Session closed.")
            break


def main():
    parser = argparse.ArgumentParser(description="Inference with BERT LoRA Helpdesk")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="checkpoints/bert-lora-helpdesk",
        help="Local path to LoRA checkpoint or Hugging Face Hub repo ID"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="bert-base-uncased",
        help="Base 110M model identifier"
    )
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.adapter_path, args.base_model)
    interactive_chat(model, tokenizer, device)


if __name__ == "__main__":
    main()

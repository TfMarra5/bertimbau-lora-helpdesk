"""
rank_ablation.py
================
Studies parameter scaling and adapter footprint across different LoRA ranks
for BERT-Base (110M parameters).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def compute_lora_stats(rank: int, d_model: int = 768, num_layers: int = 12, num_classes: int = 6):
    # Applied to query and value projections: 2 projections per layer
    num_projections = num_layers * 2
    # Matrix A: rank * d_model, Matrix B: d_model * rank
    params_per_proj = 2 * (d_model * rank)
    attention_lora_params = num_projections * params_per_proj
    # Classification head: d_model * num_classes
    classifier_params = (d_model * num_classes) + num_classes

    total_trainable = attention_lora_params + classifier_params
    total_base = 110_000_000

    size_fp32_bytes = total_trainable * 4
    size_fp32_mb = size_fp32_bytes / (1024 * 1024)
    size_fp16_mb = size_fp32_mb / 2.0

    reduction = (1.0 - (total_trainable / total_base)) * 100.0
    return total_trainable, size_fp32_mb, size_fp16_mb, reduction


def main():
    ranks = [1, 2, 4, 8, 16, 32, 64]

    print("=" * 75)
    print("📈 LoRA RANK (r) ABLATION & PARAMETER FOOTPRINT STUDY")
    print("Base Model: BERT-Base (110M Parameters, 12 Layers, Hidden Dim = 768)")
    print("=" * 75)

    print(f"\n| Rank (r) | Trainable Params | FP32 Size (MB) | FP16 Size (MB) | Parameter Reduction |")
    print(f"|:--------:|:----------------:|:--------------:|:--------------:|:-------------------:|")

    for r in ranks:
        trainable, fp32_mb, fp16_mb, reduction = compute_lora_stats(r)
        print(f"| {r:>8} | {trainable:>16,} | {fp32_mb:>14.2f} | {fp16_mb:>14.2f} | {reduction:>18.2f}% |")

    print("\n" + "=" * 75)
    print("VISUAL FOOTPRINT SCALING (FP32 MB):")
    for r in ranks:
        _, fp32_mb, _, _ = compute_lora_stats(r)
        bar = "█" * int(fp32_mb * 5)
        print(f"r={r:<2} : {bar:<45} {fp32_mb:.2f} MB")
    print("=" * 75)
    print("Recommendation: For sequence classification, r=8 provides the optimal")
    print("balance between expressiveness and keeping checkpoint size under ~1.2 MB.")
    print("=" * 75)


if __name__ == "__main__":
    main()

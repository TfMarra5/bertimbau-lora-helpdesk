"""
generate_plots.py
=================
Generates visual benchmark graphics for the GitHub README:
1. Model Storage Footprint (Full vs LoRA)
2. Rank (r) Scaling Curve
3. Inference Latency (Base vs Active vs Merged)
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def generate_benchmark_figures(output_path: str = "assets/benchmark_summary.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Color palette
    c_red = "#EF4444"
    c_blue = "#3B82F6"
    c_green = "#10B981"
    c_purple = "#8B5CF6"

    # ----------------------------------------------------
    # Subplot 1: Storage Footprint
    # ----------------------------------------------------
    ax1 = axes[0]
    categories = ["Full Model\nFine-Tuning", "LoRA Adapter\n(r = 8)"]
    sizes_mb = [419.62, 1.14]
    colors = [c_red, c_green]

    bars1 = ax1.bar(categories, sizes_mb, color=colors, width=0.55, edgecolor="#1F2937", linewidth=1.2)
    ax1.set_title("1. Disk Storage Footprint (FP32)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Checkpoint Size (Megabytes)", fontsize=11)
    ax1.set_yscale("log")
    ax1.set_ylim(0.1, 1000)

    # Data labels
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval * 1.35,
            f"{yval:.2f} MB",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    ax1.text(0.5, 0.45, "99.73% Storage Reduction", transform=ax1.transAxes,
             ha="center", fontsize=11, color=c_green, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECFDF5", edgecolor=c_green))

    # ----------------------------------------------------
    # Subplot 2: Rank (r) Scaling Curve
    # ----------------------------------------------------
    ax2 = axes[1]
    ranks = [1, 2, 4, 8, 16, 32, 64]
    adapter_sizes = [0.16, 0.30, 0.58, 1.14, 2.27, 4.52, 9.02]

    ax2.plot(ranks, adapter_sizes, marker="o", color=c_blue, linewidth=2.5, markersize=7, label="Adapter Size (MB)")
    ax2.axvline(x=8, color=c_purple, linestyle="--", linewidth=1.5, alpha=0.8, label="Recommended (r=8)")
    ax2.scatter([8], [1.14], color=c_purple, s=120, zorder=5)

    ax2.set_title("2. Adapter Size vs Rank (r)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("LoRA Rank (r)", fontsize=11)
    ax2.set_ylabel("Adapter Size (MB)", fontsize=11)
    ax2.set_xticks(ranks)
    ax2.set_xticklabels([str(r) for r in ranks])
    ax2.legend(loc="upper left", frameon=True)

    ax2.annotate(
        "Sweet Spot (1.14 MB)\nFull expressiveness",
        xy=(8, 1.14),
        xytext=(16, 0.6),
        arrowprops=dict(facecolor=c_purple, shrink=0.08, width=1.5, headwidth=6),
        fontsize=9.5,
        fontweight="semibold"
    )

    # ----------------------------------------------------
    # Subplot 3: Layer Inference Latency
    # ----------------------------------------------------
    ax3 = axes[2]
    latency_stages = ["Base Model\n(Unmodified)", "Active LoRA\n(Separate A & B)", "Merged LoRA\n(W_0 + B@A)"]
    latencies_ms = [3.14, 4.59, 3.14]
    stage_colors = ["#6B7280", "#F59E0B", c_green]

    bars3 = ax3.bar(latency_stages, latencies_ms, color=stage_colors, width=0.55, edgecolor="#1F2937", linewidth=1.2)
    ax3.set_title("3. Inference Latency (2,000 Iterations)", fontsize=13, fontweight="bold", pad=12)
    ax3.set_ylabel("Latency per Layer (ms)", fontsize=11)
    ax3.set_ylim(0, 6.0)

    for bar in bars3:
        yval = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.15,
            f"{yval:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold"
        )

    ax3.text(0.5, 0.85, "0.0% Latency Overhead (Merged)", transform=ax3.transAxes,
             ha="center", fontsize=10.5, color=c_green, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECFDF5", edgecolor=c_green))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Benchmark summary chart saved to: {output_path}")

if __name__ == "__main__":
    generate_benchmark_figures()

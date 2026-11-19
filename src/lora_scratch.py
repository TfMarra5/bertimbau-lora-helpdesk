"""
lora_scratch.py
===============
Educational, pure PyTorch implementation of Low-Rank Adaptation (LoRA)
from scratch without relying on third-party libraries (such as peft).

Mathematical Foundation:
------------------------
Given a pre-trained base weight matrix W_0 in R^[d_out, d_in]:
Instead of directly modifying W_0 during fine-tuning, W_0 remains frozen:
    W = W_0 + (alpha / r) * (B @ A)

Where:
    A in R^[r, d_in]  (initialized with Gaussian distribution)
    B in R^[d_out, r] (initialized strictly to zeros)
    r << min(d_in, d_out)
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Standard Linear layer wrapped with Low-Rank Adaptation (LoRA).
    """
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 1. Freeze base model parameters
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # 2. Low-rank projection matrices (A and B)
        # Matrix A: projects from in_features -> rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        # Matrix B: projects from rank -> out_features
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        # Optional dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # 3. Initialization following Hu et al., 2021:
        # A initialized with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B initialized to strictly ZERO
        # This guarantees B @ A = 0 initially, preserving base model behavior exactly.
        nn.init.zeros_(self.lora_B)

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.original_linear(x)

        # Base forward pass with frozen weights: x @ W_0.T
        base_out = self.original_linear(x)

        # LoRA forward pass: x -> Dropout -> A.T -> B.T * scaling
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()

        return base_out + (lora_out * self.scaling)

    def merge_weights(self):
        """
        Merges Delta W = (alpha/r) * (B @ A) directly into original W_0.
        Eliminates any additional inference latency overhead!
        """
        if not self.merged:
            with torch.no_grad():
                delta_w = (self.lora_B @ self.lora_A) * self.scaling
                self.original_linear.weight.data += delta_w
            self.merged = True
            print("Successfully merged LoRA weights into original W_0!")

    def unmerge_weights(self):
        """Unmerges LoRA weights if further fine-tuning is required."""
        if self.merged:
            with torch.no_grad():
                delta_w = (self.lora_B @ self.lora_A) * self.scaling
                self.original_linear.weight.data -= delta_w
            self.merged = False
            print("LoRA weights unmerged from W_0.")


if __name__ == "__main__":
    print("=" * 65)
    print("MATHEMATICAL DEMO: LoRA FROM SCRATCH (Pure PyTorch)")
    print("=" * 65)

    # Simulate standard BERT-base attention projection layer (768 x 768)
    in_dim, out_dim = 768, 768
    standard_linear = nn.Linear(in_dim, out_dim)

    orig_params = sum(p.numel() for p in standard_linear.parameters())
    print(f"Original linear layer parameters ({in_dim}x{out_dim}): {orig_params:,}")

    # Wrap with LoRALinear (rank r=8)
    lora_linear = LoRALinear(standard_linear, rank=8, alpha=16.0)

    trainable_params = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    print(f"Trainable parameters (LoRA A + B): {trainable_params:,}")
    print(f"Trainable parameter reduction: {100 * (1 - trainable_params / orig_params):.2f}%")
    print(f"FP32 memory footprint:")
    print(f"  - Full layer: {orig_params * 4 / 1024:.2f} KB")
    print(f"  - LoRA only:   {trainable_params * 4 / 1024:.2f} KB (Only ~48 KB!)")

    # Verify forward pass
    x = torch.randn(2, 16, in_dim)  # Batch=2, Seq=16, Dim=768
    out = lora_linear(x)
    print(f"\nOutput tensor shape: {out.shape} (Matches standard linear!)")

    # Verify weight merging
    lora_linear.merge_weights()
    merged_out = lora_linear(x)
    diff = torch.norm(out - merged_out).item()
    print(f"Output divergence after merge: {diff:.6f} (Exact numerical match!)")
    print("=" * 65)

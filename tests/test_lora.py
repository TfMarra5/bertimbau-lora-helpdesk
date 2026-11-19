import torch
import torch.nn as nn
from src.lora_scratch import LoRALinear


def test_lora_parameter_reduction():
    dim = 768
    rank = 8
    linear = nn.Linear(dim, dim)
    lora = LoRALinear(linear, rank=rank, alpha=16.0)

    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    expected_lora_params = (dim * rank) + (rank * dim)
    assert trainable == expected_lora_params
    assert trainable < 0.05 * (dim * dim)


def test_lora_initialization_zero_delta():
    dim = 256
    linear = nn.Linear(dim, dim)
    lora = LoRALinear(linear, rank=4, alpha=8.0)

    x = torch.randn(2, 8, dim)
    with torch.no_grad():
        base_out = linear(x)
        lora_out = lora(x)

    assert torch.allclose(base_out, lora_out, atol=1e-6)


def test_lora_merge_weights_equivalence():
    dim = 128
    linear = nn.Linear(dim, dim)
    lora = LoRALinear(linear, rank=4, alpha=8.0)

    # Simulate non-zero B to have an active adaptation
    nn.init.normal_(lora.lora_B, std=0.02)

    x = torch.randn(4, 10, dim)
    with torch.no_grad():
        unmerged_out = lora(x)
        lora.merge_weights()
        merged_out = lora(x)

    assert torch.allclose(unmerged_out, merged_out, atol=1e-5)

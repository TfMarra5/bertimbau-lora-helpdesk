"""
lora_scratch.py
================
Implementação pura e didática do Low-Rank Adaptation (LoRA) em PyTorch
sem depender de bibliotecas externas (como peft).

Conceito Matemático:
--------------------
Dado um peso base W_0 de dimensao [d_out, d_in]:
Em vez de atualizar W_0 diretamente, mantemos W_0 congelado e adicionamos:
    W = W_0 + (alpha / r) * (B @ A)

Onde:
    A in R^[r, d_in]  (inicializado com Gaussiana)
    B in R^[d_out, r] (inicializado com zeros)
    r << min(d_in, d_out)
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Camada Linear encapsulada com adaptação LoRA de baixo rank.
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

        # 1. Congelar os parâmetros originais da rede
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # 2. Criar as matrizes de projeção de baixo rank (A e B)
        # Matriz A: projeta de in_features -> rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        # Matriz B: projeta de rank -> out_features
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        # Dropout opcional para regularização
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # 3. Inicialização idêntica ao paper original do LoRA (Hu et al., 2021):
        # A com inicialização Kaiming/Gaussiana
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B com ZEROS absolutos!
        # Isso garante que B @ A = 0 no início do treino:
        # o modelo começa exatamente com o mesmo comportamento do modelo pré-treinado!
        nn.init.zeros_(self.lora_B)

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # Se já fundido, comporta-se como uma camada linear convencional
            return self.original_linear(x)

        # Saída do modelo base congelado: W_0 @ x
        base_out = self.original_linear(x)

        # Caminho LoRA: x -> Dropout -> A^T -> B^T * scaling
        # x tem dimensão [batch, seq_len, in_features]
        # (x @ A.T) resulta em [batch, seq_len, rank]
        # (resultado @ B.T) resulta em [batch, seq_len, out_features]
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        
        return base_out + (lora_out * self.scaling)

    def merge_weights(self):
        """
        Funde Delta W = (alpha/r) * (B @ A) diretamente no peso W_0 original.
        Elimina completamente qualquer custo de latência adicional na inferência!
        """
        if not self.merged:
            with torch.no_grad():
                delta_w = (self.lora_B @ self.lora_A) * self.scaling
                self.original_linear.weight.data += delta_w
            self.merged = True
            print("Pesos LoRA fundidos com sucesso a W_0 original!")

    def unmerge_weights(self):
        """Desfaz a fusão para continuar treinando se necessário."""
        if self.merged:
            with torch.no_grad():
                delta_w = (self.lora_B @ self.lora_A) * self.scaling
                self.original_linear.weight.data -= delta_w
            self.merged = False
            print("Pesos LoRA separados de W_0.")


if __name__ == "__main__":
    print("=" * 65)
    print("DEMONSTRAÇÃO MATEMÁTICA: LoRA DO ZERO (PyTorch Puro)")
    print("=" * 65)

    # Simular uma camada típica de atenção do BERTimbau (768 x 768)
    in_dim, out_dim = 768, 768
    camada_padrao = nn.Linear(in_dim, out_dim)

    params_original = sum(p.numel() for p in camada_padrao.parameters())
    print(f"Parâmetros da camada original ({in_dim}x{out_dim}): {params_original:,}")

    # Aplicar nosso LoRALinear com rank r=8
    camada_lora = LoRALinear(camada_padrao, rank=8, alpha=16.0)

    params_treinaveis = sum(p.numel() for p in camada_lora.parameters() if p.requires_grad)
    params_totais = sum(p.numel() for p in camada_lora.parameters())

    print(f"Parâmetros treináveis (LoRA A + B): {params_treinaveis:,}")
    print(f"Redução de parâmetros treináveis: {100 * (1 - params_treinaveis / params_original):.2f}%")
    print(f"Tamanho em memória de float32:")
    print(f"  - Camada completa: {params_original * 4 / 1024:.2f} KB")
    print(f"  - Apenas LoRA:     {params_treinaveis * 4 / 1024:.2f} KB (Apenas ~48 KB!)")

    # Teste de inferência
    x = torch.randn(2, 16, in_dim) # Batch=2, Seq=16, Dim=768
    saida = camada_lora(x)
    print(f"\nFormato da saída: {saida.shape} (Idêntico ao original!)")

    # Demonstração de Fusão (Merge)
    camada_lora.merge_weights()
    saida_fundida = camada_lora(x)
    print(f"Diferença após fusão: {torch.norm(saida - saida_fundida).item():.6f} (Perfeita equivalência!)")
    print("=" * 65)

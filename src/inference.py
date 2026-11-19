"""
inference.py
============
Script interativo para testar o modelo BERTimbau adaptado com LoRA
em chamados de Helpdesk e Suporte de TI em Português.
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

LABELS = [
    "RESET_SENHA",
    "PROBLEMA_ACESSO",
    "FALHA_HARDWARE",
    "INSTALACAO_SOFTWARE",
    "LENTIDAO_REDE",
    "DUVIDA_GERAL"
]

LABEL_DESCRIPTIONS = {
    "RESET_SENHA": "🔑 Redefinição / Desbloqueio de Senha",
    "PROBLEMA_ACESSO": "🔒 Permissão / Acesso a Pastas / VPN",
    "FALHA_HARDWARE": "🖥️ Falha Física / Monitor / Teclado / Cabo",
    "INSTALACAO_SOFTWARE": "📦 Instalação ou Licença de Programas",
    "LENTIDAO_REDE": "🌐 Conexão / Wi-Fi / Queda ou Latência de Rede",
    "DUVIDA_GERAL": "❓ Dúvidas Gerais e Políticas da Empresa"
}


def load_model(adapter_path: str, base_model_id: str = "neuralmind/bert-base-portuguese-cased"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Carregando tokenizador e modelo no dispositivo: {device}...")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path if os.path.exists(adapter_path) else base_model_id)

    # Carrega a base congelada de 110M parâmetros
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=len(LABELS)
    ).to(device)

    # Acopla o adaptador LoRA de ~1.2 MB
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
    print("🤖 HELPDESK AI (BERTimbau + LoRA 1.2 MB)")
    print("Digite a descrição do seu problema ou 'sair' para encerrar.")
    print("=" * 65)

    while True:
        try:
            user_input = input("\n👤 Usuário: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("👋 Encerrando. Até logo!")
                break

            intent, conf, all_probs = predict_intent(user_input, model, tokenizer, device)
            desc = LABEL_DESCRIPTIONS.get(intent, intent)

            print(f"🎯 Intenção Identificada: {intent} ({conf:.2f}%)")
            print(f"📌 Categoria: {desc}")

            # Top-2 probabilidades
            top_probs, top_indices = torch.topk(all_probs, k=min(2, len(LABELS)))
            if top_probs[1].item() > 0.15: # se houver segunda classe relevante
                alt_intent = LABELS[top_indices[1].item()]
                print(f"   (Segunda opção mais provável: {alt_intent} com {top_probs[1].item()*100:.1f}%)")

        except KeyboardInterrupt:
            print("\n👋 Encerrando sessão.")
            break


def main():
    parser = argparse.ArgumentParser(description="Inferência com BERTimbau LoRA Helpdesk")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="checkpoints/bertimbau-lora-helpdesk",
        help="Caminho local da pasta do LoRA ou repo ID do Hugging Face"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="neuralmind/bert-base-portuguese-cased",
        help="Modelo base de 110M"
    )
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.adapter_path, args.base_model)
    interactive_chat(model, tokenizer, device)


if __name__ == "__main__":
    main()

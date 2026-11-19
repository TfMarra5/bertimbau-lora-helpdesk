"""
train.py
========
Pipeline de Treinamento com LoRA (PEFT) sobre o BERTimbau Base (110M parâmetros)
para Classificação de Chamados de Helpdesk de TI em Português.
"""

import os
import json
import argparse
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)
import evaluate

# Mapeamento oficial de labels do nosso projeto Helpdesk
LABELS = [
    "RESET_SENHA",
    "PROBLEMA_ACESSO",
    "FALHA_HARDWARE",
    "INSTALACAO_SOFTWARE",
    "LENTIDAO_REDE",
    "DUVIDA_GERAL"
]

LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABELS)}


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    parser = argparse.ArgumentParser(description="Treinamento LoRA do BERTimbau para Helpdesk")
    parser.add_argument("--base_model", type=str, default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--data_path", type=str, default="data/helpdesk_dataset.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/bertimbau-lora-helpdesk")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Dispositivo em uso: {device}")

    # 1. Carregar dataset
    print(f"\n📂 Carregando dados de: {args.data_path}")
    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Dividir em Treino (85%) e Teste (15%)
    texts = [item["text"] for item in raw_data]
    labels = [LABEL2ID[item["label"]] for item in raw_data]

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    split_dataset = dataset.train_test_split(test_size=0.15, seed=42)
    train_ds = split_dataset["train"]
    eval_ds = split_dataset["test"]
    print(f"   Exemplos de treino: {len(train_ds)} | Exemplos de validação: {len(eval_ds)}")

    # 2. Carregar Tokenizador
    print(f"\n🔤 Carregando tokenizador: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize_func(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_tokenized = train_ds.map(tokenize_func, batched=True)
    eval_tokenized = eval_ds.map(tokenize_func, batched=True)

    # 3. Carregar Modelo Base
    print(f"\n🧠 Carregando Modelo Base ({args.base_model} - ~110M parâmetros)...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # 4. Configurar LoRA (PEFT)
    # Alvos: Projeções de query e value da camada de atenção auto-supervisionada
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save=["classifier"] # Treina também a cabeça classificadora de 6 classes
    )

    model = get_peft_model(base_model, lora_config)
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS DE PARÂMETROS DO LoRA:")
    model.print_trainable_parameters()
    print("=" * 60)

    # 5. Argumentos de Treinamento
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # 6. Treinar
    print("\n🔥 Iniciando treinamento do LoRA...")
    trainer.train()

    # 7. Salvar apenas o Adaptador LoRA e o Tokenizador
    print(f"\n💾 Salvando adaptador LoRA em: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 8. Medir o tamanho do arquivo final no disco!
    adapter_file = os.path.join(args.output_dir, "adapter_model.safetensors")
    if os.path.exists(adapter_file):
        file_size_bytes = os.path.getsize(adapter_file)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print("=" * 60)
        print(f"🎉 SUCESSO! TAMANHO DO ADAPTADOR LoRA GERADO:")
        print(f"📦 Arquivo: {adapter_file}")
        print(f"⚖️ Tamanho: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
        print(f"🎯 Enquanto o modelo original pesaria ~440 MB, seu LoRA tem ~{file_size_mb:.2f} MB!")
        print("=" * 60)


if __name__ == "__main__":
    main()

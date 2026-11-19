# How 1.2 MB Fine-Tunes a 110M Parameter AI: LoRA from Scratch

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face PEFT](https://img.shields.io/badge/PEFT-0.9+-yellow.svg)](https://github.com/huggingface/peft)
[![Model](https://img.shields.io/badge/BERT--Base-110M_Params-green.svg)](https://huggingface.co/bert-base-uncased)
[![Adapter Size](https://img.shields.io/badge/Adapter_Size-1.16_MB-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An educational and production-ready repository demonstrating how **Low-Rank Adaptation (LoRA)** fine-tunes a **110 million parameter** Transformer (`bert-base-uncased`) using an adapter checkpoint of **only ~1.2 MB** while preserving near-perfect task accuracy.

---

## 📌 Project Overview & Use Case

We train a specialized intent classification engine for enterprise **IT Helpdesk & Support Tickets**:

1. `PASSWORD_RESET` — Expired passwords, account lockout, credential renewal.
2. `ACCESS_ISSUE` — VPN authentication, shared network drive permissions, 403 errors.
3. `HARDWARE_FAILURE` — Monitor blackouts, laptop boot failures, stuck peripherals.
4. `SOFTWARE_INSTALL` — Deployment requests for VS Code, Docker, Office suites.
5. `NETWORK_LATENCY` — Wi-Fi drops, high ping latency, video conference jitter.
6. `GENERAL_QUERY` — IT policies, equipment return procedures, support SLAs.

---

## 🧠 Mathematical Foundation

In **Standard Full Fine-Tuning**, every weight matrix $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ is updated:
$$W = W_0 + \Delta W$$
For a 110M model, storing $\Delta W$ requires saving the entire model again (~440 MB in FP32).

**LoRA** (*Hu et al., 2021*) decomposes the update $\Delta W$ into the product of two low-rank matrices:
$$\Delta W = \frac{\alpha}{r} (B \cdot A)$$

Where:
* $A \in \mathbb{R}^{r \times d_{in}}$: initialized via Gaussian distribution.
* $B \in \mathbb{R}^{d_{out} \times r}$: initialized strictly to **zero** (ensuring $\Delta W = 0$ at training onset).
* $r \ll \min(d_{in}, d_{out})$ (e.g., $r = 8$).

```
           [     W_0     ]  (768 x 768)  --> FROZEN (requires_grad = False)
                  +
     [  B  ] (768 x 8)  x  [  A  ] (8 x 768)  --> TRAINABLE (12,288 parameters)
```

### Efficiency Comparison

| Metric | Full Fine-Tuning | LoRA (Rank = 8) | Savings |
| :--- | :--- | :--- | :--- |
| **Trainable Parameters** | 110,000,000 | ~295,000 | **-99.73%** |
| **Checkpoint Storage** | ~440 MB | **~1.16 MB** | **-99.74%** |
| **Training Speed** | Hours | Minutes | **~10x faster** |
| **Inference Overhead** | Baseline | **0 ms** (via weight merge) | No penalty |

---

## 📂 Repository Structure

```text
bert-lora-helpdesk/
├── README.md                          # Project documentation
├── requirements.txt                   # Environment dependencies
├── .gitignore                         # Standard git ignore rules
├── data/
│   └── helpdesk_dataset.json          # IT support ticket dataset (English)
├── notebooks/
│   └── lora_training_colab.ipynb      # Google Colab training & test notebook
└── src/
    ├── __init__.py
    ├── lora_scratch.py                # Pure PyTorch implementation of LoRA layer & merge
    ├── train.py                       # Hugging Face Trainer + PEFT pipeline
    └── inference.py                   # Real-time CLI interactive chatbot
```

---

## 🚀 Quick Start

### 1. Run in Google Colab (One-Click)
Open `notebooks/lora_training_colab.ipynb` directly in [Google Colab](https://colab.research.google.com/) to train on free GPU/CPU in under two minutes.

### 2. Local Execution

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/YOUR-USERNAME/bert-lora-helpdesk.git
cd bert-lora-helpdesk

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

**Step A: Inspect LoRA from Scratch (Pure PyTorch)**
```bash
python src/lora_scratch.py
```
*Outputs matrix decomposition, parameter reductions, and demonstrates zero-latency parameter merging.*

**Step B: Fine-Tune the Adapter**
```bash
python src/train.py --base_model bert-base-uncased --epochs 5 --rank 8
```
*Saves the adapter checkpoint to `checkpoints/bert-lora-helpdesk/adapter_model.safetensors` (~1.16 MB).*

**Step C: Interactive CLI Chatbot**
```bash
python src/inference.py --adapter_path checkpoints/bert-lora-helpdesk
```

---

## 🌐 Publishing to Hugging Face Hub

To publish your trained adapter under your Hugging Face username:

```bash
pip install huggingface_hub
huggingface-cli login
```

In Python:
```python
model.push_to_hub("YOUR-USERNAME/bert-lora-helpdesk")
tokenizer.push_to_hub("YOUR-USERNAME/bert-lora-helpdesk")
```

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

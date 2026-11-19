# How 1.2 MB Fine-Tunes a 110M Parameter AI: LoRA from Scratch

<p align="center">
  <a href="https://colab.research.google.com/github/TfMarra5/bertimbau-lora-helpdesk/blob/main/notebooks/lora_training_colab.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
  </a>
  <a href="https://github.com/TfMarra5/bertimbau-lora-helpdesk/actions/workflows/ci.yml">
    <img src="https://github.com/TfMarra5/bertimbau-lora-helpdesk/actions/workflows/ci.yml/badge.svg" alt="CI Status" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch 2.0+" />
  <img src="https://img.shields.io/badge/BERT--Base-110M_Params-green.svg" alt="BERT Base" />
  <img src="https://img.shields.io/badge/Adapter_Size-1.16_MB-brightgreen.svg" alt="Adapter Size" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License" />
</p>

An educational and production-grade implementation demonstrating how **Low-Rank Adaptation (LoRA)** fine-tunes a **110 million parameter** Transformer (`bert-base-uncased`) using an adapter checkpoint of **only ~1.2 MB** with zero added inference latency after weight merging.

---

## 📌 Project Overview & Use Case

We train a specialized intent classification and automated routing engine for enterprise **IT Helpdesk Tickets**:

* 🔑 `PASSWORD_RESET` — Expired passwords, account lockout, credential renewal.
* 🔒 `ACCESS_ISSUE` — VPN authentication, shared network drive permissions, 403 errors.
* 🖥️ `HARDWARE_FAILURE` — Monitor blackouts, laptop boot failures, stuck peripherals.
* 📦 `SOFTWARE_INSTALL` — Deployment requests for VS Code, Docker, Office suites.
* 🌐 `NETWORK_LATENCY` — Wi-Fi drops, high ping latency, video conference jitter.
* ❓ `GENERAL_QUERY` — IT policies, equipment return procedures, support SLAs.

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

```text
           [     W_0     ]  (768 x 768)  --> FROZEN (requires_grad = False)
                  +
     [  B  ] (768 x 8)  x  [  A  ] (8 x 768)  --> TRAINABLE (12,288 parameters)
```

---

## 📊 Empirical Benchmarks

Measured over 2,000 iterations comparing Base Linear vs LoRA vs Merged LoRA:

| Metric | Full Model | LoRA (Rank = 8) | Delta (%) |
| :--- | :--- | :--- | :--- |
| **Trainable Parameters** | 110,000,000 | 299,520 | **-99.73%** |
| **Disk Footprint (FP32)** | 419.62 MB | **1.14 MB** | **-99.73%** |
| **Layer Latency (Base)** | 3.14 ms | — | Baseline |
| **Layer Latency (LoRA Active)** | — | 4.59 ms | +46.1% (during training) |
| **Layer Latency (LoRA Merged)** | — | **3.14 ms** | **0.0% (Zero latency overhead!)** |

### Rank ($r$) Scaling Study

How checkpoint storage scales linearly with the chosen rank $r$:

```text
r=1  : 0.16 MB
r=2  : 0.30 MB
r=4  : 0.58 MB
r=8  : 1.14 MB  <-- Optimal sweet spot for classification
r=16 : 2.27 MB
r=32 : 4.52 MB
r=64 : 9.02 MB
```

---

## 📂 Repository Structure

```text
bert-lora-helpdesk/
├── .github/workflows/ci.yml           # Automated unit test CI pipeline
├── data/
│   └── helpdesk_dataset.json          # IT support ticket dataset (English)
├── notebooks/
│   └── lora_training_colab.ipynb      # One-click Google Colab notebook
├── src/
│   ├── __init__.py
│   ├── app.py                         # Interactive Gradio web dashboard
│   ├── benchmark.py                   # Empirical latency and memory benchmarking suite
│   ├── inference.py                   # Real-time CLI interactive chatbot
│   ├── lora_scratch.py                # Pure PyTorch implementation of LoRA layer & merge
│   ├── rank_ablation.py               # Rank ablation and parameter study
│   └── train.py                       # Hugging Face Trainer + PEFT fine-tuning pipeline
├── tests/
│   └── test_lora.py                   # Unit tests validating tensor math & weight merge
├── pyproject.toml                     # Modern package configuration
└── requirements.txt                   # Environment dependencies
```

---

## 🚀 Quick Start

### 1. Run in Google Colab (One-Click)
Click the badge above or open `notebooks/lora_training_colab.ipynb` directly in [Google Colab](https://colab.research.google.com/github/TfMarra5/bertimbau-lora-helpdesk/blob/main/notebooks/lora_training_colab.ipynb).

### 2. Local Installation

```bash
git clone https://github.com/TfMarra5/bertimbau-lora-helpdesk.git
cd bertimbau-lora-helpdesk

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -e .
```

### 3. Key Commands

* **Run Unit Tests:**
  ```bash
  pytest tests/
  ```

* **Inspect LoRA Math from Scratch (Pure PyTorch):**
  ```bash
  python src/lora_scratch.py
  ```

* **Run Latency & Parameter Benchmark:**
  ```bash
  python src/benchmark.py
  ```

* **Run Rank ($r$) Ablation Study:**
  ```bash
  python src/rank_ablation.py
  ```

* **Launch Interactive Gradio Web Dashboard:**
  ```bash
  python src/app.py
  ```

* **Train the Model:**
  ```bash
  python src/train.py --base_model bert-base-uncased --epochs 5 --rank 8
  ```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

# 🚀 Como 1.2 MB Ajustam uma IA de 110 Milhões de Parâmetros? (LoRA na Prática)

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Hugging%20Face-PEFT-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/BERTimbau-110M%20Params-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Adapter%20Size-1.16%20MB-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

Este repositório demonstra, de forma prática e matemática, como a técnica de **LoRA (Low-Rank Adaptation)** permite especializar um modelo de linguagem com mais de **110 milhões de parâmetros** (o **BERTimbau**) salvando um arquivo final de **apenas ~1.2 MB**, com acurácia próxima de 100% no domínio escolhido.

---

## 📌 O Caso de Uso: Helpdesk & Suporte de TI em Português

Classificador de intenções para automatizar o encaminhamento e triagem de chamados corporativos de TI:
1. `RESET_SENHA` — Esquecimento de senha, expiração de credencial, conta bloqueada no AD.
2. `PROBLEMA_ACESSO` — Falha na VPN, permissão em pastas de rede, erro 403.
3. `FALHA_HARDWARE` — Monitores apagados, notebook não liga, problemas de periféricos.
4. `INSTALACAO_SOFTWARE` — Pedidos de VS Code, Docker, licenças de Office e plugins.
5. `LENTIDAO_REDE` — Wi-Fi instável, alta latência em chamadas, perda de pacotes.
6. `DUVIDA_GERAL` — Políticas de backup, horários do suporte, procedimentos internos.

---

## 🧠 A Matemática por trás do LoRA

No **Fine-Tuning Tradicional**, você atualiza toda a matriz de pesos $W_0 \in \mathbb{R}^{d \times k}$:
$$W_{novo} = W_0 + \Delta W$$
Isso exige salvar todos os 110 milhões de parâmetros novamente (~440 MB em FP32).

O **LoRA** (*Hu et al., 2021*) assume que a variação $\Delta W$ possui uma **dimensão intrínseca baixa** (*low intrinsic rank*). Ele decompõe a atualização na multiplicação de duas matrizes muito menores:

$$\Delta W = \frac{\alpha}{r} (B \cdot A)$$

Onde:
* $A \in \mathbb{R}^{r \times d_{in}}$ (inicializada com distribuição Gaussiana)
* $B \in \mathbb{R}^{d_{out} \times r}$ (inicializada com **zeros**, garantindo que $\Delta W = 0$ no início)
* $r \ll \min(d_{in}, d_{out})$ (ex: $r = 8$)

```
          [   W_0   ]  (768 x 768) -> CONGELADO (requires_grad = False)
               +
  [ B ] (768 x 8)  x  [ A ] (8 x 768) -> TREINÁVEIS (Apenas 12.288 parâmetros!)
```

### Comparativo de Peso & Parâmetros

| Métrica | Fine-Tuning Completo | Com LoRA (r=8) | Redução |
| :--- | :--- | :--- | :--- |
| **Parâmetros Treináveis** | 110.000.000 | ~295.000 | **-99.73%** |
| **Tamanho do Arquivo Salvo** | ~440 MB | **~1.16 MB** | **-99.74%** |
| **Tempo de Treinamento** | Horas / GPUs caras | Minutos / GPU grátis ou CPU | **~10x mais rápido** |
| **Latência de Inferência** | 0 ms extra | **0 ms extra** (após fusão/merge) | Sem perda |

---

## 📂 Estrutura do Projeto

```
bertimbau-lora-helpdesk/
├── README.md                          # Este documento
├── requirements.txt                   # Dependências do projeto
├── .gitignore                         # Arquivos ignorados pelo Git
├── data/
│   └── helpdesk_dataset.json          # Dataset com frases de TI em português
├── notebooks/
│   └── treinamento_lora_colab.ipynb   # Notebook pronto para executar no Google Colab
└── src/
    ├── __init__.py
    ├── lora_scratch.py                # Implementação pura da matemática do LoRA em PyTorch
    ├── train.py                       # Script completo de treino usando Hugging Face PEFT
    └── inference.py                   # Chatbot interativo no terminal
```

---

## 🛠️ Como Executar

### Opção 1: No Google Colab (1-Clique)
Abra a pasta `notebooks/` e envie o arquivo `treinamento_lora_colab.ipynb` para o [Google Colab](https://colab.research.google.com). O notebook já contém a instalação das dependências, preparação dos dados, treino e o teste de chat interativo.

### Opção 2: Localmente na sua Máquina

1. **Clone o repositório e crie o ambiente:**
```bash
git clone https://github.com/SEU-USUARIO/bertimbau-lora-helpdesk.git
cd bertimbau-lora-helpdesk

python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

2. **Entenda o LoRA do Zero (PyTorch puro):**
Execute o script didático que demonstra a decomposição tensorial e o merge de matrizes sem nenhuma dependência além do PyTorch:
```bash
python src/lora_scratch.py
```

3. **Treine o Adaptador LoRA:**
```bash
python src/train.py --epochs 5 --batch_size 8 --rank 8 --alpha 16
```
Ao final do treinamento, o terminal exibirá o tamanho exato do arquivo gerado em disco (`adapter_model.safetensors` com ~1.2 MB).

4. **Inicie o Chatbot Interativo de Testes:**
```bash
python src/inference.py --adapter_path checkpoints/bertimbau-lora-helpdesk
```

---

## 🌐 Como Publicar no Hugging Face Hub

Para disponibilizar seu adaptador LoRA publicamente (como no modelo original):

1. Faça login no Hugging Face CLI:
```bash
pip install huggingface_hub
huggingface-cli login
```

2. No Python:
```python
from peft import PeftModel
from transformers import AutoTokenizer

model.push_to_hub("seu-usuario/bertimbau-lora-helpdesk")
tokenizer.push_to_hub("seu-usuario/bertimbau-lora-helpdesk")
```

---

## 📜 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

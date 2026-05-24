# Classificação de Câncer de Mama — BreaKHis

Classificação binária de imagens histopatológicas (benigno vs. maligno) usando três redes neurais convolucionais com protocolo **LOGO (Leave-One-Patient-Out)**, garantindo que nenhum paciente apareça simultaneamente no treino e no teste.

---

## Dataset

**BreaKHis** — 7.909 imagens histopatológicas de 82 pacientes, disponíveis em 4 ampliações (40X, 100X, 200X, 400X).

- Este projeto utiliza apenas a ampliação **200X**
- Classes: **Benigno** (0) e **Maligno** (1)
- As imagens são desbalanceadas: há mais amostras malignas do que benignas

---

## Estrutura do Projeto

```
PJT-1/
├── config.py              # configurações centrais (caminhos, hiperparâmetros)
├── dataset.py             # dataset PyTorch + extração de ID de paciente
├── models.py              # ResNet-50, EfficientNet-B3 e VGG-16 com fine-tuning
├── train_kfold.py         # treinamento com protocolo LOGO k-fold
├── voting.py              # comparação de estratégias de ensemble (soma, produto, max)
├── ensemble.py            # fusão soft voting legado
├── extract_patches.py     # extração de patches com janela deslizante
├── plot_resultados.py     # geração de gráficos a partir dos resultados
├── docs/
│   └── mudancas.txt       # descrição das mudanças recentes
└── resultados/
    ├── resnet/
    │   └── results.json
    ├── efficientnet/
    │   └── results.json
    ├── vgg/
    │   └── results.json
    └── voting/
        └── results.json
```

---

## Pipeline

### 1. Extração de patches

Executar **uma única vez** antes de treinar. Percorre todas as imagens do dataset e extrai patches com janela deslizante.

```bash
python extract_patches.py
```

- Tamanho do patch: **96×96 px**
- Stride: **47 px** (~50% de sobreposição)
- Ampliação: **200X**
- Os patches são salvos em disco mantendo a estrutura de pastas do dataset original (benign/malignant)

---

### 2. Treinamento

```bash
python train_kfold.py --model resnet
python train_kfold.py --model efficientnet
python train_kfold.py --model vgg
```

**Protocolo LOGO:**
- Cada rodada usa uma seed diferente para garantir divisões distintas
- Split 70% treino | 15% validação | 15% teste, dividido por paciente
- Nenhum patch do paciente de teste aparece no treino

**Balanceamento de classes:**
- Resolvido via `CrossEntropyLoss` com pesos inversamente proporcionais ao número de amostras de cada classe
- `label_smoothing=0.1` para reduzir confiança excessiva no treino

**Fine-tuning:**

| Modelo | Camadas descongeladas | Cabeça classificadora |
|---|---|---|
| ResNet-50 | `layer4` | BatchNorm + Linear(2048→256) + Dropout(0.5) + Linear(256→2) |
| EfficientNet-B3 | `features[7]` e `features[8]` | Dropout(0.5) + Linear(1536→2) |
| VGG-16 | `features[24:]` | Linear(25088→512) + BatchNorm + Dropout(0.5) + Linear(512→256) + Dropout(0.5) + Linear(256→2) |

**Outros detalhes:**
- Otimizador: AdamW com learning rates diferenciadas (cabeça vs. backbone)
- Scheduler: ReduceLROnPlateau (fator 0.5, paciência 2)
- Early stopping com paciência 10 — restaura o melhor estado ao final de cada fold
- Memória liberada entre folds: `gc.collect()` + `cuda.empty_cache()`

---

### 3. Voting Ensemble

Executar após ter os JSONs dos 3 modelos.

```bash
python voting.py
```

Compara três estratégias de fusão usando `P(maligno)` de cada modelo:

| Estratégia | Fórmula | Comportamento |
|---|---|---|
| Soma | `(p1 + p2 + p3) / 3` | Média aritmética — baseline robusto |
| Produto | `p1·p2·p3 / (p1·p2·p3 + (1−p1)·(1−p2)·(1−p3))` | Exige consenso dos 3 modelos |
| Max | `max(p1, p2, p3)` | Qualquer modelo confiante domina |

Em todos os casos: `P(maligno) >= 0.5 → Maligno`, caso contrário `→ Benigno`.

Imprime tabela comparativa (AUC-ROC, F1-macro, acurácia) e declara o melhor método. Salva os resultados em `resultados/voting/results.json`.

---

### 4. Plotagem dos resultados

```bash
python plot_resultados.py --model resnet
python plot_resultados.py --model efficientnet
python plot_resultados.py --model vgg
python plot_resultados.py --model ensemble
```

Gera gráficos para cada modelo:

- **grafico_loss.png** — curva de loss treino vs. validação (mostra overfitting)
- **grafico_acc_curva.png** — curva de acurácia treino vs. validação
- **grafico_roc.png** — curva ROC agregada
- **grafico_confusao.png** — matriz de confusão agregada

---

## Configurações (config.py)

```python
dataset_dir  = "/home/larissaac/Personal/dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
patches_dir  = "/home/larissaac/Personal/patches"
output_dir   = "/home/larissaac/Personal/PJT_1/resultados"

patch_size   = 96      # tamanho do patch em pixels
stride       = 47      # stride da janela deslizante
magnification = "200X" # ampliação utilizada

num_epochs   = 50      # épocas por fold (early stopping pode parar antes)
batch_size   = 64
lr_head      = 1e-4    # learning rate da cabeça classificadora
lr_backbone  = 5e-5    # learning rate das camadas descongeladas
weight_decay = 1e-3
patience     = 10      # early stopping

k_folds      = 5
seed         = 42
```

---

## Ambiente

```bash
# Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install torch torchvision scikit-learn matplotlib seaborn tqdm Pillow opencv-python-headless
```

---

## Requisitos

```
torch
torchvision
numpy
scikit-learn
matplotlib
seaborn
Pillow
opencv-python-headless
tqdm
```

---

## Hardware recomendado

- GPU: NVIDIA RTX 3060 (12 GB VRAM) ou equivalente
- Tempo estimado por modelo com GPU (5 folds × até 50 épocas com early stopping): ~5–6 horas
- Sem GPU: inviável para treino completo

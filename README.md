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
├── ensemble.py            # fusão das CNN com Late Fusion / Soft Voting
├── extract_patches.py     # extração de patches com janela deslizante
├── model_resnet.py        # ResNet-50 com fine-tuning
├── model_efficientnet.py  # EfficientNet-B0 com fine-tuning
├── model_vgg.py           # VGG-16 com fine-tuning
├── train_logo.py          # treinamento com protocolo LOGO
├── plot_resultados.py     # geração de gráficos a partir dos resultados
└── resultados/
    ├── resnet/
    │   ├── results.json
    │   ├── grafico_acuracia.png
    │   ├── grafico_roc.png
    │   └── grafico_confusao.png
    ├── efficientnet/
    └── vgg/
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
python train_logo.py --model resnet
python train_logo.py --model efficientnet
python train_logo.py --model vgg
```

Argumentos opcionais (exceto 'model'):

| Argumento | Descrição | Padrão |
|---|---|---|
| `--model` | modelo a treinar (`resnet`, `efficientnet`, `vgg`) | obrigatório |
| `--max-folds` | limita o número de folds (0 = todos) | valor em config.py |
| `--epochs` | número de épocas por fold | valor em config.py |
| `--workers` | número de workers do DataLoader | valor em config.py |

**Protocolo LOGO:**
- Cada fold usa um paciente diferente como teste
- Os folds são sorteados com `seed=42` — os 3 modelos usam exatamente os mesmos folds (essencial para a fusão posterior)
- Nenhum patch do paciente de teste aparece no treino

**Balanceamento de classes:**
- Resolvido via `CrossEntropyLoss` com pesos inversamente proporcionais ao número de amostras de cada classe

**Fine-tuning:**

| Modelo | Camadas descongeladas | Cabeça classificadora |
|---|---|---|
| ResNet-50 | `layer4` | Linear(2048→512→256→2) |
| EfficientNet-B0 | `features[6]` e `features[7]` | Dropout + Linear(1280→2) |
| VGG-16 | `features[24:]` | Linear(25088→4096→512→2) |

**Outros detalhes:**
- Otimizador: AdamW com learning rates diferenciadas (cabeça vs. backbone)
- Scheduler: ReduceLROnPlateau (fator 0.5, paciência 3)
- Sem early stopping — épocas fixas
- Melhor estado do modelo (menor val loss) é restaurado ao final de cada fold
- Memória liberada entre folds: `gc.collect()` + `cuda.empty_cache()`

---

### 3. Fusão das redes

Executar após ter os JSONs da ResNet50, VGG16 e EfficientNetB3.

```bash
python ensemble.py

# Opcional para especificar pasta de output customizada
python ensemble.py --output-dir "caminho/para/outputs"
```

Faz a fusão a nível de decisão dos 3 modelos de redes neurais após o treinamento individual de cada uma (Late Fusion).
Utiliza a saída da camada Softmax de cada rede para fazer a média das probabilidades (Soft Voting) para cada patch analisado.



---

### 4. Plotagem dos resultados

```bash
python plot_resultados.py --model resnet
python plot_resultados.py --model efficientnet
python plot_resultados.py --model vgg
python plot_resultados.py --model vgg
python plot_resultados.py --model ensemble
```

Gera 3 gráficos separado
s para cada modelo:

- **grafico_acuracia.png** — acurácia por fold com linha da média
- **grafico_roc.png** — curva ROC agregada (todos os folds combinados)
- **grafico_confusao.png** — matriz de confusão agregada

---

## Configurações (config.py)

```python
patch_size  = 96       # tamanho do patch em pixels
stride      = 47       # stride da janela deslizante
magnification = "200X" # ampliação utilizada

num_epochs  = 10       # épocas por fold (fixo, sem early stopping)
batch_size  = 128
lr_head     = 5e-3     # learning rate da cabeça classificadora
lr_backbone = 1e-4     # learning rate das camadas descongeladas
weight_decay = 1e-4
num_workers = 2        # use 0 se travar no Windows

max_folds   = 0        # 0 = todos os pacientes | N = modo rápido
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
opencv-python
tqdm
```

---

## Hardware utilizado

- GPU: NVIDIA RTX 3060 (12 GB VRAM)
- Tempo estimado por modelo (20 folds × 10 épocas): ~5–6 horas

"""
ResNet-50 adaptada para classificação binária no BreaKHis.

"""

import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes: int = 2, dropout: float = 0.2) -> nn.Module:
    """
    ResNet-50 pré-treinada (ImageNet V2).
    Estratégia de fine-tuning:
      - backbone (layer1-3): congelado
      - layer4: descongelado (features de alto nível específicas do domínio)
      - fc: substituído por cabeça nova (512 → num_classes)
    """
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # congela backbone inteiro
    for p in resnet.parameters():
        p.requires_grad = False

    # descongela apenas layer4 para fine-tuning suave
    for p in resnet.layer4.parameters():
        p.requires_grad = True

    resnet.fc = nn.Sequential(
        nn.Linear(2048, 512),   # ResNet-50 sempre sai com 2048 features
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )

    treinaveis = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in resnet.parameters())
    print(f"Parâmetros treináveis: {treinaveis:,} / {total:,} ({100*treinaveis/total:.1f}%)")

    return resnet

"""
EfficientNet-B3 adaptada para classificação binária no BreaKHis.
"""

import torch.nn as nn
from torchvision import models


def build_efficientnet_b3(num_classes: int = 2, dropout: float = 0.3) -> nn.Module:
    """
    EfficientNet-B3 pré-treinada (ImageNet).
    Estratégia de fine-tuning:
      - blocos iniciais: congelados
      - blocos 6 e 7: descongelados para características abstratas
      - classifier: nova cabeça de classificação
    """
    weights = models.EfficientNet_B3_Weights.DEFAULT
    model = models.efficientnet_b3(weights=weights)

    # Congela backbone inteiro
    for p in model.parameters():
        p.requires_grad = False

    # Descongela blocos finais (6 e 7) para fine-tuning
    for p in model.features[6].parameters():
        p.requires_grad = True
    for p in model.features[7].parameters():
        p.requires_grad = True

    # Substitui a classifier head
    num_ftrs = model.classifier[1].in_features  # = 1536 no B3
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(num_ftrs, num_classes),
    )

    treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros treináveis: {treinaveis:,} / {total:,} ({100*treinaveis/total:.1f}%)")

    return model

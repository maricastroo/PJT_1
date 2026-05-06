"""
VGG-16 adaptada para classificação binária no BreaKHis.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights

def build_vgg16(num_classes: int = 2, dropout: float = 0.5) -> nn.Module:
    """
    VGG-16 pré-treinada (ImageNet).
    Estratégia de fine-tuning:
      - features (camadas iniciais e intermediárias): congeladas
      - bloco final de convolução (features[24] em diante): descongelado
      - classifier: nova cabeça de classificação
    """
    weights = VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights)

    # Congela backbone inteiro
    for p in model.parameters():
        p.requires_grad = False

    # Descongela o último bloco convolucional para fine tuning
    # Na VGG-16, o último bloco começa por volta do índice 24 das features
    for p in model.features[24:].parameters():
        p.requires_grad = True

    # Substitui a classifier head 
    # VGG tem 3 camadas lineares
    # A entrada original é 512 * 7 * 7 = 25088
    in_features = model.classifier[0].in_features
    
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, num_classes)
    )

    treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Parâmetros treináveis: {treinaveis:,} / {total:,} ({100*treinaveis/total:.1f}%)")

    return model
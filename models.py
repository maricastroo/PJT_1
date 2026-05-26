"""
Define as 3 CNNs usadas no projeto: ResNet-50, EfficientNet-B3 e VGG-16.
Todas são pré-treinadas no ImageNet com fine-tuning apenas nas camadas finais.

PADRÃO DO CLASSIFICADOR (todos os modelos):
- Flatten
- Dropout(0.2)
- Linear(in_features → 512) + ReLU + BatchNorm + Dropout(0.2)
- Linear(512 → 256) + ReLU + Dropout(0.2)
- Linear(256 → num_classes)

Todos os modelos usam pesos IMAGENET1K_V1 para consistência.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_B3_Weights, VGG16_Weights

def _build_classifier(in_features: int, num_classes: int, dropout: float = 0.2) -> nn.Sequential:
    """
    Constrói o classificador padronizado: in_features → 512 → 256 → num_classes
    Usado por todos os modelos para garantir consistência
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=dropout),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )

def build_resnet50(num_classes: int = 2, dropout: float = 0.2) -> nn.Module:
    """
    ResNet-50 com classificador padronizado: 2048 → 512 → 256 → num_classes
    Fine-tuning apenas no layer4 (última camada convolucional).
    """
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True

    model.fc = _build_classifier(model.fc.in_features, num_classes, dropout)
    return model

def build_efficientnet_b3(num_classes: int = 2, dropout: float = 0.2) -> nn.Module:
    """
    EfficientNet-B3 com classificador padronizado: 1536 → 512 → 256 → num_classes
    Fine-tuning nos blocos 7 e 8 (últimas camadas convolucionais).
    """
    model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    
    for p in model.parameters():
        p.requires_grad = False
    for p in model.features[7].parameters():
        p.requires_grad = True
    for p in model.features[8].parameters():
        p.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = _build_classifier(in_features, num_classes, dropout)
    return model

def build_vgg16(num_classes: int = 2, dropout: float = 0.2) -> nn.Module:
    """
    VGG-16 com classificador padronizado: 25088 → 512 → 256 → num_classes
    Fine-tuning a partir do bloco 24 (últimas camadas convolucionais).
    """
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.features[24:].parameters():
        p.requires_grad = True

    model.classifier = _build_classifier(
        model.classifier[0].in_features, num_classes, dropout
    )
    return model

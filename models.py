"""
Define as 3 CNNs usadas no projeto: ResNet-50, EfficientNet-B3 e VGG-16.
Todas são pré-treinadas no ImageNet com fine-tuning apenas nas camadas finais.

Não é executado diretamente, é importado pelo train_kfold.py.
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_B3_Weights, VGG16_Weights


class ResNetHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float):
        super().__init__()
        self.bn   = nn.BatchNorm1d(in_features)
        self.fc1  = nn.Linear(in_features, 256)
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def build_resnet50(num_classes: int = 2, dropout: float = 0.5) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters(): p.requires_grad = False
    for p in model.layer4.parameters(): p.requires_grad = True

    in_features = model.fc.in_features  # 2048
    model.fc = ResNetHead(in_features, num_classes, dropout)
    return model


def build_efficientnet_b3(num_classes: int = 2, dropout: float = 0.5) -> nn.Module:
    model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    for p in model.parameters(): p.requires_grad = False

    # Descongela blocos 7 e 8 para fine-tuning
    for p in model.features[7].parameters(): p.requires_grad = True
    for p in model.features[8].parameters(): p.requires_grad = True

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(num_ftrs, num_classes),
    )
    return model


def build_vgg16(num_classes: int = 2, dropout: float = 0.5, use_gap: bool = False) -> nn.Module:
    """
    use_gap=False (padrão): classificador original com Linear(25088→512), ~13M parâmetros
    use_gap=True:           GAP(1×1) antes do classificador, Linear(512→256), ~132K parâmetros
    """
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    for p in model.parameters(): p.requires_grad = False
    # Descongela a partir do bloco 24
    for p in model.features[24:].parameters(): p.requires_grad = True

    if use_gap:
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
    else:
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
    return model
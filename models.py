"""
Define as 3 CNNs usadas no projeto: ResNet-50, EfficientNet-B3 e VGG-16.
Todas são pré-treinadas no ImageNet com fine-tuning apenas nas camadas finais.

Não é executado diretamente, é importado pelo train_kfold.py.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_B3_Weights, VGG16_Weights

def build_resnet50(num_classes: int = 2, dropout: float = 0.2) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters(): p.requires_grad = False
    for p in model.layer4.parameters(): p.requires_grad = True
    
    model.fc = nn.Sequential(
        nn.Linear(2048, 512), 
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )
    return model

def build_efficientnet_b3(num_classes: int = 2, dropout: float = 0.3) -> nn.Module:
    model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    for p in model.parameters(): p.requires_grad = False
    
    # Descongela blocos 7 e 8 para fine-tuning
    for p in model.features[7].parameters(): p.requires_grad = True
    for p in model.features[8].parameters(): p.requires_grad = True
    
    # Cabeça recebe 1536 features
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(num_ftrs, num_classes),
    )
    return model

def build_vgg16(num_classes: int = 2, dropout: float = 0.5) -> nn.Module:
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    for p in model.parameters(): p.requires_grad = False
    # Descongela a partir do bloco 24
    for p in model.features[24:].parameters(): p.requires_grad = True
    
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
    return model
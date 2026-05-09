"""
Modelos pré-treinados adaptados para classificação binária no BreaKHis.
"""

from typing import Literal
import torch
import torch.nn as nn
from torchvision import models


ModelType = Literal["resnet50", "efficientnet_b3", "vgg16"]


def build_model(
    model_type: ModelType = "resnet50",
    num_classes: int = 2,
    dropout: float = 0.2,
    pretrained: bool = True
) -> nn.Module:
    """
    Factory function para criar modelos.
    
    Args:
        model_type: Tipo de modelo ('resnet50', 'efficientnet_b3', 'vgg16')
        num_classes: Número de classes (padrão: 2 para binário)
        dropout: Taxa de dropout
        pretrained: Se deve carregar pesos do ImageNet
    
    Returns:
        Modelo PyTorch pronto para treinar
    
    Raises:
        ValueError: Se model_type não for suportado
    """
    builders = {
        "resnet50": _build_resnet50,
        "efficientnet_b3": _build_efficientnet,
        "vgg16": _build_vgg,
    }
    
    if model_type not in builders:
        raise ValueError(
            f"Modelo '{model_type}' não suportado. "
            f"Opções disponíveis: {list(builders.keys())}"
        )
    
    return builders[model_type](num_classes, dropout, pretrained)


def _build_resnet50(num_classes: int, dropout: float, pretrained: bool) -> nn.Module:
    """ResNet-50 com fine-tuning apenas no layer4."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    # Congela backbone
    for param in model.parameters():
        param.requires_grad = False

    # Descongela layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Nova cabeça classificadora
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )

    _print_trainable_params(model, "ResNet-50")
    return model


def _build_efficientnet(num_classes: int, dropout: float, pretrained: bool) -> nn.Module:
    """EfficientNet-B3 com fine-tuning nos blocos finais."""
    weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b3(weights=weights)

    # Congela tudo
    for param in model.parameters():
        param.requires_grad = False

    # Descongela blocos 6 e 7 (features finais)
    for param in model.features[6].parameters():
        param.requires_grad = True
    for param in model.features[7].parameters():
        param.requires_grad = True

    # Nova cabeça
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(num_ftrs, num_classes),
    )

    _print_trainable_params(model, "EfficientNet-B3")
    return model


def _build_vgg(num_classes: int, dropout: float, pretrained: bool) -> nn.Module:
    """VGG-16 com fine-tuning no bloco convolucional final."""
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16(weights=weights)

    # Congela tudo
    for param in model.parameters():
        param.requires_grad = False

    # Descongela último bloco convolucional (camadas 24+)
    for param in model.features[24:].parameters():
        param.requires_grad = True

    # Nova cabeça
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

    _print_trainable_params(model, "VGG-16")
    return model


def _print_trainable_params(model: nn.Module, name: str) -> None:
    """Imprime estatísticas de parâmetros treináveis."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total > 0 else 0
    
    print(f"\n{name}:")
    print(f"  Parâmetros treináveis: {trainable:,} / {total:,} ({pct:.2f}%)")

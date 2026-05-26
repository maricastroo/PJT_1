from pathlib import Path
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

# Configurações específicas por modelo (de acordo com os pesos oficiais do torchvision)
_MODEL_CONFIG = {
    "resnet": {
        "resize": 232,
        "crop": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": transforms.InterpolationMode.BILINEAR
    },
    "efficientnet": {
        "resize": 320,
        "crop": 300,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": transforms.InterpolationMode.BICUBIC
    },
    "vgg": {
        "resize": 256,
        "crop": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": transforms.InterpolationMode.BILINEAR
    },
}

def get_transforms(train: bool, model_name: str) -> transforms.Compose:
    cfg = _MODEL_CONFIG[model_name]

    if train:
        return transforms.Compose([
            transforms.Resize(cfg["resize"], interpolation=cfg["interpolation"]),
            transforms.RandomCrop(cfg["crop"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
            transforms.RandomErasing(p=0.3),
        ])
    
    return transforms.Compose([
        transforms.Resize(cfg["resize"], interpolation=cfg["interpolation"]),
        transforms.CenterCrop(cfg["crop"]),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])

def label_from_path(path: Path) -> int:
    parts_lower = {p.lower() for p in path.parts}
    return 0 if "benign" in parts_lower or "benigno" in parts_lower else 1

def patient_id_from_path(path: Path) -> str:
    """
    Extrai o ID do paciente a partir do nome do arquivo BreaKHis.
    Formato do arquivo: SOB_{B|M}_{subtipo}-{ano}-{id}-{ampliação}-{slide}_p{n}.png
    Exemplo: SOB_B_A-14-22549AB-200X-001_p0  →  paciente "14-22549AB"

    Fallback: usa o diretório pai como identificador.
    """
    stem = re.sub(r"_p\d+$", "", path.stem)
    parts = stem.split("-")
    if len(parts) >= 3 and parts[0].upper().startswith("SOB"):
        return f"{parts[1]}-{parts[2]}"
    return path.parent.name


class BreaKHisDataset(Dataset):
    def __init__(self, paths: list, transform=None):
        self.samples = [(p, label_from_path(p)) for p in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (96, 96), 0)
        if self.transform:
            img = self.transform(img)
        return img, label

def tem_tecido(path: Path, limiar: float = 220.0) -> bool:
    """Retorna True se o patch tiver tecido, descarta fundo branco"""
    try:
        img = Image.open(path).convert("RGB")
        img_array = np.asarray(img, dtype=np.uint8)
        return float(np.mean(img_array)) < limiar
    except Exception:
        return False


def scan_patches(patches_dir: str) -> tuple:
    """
    Retorna (paths, labels, patient_ids), o array de IDs de paciente
    é necessário para o agrupamento LOGO.
    Patches de fundo branco (sem tecido) são descartados automaticamente.
    """
    patches_path = Path(patches_dir)
    paths, labels, patient_ids = [], [], []
    descartados = 0
    
    all_patches = sorted(patches_path.rglob("*.png"))
    print(f"Encontrados {len(all_patches)} patches PNG. Filtrando por tecido...")
    
    for p in tqdm(all_patches, desc="Filtrando patches"):
        if not tem_tecido(p):
            descartados += 1
            continue
        paths.append(p)
        labels.append(label_from_path(p))
        patient_ids.append(patient_id_from_path(p))

    print(f"Patches com tecido: {len(paths):,} | Descartados (fundo): {descartados:,}")
    return np.array(paths), np.array(labels), np.array(patient_ids)

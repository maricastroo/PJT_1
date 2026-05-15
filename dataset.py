from pathlib import Path
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# Tamanhos oficiais de cada modelo 
_MODEL_SIZES = {
    "resnet": (232, 224),
    "vgg": (256, 224),
    "efficientnet": (320, 300),
}

def get_transforms(train: bool, model_name: str) -> transforms.Compose:
    resize_size, crop_size = _MODEL_SIZES[model_name]

    if train:
        return transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90), #antes era 15 graus, mas aumentei para 90 para mais diversidade e para o treino nao acostumar rapido
            transforms.RandomGrayscale(p=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
            transforms.RandomErasing(p=0.2),  #cobre regiões aleatórias
        ])
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

def label_from_path(path: Path) -> int:
    parts_lower = {p.lower() for p in path.parts}
    return 0 if "benign" in parts_lower or "benigno" in parts_lower else 1

# LOGO: extração do ID do paciente
def patient_id_from_path(path: Path) -> str:
    """
    Extrai o ID do paciente a partir do nome do arquivo BreaKHis.
    Formato do arquivo: SOB_{B|M}_{subtipo}-{ano}-{id}-{ampliação}-{slide}_p{n}.png
    Exemplo: SOB_B_A-14-22549AB-200X-001_p0  →  paciente "14-22549AB"

    Fallback: usa o diretório pai como identificador.
    """
    # remove sufixo de patch (_p0, _p1, etc.)
    stem = re.sub(r"_p\d+$", "", path.stem)
    parts = stem.split("-")
    # formato: ['SOB_B_A', '14', '22549AB', '200X', '001']
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
    """Retorna True se o patch tiver tecido — descarta fundo branco do H&E."""
    try:
        img = np.array(Image.open(path).convert("RGB"))
        return img.mean() < limiar
    except Exception:
        return False


def scan_patches(patches_dir: str) -> tuple:
    """
    Retorna (paths, labels, patient_ids) — o array de IDs de paciente
    é necessário para o agrupamento LOGO.
    Patches de fundo branco (sem tecido) são descartados automaticamente.
    """
    patches_path = Path(patches_dir)
    paths, labels, patient_ids = [], [], []
    descartados = 0
    for p in sorted(patches_path.rglob("*.png")):
        if not tem_tecido(p):
            descartados += 1
            continue
        paths.append(p)
        labels.append(label_from_path(p))
        patient_ids.append(patient_id_from_path(p))

    print(f"Patches com tecido: {len(paths):,} | Descartados (fundo): {descartados:,}")
    return np.array(paths), np.array(labels), np.array(patient_ids)
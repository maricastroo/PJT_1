"""
Dataset do BreaKHis: carrega patches salvos em disco + extrai ID de paciente.
"""

import re
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ── ImageNet stats (patches histológicos herdam esses valores) ─────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


# ── Helpers ────────────────────────────────────────────────────────────────────

def patient_id_from_stem(stem: str) -> str:
    """
    Extrai o ID de paciente do nome do arquivo.
    Exemplos BreaKHis:
      'SOB_B_A-14-22549AB-40X-001_p0'  →  'SOB_B_A-14-22549AB'
      'SOB_M_DC-14-2523-40X-001_p3'   →  'SOB_M_DC-14-2523'
    """
    # remove sufixo de patch (_p0, _patch_0, etc.)
    stem = re.sub(r"(_p\d+|_patch_\d+)$", "", stem)
    partes = stem.split("-")
    if len(partes) >= 3:
        return f"{partes[0]}-{partes[1]}-{partes[2]}"
    return stem


def label_from_path(path: Path) -> int:
    """0 = benigno, 1 = maligno — detecta pela pasta pai."""
    parts_lower = {p.lower() for p in path.parts}
    if "benign" in parts_lower or "benigno" in parts_lower:
        return 0
    return 1


# ── Dataset ────────────────────────────────────────────────────────────────────

class BreaKHisDataset(Dataset):
    """Carrega patches salvos em disco.

    Params
    ------
    paths     : lista de Path com os arquivos de patch
    transform : torchvision transform (use get_transforms)
    """

    def __init__(self, paths: list, transform=None):
        self.samples   = [(p, label_from_path(p)) for p in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Varredura de patches com IDs de paciente ───────────────────────────────────

def scan_patches(patches_dir: str) -> tuple:
    """
    Lê todos os patches salvos em disco e retorna:
        paths      : np.ndarray de Path
        labels     : np.ndarray de int (0=benigno, 1=maligno)
        patient_ids: np.ndarray de str (ID único por paciente)
    """
    patches_path = Path(patches_dir)
    if not patches_path.exists():
        raise FileNotFoundError(
            f"Diretório de patches não encontrado: {patches_path}\n"
            "Execute extract_patches.py primeiro."
        )

    paths, labels, pids = [], [], []
    for p in sorted(patches_path.rglob("*.png")):
        paths.append(p)
        labels.append(label_from_path(p))
        pids.append(patient_id_from_stem(p.stem))

    if not paths:
        raise RuntimeError(f"Nenhum patch encontrado em {patches_path}")

    return np.array(paths), np.array(labels), np.array(pids)

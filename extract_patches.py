"""
Passo 1 – Extração de patches com janela deslizante.
Executar UMA VEZ antes de treinar.

Uso:
    python extract_patches.py
    python extract_patches.py --patch-size 94 --stride 47 (excemplo)
"""

import argparse
import cv2
from pathlib import Path
from tqdm import tqdm

from config import Config


def extrair_patches(cfg: Config) -> None:
    dataset_path = Path(cfg.dataset_dir)
    patches_path = Path(cfg.patches_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

    imagens = (
        list(dataset_path.rglob("*.png"))
        + list(dataset_path.rglob("*.jpg"))
        + list(dataset_path.rglob("*.tif"))
    )

    if not imagens:
        raise RuntimeError(f"Nenhuma imagem encontrada em {dataset_path}")

    print(f"Imagens encontradas: {len(imagens)}")
    print(f"Tamanho do patch: {cfg.patch_size}×{cfg.patch_size}")
    print(f"Stride: {cfg.stride}")
    print(f"Destino: {patches_path}\n")

    total_patches = 0
    ignoradas = 0

    for img_path in tqdm(imagens, desc="Extraindo patches"):
        img = cv2.imread(str(img_path))
        if img is None:
            ignoradas += 1
            continue

        h, w = img.shape[:2]
        if h < cfg.patch_size or w < cfg.patch_size:
            ignoradas += 1
            continue

        rel  = img_path.relative_to(dataset_path)
        stem = img_path.stem
        dest = patches_path / rel.parent
        dest.mkdir(parents=True, exist_ok=True)

        k = 0
        for y in range(0, h - cfg.patch_size + 1, cfg.stride):
            for x in range(0, w - cfg.patch_size + 1, cfg.stride):
                patch = img[y : y + cfg.patch_size, x : x + cfg.patch_size]
                cv2.imwrite(str(dest / f"{stem}_p{k}.png"), patch)
                k += 1
        total_patches += k

    print(f"\nPatches extraídos: {total_patches:,}")
    if ignoradas:
        print(f"Imagens ignoradas:{ignoradas}(não lidas ou menores que o patch)")
    print(f"Concluído → {patches_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extração de patches – BreaKHis")
    parser.add_argument("--dataset-dir", default=None, help="Caminho do dataset original")
    parser.add_argument("--patches-dir", default=None, help="Onde salvar os patches")
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.dataset_dir: cfg.dataset_dir = args.dataset_dir
    if args.patches_dir: cfg.patches_dir = args.patches_dir
    if args.patch_size: cfg.patch_size  = args.patch_size
    if args.stride: cfg.stride = args.stride

    extrair_patches(cfg)

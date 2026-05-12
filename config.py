from dataclasses import dataclass

@dataclass
class Config:
    # ∘₊✧──✧₊∘ Caminhos ∘₊✧──────✧₊∘
    dataset_dir: str = r"C:\Users\maria\Documents\PJT-1\BreaKHis_v1\histology_slides\breast"
    patches_dir: str = r"C:\Users\maria\Documents\PJT-1\BreaKHis_Patches"
    output_dir:  str = r"C:\Users\maria\Documents\PJT-1\resultados"

    # ∘₊✧──✧₊∘ Janela deslizante ∘₊✧──────✧₊∘
    patch_size: int = 96
    stride: int = 47
    magnification: str = "200X"

    # ∘₊✧──✧₊∘ Treinamento ∘₊✧──────✧₊∘
    num_epochs: int = 50
    batch_size: int = 16 
    lr_head: float = 5e-3
    lr_backbone:  float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0  # 0 para estabilidade no Windows
    patience: int = 10

    # ∘₊✧──✧₊∘ Protocolo K-Fold (70-15-15) ∘₊✧──────✧₊∘
    k_folds: int = 5
    seed: int = 42
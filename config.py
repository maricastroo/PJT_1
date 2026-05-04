from dataclasses import dataclass


@dataclass
class Config:
    # ── Caminhos ───────────────────────────────────────────────────────────────
    dataset_dir: str = r"C:\Users\maria\Documents\PJT-1\BreaKHis_v1\histology_slides\breast"
    patches_dir: str = r"C:\Users\maria\Documents\PJT-1\BreaKHis_Patches"
    output_dir:  str = r"C:\Users\maria\Documents\PJT-1\resultados"

    # ── Janela deslizante ──────────────────────────────────────────────────────
    patch_size: int = 94   # sugestão do orientador 
    stride:     int = 47   # sobreposição de 50 %

    # ── Treinamento ────────────────────────────────────────────────────────────
    num_epochs:   int   = 30
    batch_size:   int   = 64
    lr_head:      float = 5e-3   # cabeça classificadora (camadas novas)
    lr_backbone:  float = 1e-4   # layer4 do ResNet (fine-tuning suave)
    weight_decay: float = 1e-4
    patience:     int   = 7      # early stopping

    # Windows: num_workers > 0 pode causar problemas; use 0 se travar
    num_workers: int = 2

    # ── Protocolo LOGO ─────────────────────────────────────────────────────────
    # 0 = todos os pacientes (LOGO completo)
    # N > 0 = apenas N folds (modo rápido para testes)
    max_folds: int = 0

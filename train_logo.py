"""
AQUI DEPOIS – Treina ResNet-50 com protocolo LOGO (Leave-One-Patient-Out).

O LOGO garante que patches do mesmo paciente nunca aparecem
simultaneamente no treino e no teste — evita vazamento de dados.

Uso:
    python train_logo.py                         # todos os pacientes
    python train_logo.py --max-folds 5           # teste rápido (5 pacientes)
    python train_logo.py --patches-dir "C:/..."  # caminho personalizado
"""

import argparse
import copy
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import Config
from dataset import BreaKHisDataset, get_transforms, scan_patches
from model import build_resnet50

warnings.filterwarnings("ignore")


# ∘₊✧───✧₊∘ Época ∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """Roda um epoch de treino (optimizer != None) ou avaliação."""
    training = optimizer is not None
    model.train() if training else model.eval()

    loss_total, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_total += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

            probs = torch.softmax(logits, 1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return loss_total / total, correct / total, auc, all_labels, all_probs


# ∘₊✧───✧₊∘ Treino de um fold LOGO ∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘

def train_fold(train_paths, test_paths, cfg: Config, device, fold_idx: int) -> dict:
    """Treina e avalia um único fold do LOGO."""
    tf_train = get_transforms(train=True)
    tf_test  = get_transforms(train=False)

    train_ds = BreaKHisDataset(train_paths.tolist(), transform=tf_train)
    test_ds  = BreaKHisDataset(test_paths.tolist(),  transform=tf_test)

    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ∘₊✧──✧₊ Balanceamento de classes ∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘
    counts = np.zeros(2)
    for _, lbl in train_ds.samples:
        counts[lbl] += 1
    print(f"  Treino → Benignos: {int(counts[0]):,} | Malignos: {int(counts[1]):,}")

    # peso por amostra para o WeightedRandomSampler (batches balanceados 50/50)
    class_w = counts.sum() / (2 * counts + 1e-9)
    sample_w = [class_w[lbl] for _, lbl in train_ds.samples]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    # peso de classe para a loss (dupla proteção contra desbalanceamento)
    class_w_tensor = torch.tensor(class_w, dtype=torch.float).to(device)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler, # sem shuffle — sampler já embaralha
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = build_resnet50().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w_tensor)
    optimizer = optim.AdamW(
        [
            {"params": model.layer4.parameters(), "lr": cfg.lr_backbone},
            {"params": model.fc.parameters(), "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_loss, best_state, no_improve = float("inf"), None, 0

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc, _, _  = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, y_true, y_prob = run_epoch(model, test_loader, criterion, device=device)
        scheduler.step(val_loss)

        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            mark = " ←"
        else:
            no_improve += 1
            mark = ""

        print(
            f"  fold {fold_idx:03d} | ep {epoch:02d} | "
            f"tr {tr_loss:.4f}/{tr_acc:.4f} | "
            f"val {val_loss:.4f}/{val_acc:.4f}/{val_auc:.4f}"
            f"{mark} | {time.time()-t0:.1f}s"
        )

        if no_improve >= cfg.patience:
            print(f"Early stopping no fold {fold_idx} (época {epoch}).")
            break

    # avalia com os melhores pesos
    model.load_state_dict(best_state)
    _, _, _, y_true, y_prob = run_epoch(model, test_loader, criterion, device=device)
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    classes_presentes = len(set(y_true))
    auc = roc_auc_score(y_true, y_prob) if classes_presentes > 1 else None
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if classes_presentes == 1:
        classe = "Benigno" if y_true[0] == 0 else "Maligno"
        print(f"  Aviso: paciente de teste com apenas uma classe ({classe}) — AUC indefinido.")

    return {
        "acc": accuracy_score(y_true, y_pred),
        "auc": auc,   # None quando teste tem só uma classe
        "f1": f1,    # macro: média das duas classes
        "cm": confusion_matrix(y_true, y_pred).tolist(),
        "y_true": y_true,
        "y_prob": y_prob,
    }


# ∘₊✧───✧₊∘ LOGO principal ∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘

def run_logo(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nVarrendo patches em disco...")
    paths, labels, pids = scan_patches(cfg.patches_dir)
    unique_pids = np.unique(pids)
    print(f"Patches: {len(paths):,}")
    print(f"Pacientes: {len(unique_pids)}")
    print(f"Benignos: {(labels == 0).sum():,} | Malignos: {(labels == 1).sum():,}\n")

    logo   = LeaveOneGroupOut()
    splits = list(logo.split(paths, labels, pids))

    if cfg.max_folds > 0:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(splits), size=min(cfg.max_folds, len(splits)), replace=False)
        splits = [splits[i] for i in sorted(idx)]
        print(f"Modo rápido: {len(splits)} folds sorteados de {len(unique_pids)} pacientes.\n")

    all_results = []
    all_y_true, all_y_prob = [], []

    for fold_num, (train_idx, test_idx) in enumerate(splits, 1):
        test_pid = np.unique(pids[test_idx])[0]
        print(f"\n{'='*65}")
        print(f"Fold {fold_num}/{len(splits)} — paciente de teste: {test_pid}")
        print(f"  Treino: {len(train_idx):,} patches | Teste: {len(test_idx):,} patches")

        # ── Verificação LOGO: garante zero vazamento entre pacientes ──────────
        train_pids = set(pids[train_idx])
        test_pids  = set(pids[test_idx])
        overlap    = train_pids & test_pids
        if overlap:
            print(f"  AVISO: vazamento detectado! Pacientes em ambos os sets: {overlap}")
        else:
            print(f"  LOGO OK: nenhum paciente compartilhado entre treino e teste")

        result = train_fold(
            train_paths=paths[train_idx],
            test_paths=paths[test_idx],
            cfg=cfg,
            device=device,
            fold_idx=fold_num,
        )
        result["patient"] = test_pid
        all_results.append(result)
        all_y_true.extend(result["y_true"])
        all_y_prob.extend(result["y_prob"])

        auc_str = f"{result['auc']:.4f}" if result['auc'] is not None else "N/A"
        print(f"  → acc={result['acc']:.4f} | auc={auc_str} | f1={result['f1']:.4f}")

    # ∘₊✧──────✧₊∘ Métricas agregadas ∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘
    accs = [r["acc"] for r in all_results]
    aucs = [r["auc"] for r in all_results if r["auc"] is not None]  # ignora folds de 1 classe
    f1s  = [r["f1"]  for r in all_results]

    folds_sem_auc = sum(1 for r in all_results if r["auc"] is None)

    all_y_pred = [1 if p >= 0.5 else 0 for p in all_y_prob]
    cm_total   = confusion_matrix(all_y_true, all_y_pred)

    print(f"\n{'='*65}")
    print("RESULTADO FINAL — ResNet-50 | BreaKHis | Protocolo LOGO")
    print(f"  Acurácia : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  AUC-ROC  : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  ({folds_sem_auc} folds ignorados — paciente com 1 classe)")
    print(f"  F1-macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print()
    print(classification_report(all_y_true, all_y_pred, target_names=["Benigno", "Maligno"]))

    # ∘₊✧──────✧₊∘ Salva resultados ∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘
    summary = {
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "f1_mean":  float(np.mean(f1s)),  "f1_std":  float(np.std(f1s)),
        "folds_sem_auc": folds_sem_auc,
        "folds": all_results,
    }
    with open(out_dir / "logo_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _plot_results(accs, aucs, cm_total, out_dir)
    print(f"\nResultados salvos em: {out_dir}")


# ∘₊✧──────✧₊∘ Gráficos ∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘

def _plot_results(accs: list, aucs: list, cm: np.ndarray, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    folds = range(1, len(accs) + 1)

    axes[0].bar(folds, accs, color="steelblue", alpha=0.75)
    axes[0].axhline(np.mean(accs), color="red", ls="--",
                    label=f"Média = {np.mean(accs):.3f}")
    axes[0].set_title("Acurácia por Fold (LOGO)")
    axes[0].set_xlabel("Fold (paciente)"); axes[0].set_ylabel("Acurácia")
    axes[0].set_ylim(0, 1); axes[0].legend(); axes[0].grid(alpha=0.3)

    # aucs pode ter menos elementos que accs (folds com 1 classe ignorados)
    folds_auc = range(1, len(aucs) + 1)
    axes[1].bar(folds_auc, aucs, color="darkorange", alpha=0.75)
    axes[1].axhline(np.mean(aucs), color="red", ls="--",
                    label=f"Média = {np.mean(aucs):.3f}")
    axes[1].set_title("AUC-ROC por Fold (LOGO, folds válidos)")
    axes[1].set_xlabel("Fold (paciente)"); axes[1].set_ylabel("AUC-ROC")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(alpha=0.3)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
                xticklabels=["Benigno", "Maligno"],
                yticklabels=["Benigno", "Maligno"])
    axes[2].set_title("Matriz de Confusão Agregada")
    axes[2].set_xlabel("Predito"); axes[2].set_ylabel("Real")

    plt.suptitle("ResNet-50 — BreaKHis — Protocolo LOGO", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "logo_resultados.png", dpi=150, bbox_inches="tight")
    plt.show()


# ∘₊✧──────✧₊∘ Main ∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘∘₊✧──────✧₊∘

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOGO Training – BreaKHis ResNet-50")
    parser.add_argument("--patches-dir", default=None)
    parser.add_argument("--output-dir",  default=None)
    parser.add_argument("--max-folds",   type=int, default=None,
                        help="Limite de folds (0 = todos). Útil para testes rápidos.")
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--batch-size",  type=int, default=None)
    parser.add_argument("--workers",     type=int, default=None,
                        help="num_workers do DataLoader (0 = sem multiprocessing)")
    args = parser.parse_args()

    cfg = Config()
    if args.patches_dir:cfg.patches_dir = args.patches_dir
    if args.output_dir:cfg.output_dir  = args.output_dir
    if args.max_folds is not None: cfg.max_folds  = args.max_folds
    if args.epochs:cfg.num_epochs  = args.epochs
    if args.batch_size:cfg.batch_size  = args.batch_size
    if args.workers is not None:cfg.num_workers = args.workers

    run_logo(cfg)

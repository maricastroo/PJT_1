"""
Treina ResNet-50 com protocolo LOGO (Leave-One-Patient-Out).
Garante que patches do mesmo paciente nunca aparecem no treino e no teste ao mesmo tempo.

Uso:
    python train_logo.py                   # todos os pacientes
    python train_logo.py --max-folds 10    # teste rápido com 10 pacientes
"""

import argparse
import copy
import json
import time
import warnings
from pathlib import Path

import numpy as np
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


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """Roda uma época de treino (com optimizer) ou avaliação (sem optimizer)."""
    treinando = optimizer is not None
    model.train() if treinando else model.eval()

    loss_total, acertos, total = 0.0, 0, 0
    todos_labels, todas_probs = [], []

    ctx = torch.enable_grad() if treinando else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if treinando:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_total += loss.item() * imgs.size(0)
            acertos    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

            probs = torch.softmax(logits, 1)[:, 1].detach().cpu().numpy()
            todas_probs.extend(probs.tolist())
            todos_labels.extend(labels.cpu().numpy().tolist())

    # AUC só faz sentido se tiver as duas classes no conjunto
    auc = roc_auc_score(todos_labels, todas_probs) if len(set(todos_labels)) > 1 else 0.0
    return loss_total / total, acertos / total, auc, todos_labels, todas_probs


def train_fold(train_paths, test_paths, cfg: Config, device, fold_idx: int) -> dict:
    """Treina e avalia um único fold do LOGO."""
    train_ds = BreaKHisDataset(train_paths.tolist(), transform=get_transforms(train=True))
    test_ds  = BreaKHisDataset(test_paths.tolist(),  transform=get_transforms(train=False))

    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )

    # conta quantos patches de cada classe tem no treino
    counts = np.zeros(2)
    for _, lbl in train_ds.samples:
        counts[lbl] += 1
    print(f"  Treino → Benignos: {int(counts[0]):,} | Malignos: {int(counts[1]):,}")

    # peso por amostra pro sampler (deixa os batches 50/50)
    class_w  = counts.sum() / (2 * counts + 1e-9)
    sample_w = [class_w[lbl] for _, lbl in train_ds.samples]
    sampler  = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    # peso de classe na loss também, proteção dupla contra desbalanceamento
    class_w_tensor = torch.tensor(class_w, dtype=torch.float).to(device)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )

    model     = build_resnet50().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w_tensor)
    optimizer = optim.AdamW(
        [
            {"params": model.layer4.parameters(), "lr": cfg.lr_backbone},
            {"params": model.fc.parameters(),     "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )
    # reduz o lr pela metade se a val_loss parar de melhorar por 3 épocas
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    melhor_loss, melhor_estado = float("inf"), None

    for epoca in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc, _, _               = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, y_true, y_prob  = run_epoch(model, test_loader, criterion, device=device)
        scheduler.step(val_loss)

        if val_loss < melhor_loss:
            melhor_loss   = val_loss
            melhor_estado = copy.deepcopy(model.state_dict())
            marca = " ←"
        else:
            marca = ""

        print(
            f"  fold {fold_idx:03d} | ep {epoca:02d} | "
            f"tr {tr_loss:.4f}/{tr_acc:.4f} | "
            f"val {val_loss:.4f}/{val_acc:.4f}/{val_auc:.4f}"
            f"{marca} | {time.time()-t0:.1f}s"
        )

    # avalia com os melhores pesos que salvou durante o treino
    model.load_state_dict(melhor_estado)
    _, _, _, y_true, y_prob = run_epoch(model, test_loader, criterion, device=device)
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    classes_presentes = len(set(y_true))
    auc = roc_auc_score(y_true, y_prob) if classes_presentes > 1 else None
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # no BreaKHis cada paciente é só benigno ou só maligno
    if classes_presentes == 1:
        classe = "Benigno" if y_true[0] == 0 else "Maligno"
        print(f"  Aviso: paciente de teste com uma só classe ({classe}) — AUC indefinido.")

    return {
        "acc":    accuracy_score(y_true, y_pred),
        "auc":    auc,
        "f1":     f1,
        "cm":     confusion_matrix(y_true, y_pred).tolist(),
        "y_true": y_true,
        "y_prob": y_prob,
    }


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

    splits = list(LeaveOneGroupOut().split(paths, labels, pids))

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

        # garante que nenhum paciente aparece nos dois conjuntos ao mesmo tempo
        overlap = set(pids[train_idx]) & set(pids[test_idx])
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

        auc_str = f"{result['auc']:.4f}" if result["auc"] is not None else "N/A"
        print(f"  → acc={result['acc']:.4f} | auc={auc_str} | f1={result['f1']:.4f}")

    # métricas finais
    accs = [r["acc"] for r in all_results]
    f1s  = [r["f1"]  for r in all_results]
    folds_sem_auc = sum(1 for r in all_results if r["auc"] is None)

    all_y_pred = [1 if p >= 0.5 else 0 for p in all_y_prob]
    cm_total   = confusion_matrix(all_y_true, all_y_pred)

    # AUC global: junta todos os folds e calcula de uma vez 
    auc_global = roc_auc_score(all_y_true, all_y_prob) if len(set(all_y_true)) > 1 else None
    auc_str    = f"{auc_global:.4f}" if auc_global is not None else "N/A"

    print(f"\n{'='*65}")
    print("RESULTADO FINAL — ResNet-50 | BreaKHis | Protocolo LOGO")
    print(f"  Acurácia : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  AUC-ROC  : {auc_str}  (agregado — {folds_sem_auc} folds com 1 classe)")
    print(f"  F1-macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print()
    print(classification_report(all_y_true, all_y_pred, target_names=["Benigno", "Maligno"]))

    summary = {
        "acc_mean":      float(np.mean(accs)),
        "acc_std":       float(np.std(accs)),
        "auc_global":    float(auc_global) if auc_global is not None else None,
        "f1_mean":       float(np.mean(f1s)),
        "f1_std":        float(np.std(f1s)),
        "folds_sem_auc": folds_sem_auc,
        "folds":         all_results,
    }
    with open(out_dir / "logo_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResultados salvos em: {out_dir}")
    print("Para gerar os gráficos rode: python plot_resultados.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino LOGO – BreaKHis ResNet-50")
    parser.add_argument("--patches-dir", default=None)
    parser.add_argument("--output-dir",  default=None)
    parser.add_argument("--max-folds",   type=int, default=None,
                        help="Limite de folds (0 = todos). Útil pra testar rápido.")
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--batch-size",  type=int, default=None)
    parser.add_argument("--workers",     type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.patches_dir:           cfg.patches_dir = args.patches_dir
    if args.output_dir:            cfg.output_dir  = args.output_dir
    if args.max_folds is not None: cfg.max_folds   = args.max_folds
    if args.epochs:                cfg.num_epochs  = args.epochs
    if args.batch_size:            cfg.batch_size  = args.batch_size
    if args.workers is not None:   cfg.num_workers = args.workers

    run_logo(cfg)

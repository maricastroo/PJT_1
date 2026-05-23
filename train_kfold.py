"""
Treina os modelos com protocolo LOGO (70% treino | 15% validação | 15% teste),
dividindo os dados por paciente para evitar vazamento entre conjuntos.
Cada rodada usa uma seed diferente para garantir divisões distintas.

Uso:
    python train_kfold.py --model resnet
    python train_kfold.py --model efficientnet
    python train_kfold.py --model vgg
"""

import argparse
import copy
import json
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import BreaKHisDataset, get_transforms, scan_patches
from models import build_resnet50, build_efficientnet_b3, build_vgg16

# ∘₊✧──✧₊∘ LOGO: divisão por paciente ∘₊✧──────✧₊∘
def logo_split(paths: np.ndarray, labels: np.ndarray, patient_ids: np.ndarray,
               seed: int, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Divide patches em treino / validação / teste garantindo que todos os
    patches de um mesmo paciente estejam em apenas um dos três conjuntos.

    Proporção alvo  ->  70 % treino | 15 % validação | 15 % teste (em patches).
    A divisão é feita em nível de paciente e depois expandida para patches.
    """
    rng = np.random.default_rng(seed)

    unique_patients = np.unique(patient_ids)
    rng.shuffle(unique_patients)

    n = len(unique_patients)
    n_test  = max(1, round(n * test_ratio))
    n_val   = max(1, round(n * val_ratio))
    # O restante vai para treino
    n_train = n - n_test - n_val

    patients_test  = set(unique_patients[:n_test])
    patients_val   = set(unique_patients[n_test:n_test + n_val])
    patients_train = set(unique_patients[n_test + n_val:])

    train_mask = np.array([pid in patients_train for pid in patient_ids])
    val_mask   = np.array([pid in patients_val   for pid in patient_ids])
    test_mask  = np.array([pid in patients_test  for pid in patient_ids])

    return train_mask, val_mask, test_mask, {
        "n_patients_train": n_train,
        "n_patients_val":   n_val,
        "n_patients_test":  n_test,
        "patients_test":    sorted(patients_test),
    }


def run_epoch(model, loader, criterion, optimizer=None, device="cpu", desc=""):
    training = optimizer is not None
    model.train() if training else model.eval()
    loss_total, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    with torch.enable_grad() if training else torch.no_grad():
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, lbls)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = imgs.size(0)
            loss_total += loss.item() * bs
            correct += (logits.argmax(1) == lbls).sum().item()
            total += bs

            probs = torch.softmax(logits, 1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(lbls.cpu().numpy().tolist())

            pbar.set_postfix({
                'loss': f"{loss_total / total:.4f}",
                'acc':  f"{correct / total:.4f}"
            })

    return loss_total / total, correct / total, all_labels, all_probs


def get_optimizer(model, model_name, cfg):
    if model_name == "resnet":
        return optim.AdamW([
            {"params": model.layer4.parameters(), "lr": cfg.lr_backbone},
            {"params": model.fc.parameters(), "lr": cfg.lr_head},
        ], weight_decay=cfg.weight_decay)
    elif model_name == "efficientnet":
        return optim.AdamW([
            {"params": model.features[7].parameters(), "lr": cfg.lr_backbone},
            {"params": model.features[8].parameters(), "lr": cfg.lr_backbone},
            {"params": model.classifier.parameters(), "lr": cfg.lr_head},
        ], weight_decay=cfg.weight_decay)
    elif model_name == "vgg":
        backbone_params = [
            p for layer in list(model.features.children())[24:]
            for p in layer.parameters()
        ]
        return optim.AdamW([
            {"params": backbone_params, "lr": cfg.lr_backbone},
            {"params": model.classifier.parameters(), "lr": cfg.lr_head},
        ], weight_decay=cfg.weight_decay)


def run_kfold(cfg: Config, model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_dir = Path(cfg.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # scan_patches agora devolve também os IDs de paciente
    paths, labels, patient_ids = scan_patches(cfg.patches_dir)
    n_patients = len(np.unique(patient_ids))
    print(f"Total de Patches: {len(paths)} | Pacientes únicos: {n_patients}\n")

    resultados_folds = []

    # k_folds agora representa o número de rodadas LOGO com seeds distintas
    for fold in range(cfg.k_folds):
        current_seed = cfg.seed + fold
        print(f"\nTREINANDO {model_name.upper()} — RODADA LOGO {fold+1}/{cfg.k_folds} (Seed: {current_seed})\n")

        # Divisão por paciente (LOGO 70-15-15) 
        train_mask, val_mask, test_mask, split_info = logo_split(
            paths, labels, patient_ids, seed=current_seed
        )

        X_train, y_train = paths[train_mask],  labels[train_mask]
        X_val,   y_val   = paths[val_mask],    labels[val_mask]
        X_test,  y_test  = paths[test_mask],   labels[test_mask]

        # limita patches por paciente no treino para reduzir overfitting
        if cfg.max_patches_por_paciente > 0:
            pids_train = patient_ids[train_mask]
            rng = np.random.default_rng(current_seed)
            idx_selecionados = []
            for pid in np.unique(pids_train):
                idx_pid = np.where(pids_train == pid)[0]
                if len(idx_pid) > cfg.max_patches_por_paciente:
                    idx_pid = rng.choice(idx_pid, cfg.max_patches_por_paciente, replace=False)
                idx_selecionados.extend(idx_pid.tolist())
            X_train = X_train[idx_selecionados]
            y_train = y_train[idx_selecionados]

        print(f"Pacientes → Treino: {split_info['n_patients_train']} | "
              f"Val: {split_info['n_patients_val']} | "
              f"Teste: {split_info['n_patients_test']}")
        print(f"Patches   → Treino: {len(X_train)} | "
              f"Val: {len(X_val)} | Teste: {len(X_test)}")
        
        # Transforms específicos por modelo
        tf_train = get_transforms(train=True,  model_name=model_name)
        tf_test  = get_transforms(train=False, model_name=model_name)

        train_loader = DataLoader(
            BreaKHisDataset(X_train, transform=tf_train),
            batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers
        )
        val_loader = DataLoader(
            BreaKHisDataset(X_val, transform=tf_test),
            batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )
        test_loader = DataLoader(
            BreaKHisDataset(X_test, transform=tf_test),
            batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )

        counts  = np.bincount(y_train)
        class_w = torch.tensor(
            counts.sum() / (2.0 * counts + 1e-9), dtype=torch.float
        ).to(device)

        if   model_name == "resnet": model = build_resnet50().to(device)
        elif model_name == "efficientnet": model = build_efficientnet_b3().to(device)
        elif model_name == "vgg": model = build_vgg16().to(device)

        criterion = nn.CrossEntropyLoss(weight=class_w)
        optimizer = get_optimizer(model, model_name, cfg)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        best_loss, best_state, no_improve = float("inf"), None, 0
        historico = []

        for epoch in range(1, cfg.num_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, _, _   = run_epoch(model, train_loader, criterion, optimizer, device,
                                                 desc=f"Ep {epoch:02d} Treino")
            val_loss, val_acc, _, _ = run_epoch(model, val_loader,   criterion, device=device,
                                                 desc=f"Ep {epoch:02d} Validação")
            scheduler.step(val_loss)

            historico.append({
                "epoca":    epoch,
                "tr_loss":  round(tr_loss,  4),
                "tr_acc":   round(tr_acc,   4),
                "val_loss": round(val_loss, 4),
                "val_acc":  round(val_acc,  4),
            })

            if val_loss < best_loss:
                best_loss, best_state, no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
                mark = "*"
            else:
                no_improve += 1
                mark = ""

            print(
                f"  ep {epoch:02d}/{cfg.num_epochs} | "
                f"treino  loss={tr_loss:.4f}  acc={tr_acc:.4f} | "
                f"val  loss={val_loss:.4f}  acc={val_acc:.4f} | "
                f"{time.time()-t0:.1f}s {mark}"
            )
            if no_improve >= cfg.patience:
                print("Early stopping ativado.")
                break

        model.load_state_dict(best_state)
        print("Avaliando no conjunto de Teste...")
        _, test_acc, y_true, y_prob = run_epoch(
            model, test_loader, criterion, device=device, desc="Avaliando Teste"
        )
        print(f"Acurácia de Teste (Rodada {fold+1}): {test_acc:.4f}")

        resultados_folds.append({
            "fold":             fold + 1,
            "seed":             current_seed,
            "protocolo":        "LOGO-70-15-15",
            "n_patients_train": split_info["n_patients_train"],
            "n_patients_val":   split_info["n_patients_val"],
            "n_patients_test":  split_info["n_patients_test"],
            "patients_test":    split_info["patients_test"],
            "n_patches_train":  int(len(X_train)),
            "n_patches_val":    int(len(X_val)),
            "n_patches_test":   int(len(X_test)),
            "test_acc":         test_acc,
            "y_true":           y_true,
            "y_prob":           y_prob,
            "historico":        historico,
        })

        del model, optimizer, train_loader, val_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    resultado_final = {"modelo": model_name, "protocolo": "LOGO-70-15-15", "folds": resultados_folds}
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(resultado_final, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Resultados salvos em: {out_dir / 'results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet", "efficientnet", "vgg"], required=True)
    args = parser.parse_args()

    cfg = Config()
    run_kfold(cfg, args.model)
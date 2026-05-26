"""
Treina os modelos com protocolo LOGO (70% treino | 15% validação | 15% teste),
dividindo os dados por paciente para evitar vazamento entre conjuntos.
Cada rodada usa uma seed diferente para garantir divisões distintas.

Uso:
    python train_kfold.py --model resnet
    python train_kfold.py --model efficientnet
    python train_kfold.py --model vgg
    python train_kfold.py --model vgg --use-gap
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import BreaKHisDataset, get_transforms, scan_patches
from models import build_resnet50, build_efficientnet_b3, build_vgg16

# Configuração de construção de modelos
MODEL_BUILDERS = {
    "resnet": build_resnet50,
    "efficientnet": build_efficientnet_b3,
    "vgg": build_vgg16,
}

# Configuração de parâmetros treináveis por modelo
MODEL_TRAINABLE_PARAMS = {
    "resnet": lambda model: [
        {"params": model.layer4.parameters(), "name": "layer4"},
        {"params": model.fc.parameters(), "name": "fc"},
    ],
    "efficientnet": lambda model: [
        {"params": model.features[7].parameters(), "name": "features[7]"},
        {"params": model.features[8].parameters(), "name": "features[8]"},
        {"params": model.classifier.parameters(), "name": "classifier"},
    ],
    "vgg": lambda model: [
        {"params": layer.parameters(), "name": f"features[{i}]"}
        for i, layer in enumerate(list(model.features.children())[24:], start=24)
    ] + [{"params": model.classifier.parameters(), "name": "classifier"}],
}

def logo_split(paths: np.ndarray, labels: np.ndarray, patient_ids: np.ndarray,
               seed: int, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Divide patches em treino / validação / teste garantindo que todos os
    patches de um mesmo paciente estejam em apenas um dos três conjuntos.
    """
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    rng.shuffle(unique_patients)

    n = len(unique_patients)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))

    patients_test = set(unique_patients[:n_test])
    patients_val = set(unique_patients[n_test:n_test + n_val])
    patients_train = set(unique_patients[n_test + n_val:])

    train_mask = np.array([pid in patients_train for pid in patient_ids])
    val_mask = np.array([pid in patients_val for pid in patient_ids])
    test_mask = np.array([pid in patients_test for pid in patient_ids])

    return train_mask, val_mask, test_mask, {
        "n_patients_train": len(patients_train),
        "n_patients_val": len(patients_val),
        "n_patients_test": len(patients_test),
        "patients_test": sorted(patients_test),
    }

def run_epoch(model, loader, criterion, optimizer=None, device="cpu", desc=""):
    """
    Executa uma época de treino ou validação.
    
    CrossEntropyLoss espera:
    - Logits [batch_size, num_classes] do modelo
    - Labels [batch_size] tipo Long com valores {0, 1, ..., num_classes-1}
    """
    training = optimizer is not None
    model.train() if training else model.eval()
    loss_total, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    with torch.enable_grad() if training else torch.no_grad():
        for imgs, lbls in pbar:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True).long()
            
            logits = model(imgs)
            loss = criterion(logits, lbls)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = imgs.size(0)
            loss_total += loss.item() * bs
            correct += (logits.argmax(dim=1) == lbls).sum().item()
            total += bs

            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(lbls.cpu().numpy().tolist())

            pbar.set_postfix({
                'loss': f"{loss_total / total:.4f}",
                'acc': f"{correct / total:.4f}"
            })

    return loss_total / total, correct / total, all_labels, all_probs

def get_optimizer(model, model_name, cfg):
    """
    Cria otimizador com learning rates diferenciadas.
    Backbone usa lr_backbone, Classifier usa lr_head.
    """
    param_groups = MODEL_TRAINABLE_PARAMS[model_name](model)
    
    optimizer_params = []
    for group in param_groups:
        lr = cfg.lr_head if "classifier" in group["name"] or "fc" in group["name"] else cfg.lr_backbone
        optimizer_params.append({"params": group["params"], "lr": lr})
    
    return optim.AdamW(optimizer_params, weight_decay=cfg.weight_decay)

def create_dataloaders(X_train, X_val, X_test, model_name, cfg, device):
    """Cria DataLoaders para treino, validação e teste."""
    tf_train = get_transforms(train=True, model_name=model_name)
    tf_test = get_transforms(train=False, model_name=model_name)
    
    pin_memory = device.type == "cuda"
    
    train_loader = DataLoader(
        BreaKHisDataset(X_train, transform=tf_train),
        batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        BreaKHisDataset(X_val, transform=tf_test),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        BreaKHisDataset(X_test, transform=tf_test),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader

def limit_patches_per_patient(X_train, y_train, patient_ids, train_mask, max_patches, seed):
    """Limita o número de patches por paciente no conjunto de treino."""
    if max_patches <= 0:
        return X_train, y_train
    
    pids_train = patient_ids[train_mask]
    rng = np.random.default_rng(seed)
    idx_selecionados = []
    
    for pid in np.unique(pids_train):
        idx_pid = np.where(pids_train == pid)[0]
        if len(idx_pid) > max_patches:
            idx_pid = rng.choice(idx_pid, max_patches, replace=False)
        idx_selecionados.extend(idx_pid.tolist())
    
    return X_train[idx_selecionados], y_train[idx_selecionados]

def train_one_fold(cfg, model_name, fold, current_seed, paths, labels, patient_ids, device):
    print(f"\nTREINANDO {model_name.upper()} — RODADA LOGO {fold+1}/{cfg.k_folds} (Seed: {current_seed})\n")

    # Divisão por paciente
    train_mask, val_mask, test_mask, split_info = logo_split(
        paths, labels, patient_ids, seed=current_seed
    )

    X_train, y_train = paths[train_mask], labels[train_mask]
    X_val, y_val = paths[val_mask], labels[val_mask]
    X_test, y_test = paths[test_mask], labels[test_mask]

    # Limita patches por paciente
    X_train, y_train = limit_patches_per_patient(
        X_train, y_train, patient_ids, train_mask, 
        cfg.max_patches_por_paciente, current_seed
    )

    print(f"Pacientes → Treino: {split_info['n_patients_train']} | "
          f"Val: {split_info['n_patients_val']} | "
          f"Teste: {split_info['n_patients_test']}")
    print(f"Patches   → Treino: {len(X_train)} | "
          f"Val: {len(X_val)} | Teste: {len(X_test)}")

    # Cria DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, model_name, cfg, device
    )

    # Balanceamento de classes
    counts = np.bincount(y_train)
    class_w = torch.tensor(
        counts.sum() / (2.0 * counts + 1e-9), dtype=torch.float
    ).to(device)

    # Cria modelo
    model = MODEL_BUILDERS[model_name](num_classes=2, dropout=0.2).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.1)
    optimizer = get_optimizer(model, model_name, cfg)]
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Treinamento
    best_loss, best_state, no_improve = float("inf"), None, 0
    historico = []

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"Ep {epoch:02d} Treino"
        )
        val_loss, val_acc, _, _ = run_epoch(
            model, val_loader, criterion, device=device,
            desc=f"Ep {epoch:02d} Validação"
        )
        scheduler.step(val_loss)

        historico.append({
            "epoca": epoch,
            "tr_loss": round(tr_loss, 4),
            "tr_acc": round(tr_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            mark = "✓"
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

    # Avaliação final
    model.load_state_dict(best_state)
    print("Avaliando no conjunto de Teste...")
    _, test_acc, y_true, y_prob = run_epoch(
        model, test_loader, criterion, device=device, desc="Avaliando Teste"
    )
    print(f"Acurácia de Teste (Rodada {fold+1}): {test_acc:.4f}")

    # Limpeza de memória
    del model, optimizer, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold": fold + 1,
        "seed": current_seed,
        "protocolo": "LOGO-70-15-15",
        "n_patients_train": split_info["n_patients_train"],
        "n_patients_val": split_info["n_patients_val"],
        "n_patients_test": split_info["n_patients_test"],
        "patients_test": split_info["patients_test"],
        "n_patches_train": int(len(X_train)),
        "n_patches_val": int(len(X_val)),
        "n_patches_test": int(len(X_test)),
        "test_acc": test_acc,
        "y_true": y_true,
        "y_prob": y_prob,
        "historico": historico,
    }

def run_kfold(cfg: Config, model_name: str):
    """Executa treinamento com protocolo LOGO."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_dir = Path(cfg.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, labels, patient_ids = scan_patches(cfg.patches_dir)
    n_patients = len(np.unique(patient_ids))
    print(f"Total de Patches: {len(paths)} | Pacientes únicos: {n_patients}\n")

    resultados_folds = [
        train_one_fold(cfg, model_name, fold, cfg.seed + fold,
                      paths, labels, patient_ids, device)
        for fold in range(cfg.k_folds)
    ]

    # Salva resultados
    resultado_final = {
        "modelo": model_name,
        "protocolo": "LOGO-70-15-15",
        "dropout": 0.2,
        "folds": resultados_folds
    }

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(resultado_final, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Resultados salvos em: {out_dir / 'results.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet", "efficientnet", "vgg"], 
                       required=True, help="Modelo a treinar")
    args = parser.parse_args()

    cfg = Config()
    run_kfold(cfg, args.model)
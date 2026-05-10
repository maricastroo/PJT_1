"""
Lê o results.json gerado pelo train_logo.py e plota os gráficos.
Uso:
    python plot_resultados.py --model resnet
    python plot_resultados.py --model efficientnet
    python plot_resultados.py --model vgg
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, f1_score, accuracy_score
)

from config import Config


def plotar(model_name: str) -> None:
    cfg = Config()
    out_dir = Path(cfg.output_dir) / model_name
    json_path = out_dir / "results.json"

    print(f"Lendo: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    folds = data["folds"]

    # agrega y_true e y_prob de todos os folds
    all_y_true, all_y_prob = [], []
    for r in folds:
        all_y_true.extend(r["y_true"])
        all_y_prob.extend(r["y_prob"])

    all_y_pred = [1 if p >= 0.5 else 0 for p in all_y_prob]

    # métricas por fold
    accs = [r["acc"] for r in folds]
    f1s  = [r["f1"]  for r in folds]
    folds_sem_auc = sum(1 for r in folds if r["auc"] is None)

    # AUC global agregado
    auc_global = roc_auc_score(all_y_true, all_y_prob) if len(set(all_y_true)) > 1 else None
    auc_str = f"{auc_global:.4f}" if auc_global is not None else "N/A"

    # matriz de confusão agregada
    cm_total = confusion_matrix(all_y_true, all_y_pred)

    print(f"\n{'='*65}")
    print(f"RESULTADO FINAL — {model_name.upper()} | BreaKHis | Protocolo LOGO")
    print(f"Acurácia: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"AUC-ROC: {auc_str}  (agregado — {folds_sem_auc} folds com 1 classe)")
    print(f"F1-macro: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print()
    print(classification_report(all_y_true, all_y_pred, target_names=["Benigno", "Maligno"]))

    titulo = f"{model_name.upper()} — BreaKHis — Protocolo LOGO"
    folds_idx = range(1, len(accs) + 1)

    # gráfico 1: acurácia por fold 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(folds_idx, accs, color="steelblue", alpha=0.75)
    ax.axhline(np.mean(accs), color="red", ls="--", label=f"Média = {np.mean(accs):.3f}")
    ax.set_title(f"Acurácia por Fold (LOGO)\n{titulo}")
    ax.set_xlabel("Fold (paciente)")
    ax.set_ylabel("Acurácia")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    caminho = out_dir / "grafico_acuracia.png"
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Salvo: {caminho}")

    # gráfico 2: curva ROC agregada
    if auc_global is not None:
        fig, ax = plt.subplots(figsize=(7, 7))
        fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc_str}")
        ax.plot([0, 1], [0, 1], color="gray", ls="--", label="Aleatório")
        ax.set_title(f"Curva ROC Agregada\n{titulo}")
        ax.set_xlabel("Taxa de Falsos Positivos")
        ax.set_ylabel("Taxa de Verdadeiros Positivos")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        caminho = out_dir / "grafico_roc.png"
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Salvo: {caminho}")
    else:
        print("Curva ROC não gerada: todos os folds tinham apenas uma classe.")

    # gráfico 3: matriz de confusão 
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Benigno", "Maligno"],
                yticklabels=["Benigno", "Maligno"])
    ax.set_title(f"Matriz de Confusão Agregada\n{titulo}")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.tight_layout()
    caminho = out_dir / "grafico_confusao.png"
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Salvo: {caminho}")

    # atualiza o json com auc_global
    data["auc_global"] = float(auc_global) if auc_global is not None else None
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nJSON atualizado com auc_global={auc_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plota resultados LOGO — BreaKHis")
    parser.add_argument("--model", choices=["resnet", "efficientnet", "vgg"], required=True)
    args = parser.parse_args()
    plotar(args.model)

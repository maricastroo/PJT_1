"""
Lê o results.json gerado pelo train_kfold.py e plota os gráficos.
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
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

from config import Config


def plotar(model_name: str) -> None:
    cfg = Config()

    if model_name == "ensemble":
        comparison_path = Path(cfg.output_dir) / "ensemble_comparison" / "results.json"
        if not comparison_path.exists():
            raise FileNotFoundError(
                f"Execute 'python ensemble.py' antes de plotar o ensemble.\n"
                f"Esperado: {comparison_path}"
            )
        with open(comparison_path, "r", encoding="utf-8") as f:
            comparison = json.load(f)
        melhor = comparison["melhor_estrategia"]
        out_dir = Path(cfg.output_dir) / f"ensemble_{melhor}"
        print(f"Melhor estratégia: {melhor.upper()}")
    else:
        out_dir = Path(cfg.output_dir) / model_name

    json_path = out_dir / "results.json"

    print(f"Lendo: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    folds = data["folds"]
    titulo = f"{model_name.upper()} — BreaKHis — Protocolo LOGO"

    # agrega y_true e y_prob de todos os folds
    all_y_true, all_y_prob = [], []
    for r in folds:
        all_y_true.extend(r["y_true"])
        all_y_prob.extend(r["y_prob"])

    all_y_pred = [1 if p >= 0.5 else 0 for p in all_y_prob]

    accs = [r.get("test_acc", r.get("acc", 0.0)) for r in folds]
    auc_global = roc_auc_score(all_y_true, all_y_prob) if len(set(all_y_true)) > 1 else None
    auc_str = f"{auc_global:.4f}" if auc_global is not None else "N/A"
    cm_total = confusion_matrix(all_y_true, all_y_pred)

    print(f"\n{'='*65}")
    print(f"RESULTADO FINAL — {model_name.upper()} | BreaKHis | Protocolo LOGO")
    print(f"Acurácia média: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"AUC-ROC global: {auc_str}")
    print()
    print(classification_report(all_y_true, all_y_pred, target_names=["Benigno", "Maligno"]))

    # --- gráfico 1: curvas de treino e validação (loss) ---
    historicos = [r["historico"] for r in folds if "historico" in r]
    if historicos:
        n_epocas = max(len(h) for h in historicos)
        epocas = list(range(1, n_epocas + 1))

        # média e desvio por época entre os folds
        tr_losses  = np.full((len(historicos), n_epocas), np.nan)
        val_losses = np.full((len(historicos), n_epocas), np.nan)
        tr_accs    = np.full((len(historicos), n_epocas), np.nan)
        val_accs   = np.full((len(historicos), n_epocas), np.nan)

        for i, h in enumerate(historicos):
            for ep in h:
                idx = ep["epoca"] - 1
                tr_losses[i, idx]  = ep["tr_loss"]
                val_losses[i, idx] = ep["val_loss"]
                tr_accs[i, idx]    = ep["tr_acc"]
                val_accs[i, idx]   = ep["val_acc"]

        # curva de loss
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epocas, np.nanmean(tr_losses, axis=0),  color="steelblue",  lw=2, label="Treino")
        ax.plot(epocas, np.nanmean(val_losses, axis=0), color="darkorange", lw=2, label="Validação")
        ax.fill_between(epocas,
                        np.nanmean(tr_losses, axis=0) - np.nanstd(tr_losses, axis=0),
                        np.nanmean(tr_losses, axis=0) + np.nanstd(tr_losses, axis=0),
                        alpha=0.15, color="steelblue")
        ax.fill_between(epocas,
                        np.nanmean(val_losses, axis=0) - np.nanstd(val_losses, axis=0),
                        np.nanmean(val_losses, axis=0) + np.nanstd(val_losses, axis=0),
                        alpha=0.15, color="darkorange")
        ax.set_title(f"Curva de Loss (média dos folds)\n{titulo}")
        ax.set_xlabel("Época")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        caminho = out_dir / "grafico_loss.png"
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Salvo: {caminho}")

        # curva de acurácia
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epocas, np.nanmean(tr_accs, axis=0),  color="steelblue",  lw=2, label="Treino")
        ax.plot(epocas, np.nanmean(val_accs, axis=0), color="darkorange", lw=2, label="Validação")
        ax.fill_between(epocas,
                        np.nanmean(tr_accs, axis=0) - np.nanstd(tr_accs, axis=0),
                        np.nanmean(tr_accs, axis=0) + np.nanstd(tr_accs, axis=0),
                        alpha=0.15, color="steelblue")
        ax.fill_between(epocas,
                        np.nanmean(val_accs, axis=0) - np.nanstd(val_accs, axis=0),
                        np.nanmean(val_accs, axis=0) + np.nanstd(val_accs, axis=0),
                        alpha=0.15, color="darkorange")
        ax.set_title(f"Curva de Acurácia (média dos folds)\n{titulo}")
        ax.set_xlabel("Época")
        ax.set_ylabel("Acurácia")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        caminho = out_dir / "grafico_acc_curva.png"
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Salvo: {caminho}")
    else:
        print("Histórico de épocas não encontrado — retreine para gerar as curvas.")

    # --- gráfico 2: curva ROC agregada ---
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
        plt.close()
        print(f"Salvo: {caminho}")
    else:
        print("Curva ROC não gerada: apenas uma classe presente.")

    # --- gráfico 3: matriz de confusão ---
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plota resultados LOGO — BreaKHis")
    parser.add_argument("--model", choices=["resnet", "efficientnet", "vgg", "ensemble"], required=True)
    args = parser.parse_args()
    plotar(args.model)

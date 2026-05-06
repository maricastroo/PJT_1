"""
Lê logo_results.json já gerado e recalcula métricas + gráficos.
Uso:
    python plot_resultados.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report, f1_score, accuracy_score
)

from config import Config

cfg = Config()
out_dir = Path(cfg.output_dir)
json_path = out_dir / "logo_results.json"

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
print("RESULTADO FINAL — ResNet-50 | BreaKHis | Protocolo LOGO")
print(f"  Acurácia : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"  AUC-ROC  : {auc_str}  (agregado — {folds_sem_auc} folds com 1 classe)")
print(f"  F1-macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print()
print(classification_report(all_y_true, all_y_pred, target_names=["Benigno", "Maligno"]))

# gráficos
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
folds_idx = range(1, len(accs) + 1)

# acurácia por fold
axes[0].bar(folds_idx, accs, color="steelblue", alpha=0.75)
axes[0].axhline(np.mean(accs), color="red", ls="--", label=f"Média = {np.mean(accs):.3f}")
axes[0].set_title("Acurácia por Fold (LOGO)")
axes[0].set_xlabel("Fold (paciente)"); axes[0].set_ylabel("Acurácia")
axes[0].set_ylim(0, 1); axes[0].legend(); axes[0].grid(alpha=0.3)

# curva ROC agregada
fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc_str}")
axes[1].plot([0, 1], [0, 1], color="gray", ls="--", label="Aleatório")
axes[1].set_title("Curva ROC (agregada — todos os folds)")
axes[1].set_xlabel("Taxa de Falsos Positivos")
axes[1].set_ylabel("Taxa de Verdadeiros Positivos")
axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
axes[1].legend(); axes[1].grid(alpha=0.3)

# matriz de confusão
sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=["Benigno", "Maligno"],
            yticklabels=["Benigno", "Maligno"])
axes[2].set_title("Matriz de Confusão Agregada")
axes[2].set_xlabel("Predito"); axes[2].set_ylabel("Real")

plt.suptitle("ResNet-50 — BreaKHis — Protocolo LOGO", fontsize=14, fontweight="bold")
plt.tight_layout()
out_path = out_dir / "logo_resultados_v2.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nGráfico salvo em: {out_path}")

# atualiza o json 
data["auc_global"] = float(auc_global) if auc_global is not None else None
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"JSON atualizado com auc_global={auc_str}")

"""
Realiza a Fusão (Late Fusion / Soft Voting) das 3 redes neurais treinadas.
Executar após ter os JSONs da ResNet50, VGG16 e EfficientNetB3.

Uso:
python ensemble.py
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from config import Config

def carregar_resultados(caminho_json):
    with open(caminho_json, "r", encoding="utf-8") as f:
        return json.load(f)

def rodar_ensemble(cfg: Config):
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Carrega os resultados
    print("Carregando resultados dos modelos individuais...")
    res_resnet = carregar_resultados(out_dir / "resnet" / "results.json")
    res_effnet = carregar_resultados(out_dir / "efficientnet" / "results.json")
    res_vgg    = carregar_resultados(out_dir / "vgg" / "results.json")
    
    folds_resnet = res_resnet["folds"]
    folds_vgg = res_vgg["folds"]
    folds_effnet = res_effnet["folds"]
    
    # Garantia de que tudo tem o mesmo tamanho
    assert len(folds_resnet) == len(folds_vgg) == len(folds_effnet), "Erro: O número de folds é diferente entre as redes!"
    
    todos_y_true = []
    todos_y_prob_ensemble = []
    folds_ensemble = []
    
    print("\nCalculando a fusão (Soft-Voting) fold a fold...")
    
    for i in range(len(folds_resnet)):
        # Pega as probabilidades de cada modelo para o fold atual
        prob_rn = np.array(folds_resnet[i]["y_prob"])
        prob_vg = np.array(folds_vgg[i]["y_prob"])
        prob_ef = np.array(folds_effnet[i]["y_prob"])
        y_true = folds_resnet[i]["y_true"] # Gabarito é o mesmo para todos
        
        # FUSÃO: Média das probabilidades (Soft Voting)
        prob_ensemble = (prob_rn + prob_vg + prob_ef) / 3.0
        y_pred_fold = [1 if p >= 0.5 else 0 for p in prob_ensemble]

        # Calcular métricas dentro do fold (para o gráfico de barras)
        acc_fold = accuracy_score(y_true, y_pred_fold)
        f1_fold = f1_score(y_true, y_pred_fold, average="macro", zero_division=0)
        classes_presentes = len(set(y_true))
        auc_fold = roc_auc_score(y_true, prob_ensemble) if classes_presentes > 1 else None

        folds_ensemble.append({
            "patient": folds_resnet[i].get("patient", "N/A"),
            "acc": float(acc_fold),
            "auc": float(auc_fold) if auc_fold is not None else None,
            "f1": float(f1_fold),
            "cm": confusion_matrix(y_true, y_pred_fold).tolist(),
            "y_true": y_true,
            "y_prob": prob_ensemble.tolist()
        })
        
        todos_y_true.extend(y_true)
        todos_y_prob_ensemble.extend(prob_ensemble.tolist())

    # Converter probabilidades do ensemble em classes finais (Threshold de 0.5)
    todos_y_pred_ensemble = [1 if p >= 0.5 else 0 for p in todos_y_prob_ensemble]
    
    # Calcular as novas métricas globais do Ensemble
    acc_ensemble = accuracy_score(todos_y_true, todos_y_pred_ensemble)
    f1_ensemble = f1_score(todos_y_true, todos_y_pred_ensemble, average="macro", zero_division=0)
    
    classes_presentes = len(set(todos_y_true))
    auc_ensemble = roc_auc_score(todos_y_true, todos_y_prob_ensemble) if classes_presentes > 1 else None
    auc_str = f"{auc_ensemble:.4f}" if auc_ensemble is not None else "N/A"
    
    # Imprimir Relatório Final do Ensemble
    print("RESULTADO FINAL — ENSEMBLE (ResNet50 + VGG16 + EfficientNetB3)")
    print(f" Acurácia Global : {acc_ensemble:.4f}")
    print(f" AUC-ROC Global  : {auc_str}")
    print(f" F1-macro Global : {f1_ensemble:.4f}")
    print("\nRelatório de Classificação (Ensemble):")
    print(classification_report(todos_y_true, todos_y_pred_ensemble, target_names=["Benigno", "Maligno"]))
    
    print("Matriz de Confusão:")
    print(confusion_matrix(todos_y_true, todos_y_pred_ensemble))

    accs = [r["acc"] for r in folds_ensemble]
    f1s = [r["f1"] for r in folds_ensemble]
    folds_sem_auc = sum(1 for r in folds_ensemble if r["auc"] is None)
    
    summary = {
        "modelo": "ensemble",
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "auc_global": float(auc_ensemble) if auc_ensemble is not None else None,
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "folds_sem_auc": folds_sem_auc,
        "folds": folds_ensemble,
    }

    ensemble_dir = out_dir / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    
    with open(ensemble_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print(f"\nResultados do Ensemble salvos em: {ensemble_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble (Resnet50 + VGG16 + EfficientNetB3)")
    parser.add_argument("--output-dir",  default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.output_dir: cfg.output_dir  = args.output_dir
    
    rodar_ensemble(cfg)
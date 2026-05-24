"""
Compara 3 estratégias de ensemble (soma, produto, max) usando os JSONs
gerados pelo train_kfold.py para ResNet-50, EfficientNet-B3 e VGG-16.

Cada estratégia combina P(maligno) dos 3 modelos e decide a classe final:
    P(maligno) >= 0.5 → Maligno  |  P(maligno) < 0.5 → Benigno

Uso:
    python voting.py
    python voting.py --output-dir "caminho/customizado"
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from config import Config


def carregar_resultados(caminho: Path) -> dict:
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def soma(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Média das probabilidades — P(maligno) final é a média dos 3 modelos."""
    return (p1 + p2 + p3) / 3.0


def produto(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Produto normalizado — exige consenso dos 3 modelos para classificar como maligno."""
    num = p1 * p2 * p3
    den = num + (1 - p1) * (1 - p2) * (1 - p3)
    return num / (den + 1e-12)


def max_voting(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Máximo — basta um modelo confiante para classificar como maligno."""
    return np.maximum(np.maximum(p1, p2), p3)


ESTRATEGIAS = {
    "soma":    soma,
    "produto": produto,
    "max":     max_voting,
}


def rodar_voting(cfg: Config) -> None:
    out_dir = Path(cfg.output_dir)

    print("Carregando resultados dos modelos individuais...")
    res_resnet = carregar_resultados(out_dir / "resnet"       / "results.json")
    res_effnet = carregar_resultados(out_dir / "efficientnet" / "results.json")
    res_vgg    = carregar_resultados(out_dir / "vgg"          / "results.json")

    folds_rn = res_resnet["folds"]
    folds_ef = res_effnet["folds"]
    folds_vg = res_vgg["folds"]

    assert len(folds_rn) == len(folds_ef) == len(folds_vg), \
        "Número de folds diferente entre os modelos."

    # Acumula probabilidades e labels de todos os folds por estratégia
    y_true_global = []
    y_prob_por_estrategia = {nome: [] for nome in ESTRATEGIAS}

    for i in range(len(folds_rn)):
        p_rn   = np.array(folds_rn[i]["y_prob"])
        p_ef   = np.array(folds_ef[i]["y_prob"])
        p_vg   = np.array(folds_vg[i]["y_prob"])
        y_true = folds_rn[i]["y_true"]

        y_true_global.extend(y_true)

        for nome, fn in ESTRATEGIAS.items():
            p_maligno = fn(p_rn, p_ef, p_vg)
            y_prob_por_estrategia[nome].extend(p_maligno.tolist())

    # Compara as estratégias
    print(f"\n{'Estratégia':<12} {'AUC-ROC':>10} {'F1-macro':>10} {'Acurácia':>10}")
    print("-" * 46)

    saida = {}
    for nome in ESTRATEGIAS:
        p_maligno = np.array(y_prob_por_estrategia[nome])
        p_benigno = 1 - p_maligno
        y_pred    = (p_maligno >= 0.5).astype(int).tolist()

        acc = accuracy_score(y_true_global, y_pred)
        f1  = f1_score(y_true_global, y_pred, average="macro", zero_division=0)
        auc = roc_auc_score(y_true_global, p_maligno) if len(set(y_true_global)) > 1 else None

        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"{nome:<12} {auc_str:>10} {f1:>10.4f} {acc:>10.4f}")

        saida[nome] = {
            "auc_global": float(auc) if auc is not None else None,
            "f1_global":  float(f1),
            "acc_global": float(acc),
        }

    # Melhor estratégia por AUC-ROC
    aucs = {n: v["auc_global"] for n, v in saida.items() if v["auc_global"] is not None}
    melhor = max(aucs, key=aucs.get) if aucs else "N/A"

    print("-" * 46)
    print(f"\nMelhor estratégia por AUC-ROC: {melhor.upper()}")

    # Exemplo de saída para o último fold, primeira amostra
    i = len(folds_rn) - 1
    p_rn = np.array(folds_rn[i]["y_prob"])
    p_ef = np.array(folds_ef[i]["y_prob"])
    p_vg = np.array(folds_vg[i]["y_prob"])

    print("\nExemplo — primeira amostra do último fold:")
    print(f"  ResNet:      P(benigno)={1-p_rn[0]:.3f}  P(maligno)={p_rn[0]:.3f}")
    print(f"  EfficientNet:P(benigno)={1-p_ef[0]:.3f}  P(maligno)={p_ef[0]:.3f}")
    print(f"  VGG:         P(benigno)={1-p_vg[0]:.3f}  P(maligno)={p_vg[0]:.3f}")
    for nome, fn in ESTRATEGIAS.items():
        p_m = fn(p_rn[:1], p_ef[:1], p_vg[:1])[0]
        classe = "Maligno" if p_m >= 0.5 else "Benigno"
        print(f"  {nome:<10}  P(benigno)={1-p_m:.3f}  P(maligno)={p_m:.3f}  → {classe}")

    # Salva resultado
    voting_dir = out_dir / "voting"
    voting_dir.mkdir(parents=True, exist_ok=True)
    resultado_final = {"melhor_estrategia": melhor, "criterio": "auc_global", "estrategias": saida}
    with open(voting_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(resultado_final, f, indent=2, ensure_ascii=False)

    print(f"\nResultados salvos em: {voting_dir / 'results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voting Ensemble — soma, produto, max")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.output_dir:
        cfg.output_dir = args.output_dir

    rodar_voting(cfg)

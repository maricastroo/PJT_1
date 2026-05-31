"""
Ensemble e Comparação de Modelos
Combina os 3 modelos (ResNet50, EfficientNetB3, VGG16) usando diferentes estratégias.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score
)
from config import Config

def soma(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Média das probabilidades (Soft Voting)."""
    return (p1 + p2 + p3) / 3.0


def produto(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Produto normalizado - exige consenso dos 3 modelos."""
    num = p1 * p2 * p3
    den = num + (1 - p1) * (1 - p2) * (1 - p3)
    return num / (den + 1e-12)


def max_voting(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Max voting - cada modelo vota em uma classe, vence a classe com mais votos."""
    v1 = (p1 >= 0.5).astype(int)
    v2 = (p2 >= 0.5).astype(int)
    v3 = (p3 >= 0.5).astype(int)
    return (v1 + v2 + v3 >= 2).astype(float)


ESTRATEGIAS = {
    "soma": soma,
    "produto": produto,
    "max_voting": max_voting,
}


def carregar_resultados(caminho: Path) -> dict:
    """Carrega arquivo JSON de resultados."""
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def calcular_auc(y_true, y_prob):
    """Calcula AUC-ROC, retorna None se houver apenas uma classe."""
    if len(set(y_true)) > 1:
        return roc_auc_score(y_true, y_prob)
    return None


def calcular_metricas(y_true, y_prob, threshold=0.5):
    """Calcula accuracy, F1, AUC e confusion matrix."""
    y_pred = (np.array(y_prob) >= threshold).astype(int).tolist()
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc = calcular_auc(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "acc": float(acc),
        "f1": float(f1),
        "auc": float(auc) if auc is not None else None,
        "cm": cm.tolist(),
        "y_pred": y_pred
    }


def comparar_estrategias(cfg: Config, mostrar_detalhes: bool = True):
    """Compara modelos individuais e todas as estratégias de ensemble."""
    out_dir = Path(cfg.output_dir)

    print("Carregando resultados dos modelos individuais...\n")
    res_resnet = carregar_resultados(out_dir / "resnet" / "results.json")
    res_effnet = carregar_resultados(out_dir / "efficientnet" / "results.json")
    res_vgg = carregar_resultados(out_dir / "vgg" / "results.json")

    folds_rn = res_resnet["folds"]
    folds_ef = res_effnet["folds"]
    folds_vg = res_vgg["folds"]

    assert len(folds_rn) == len(folds_ef) == len(folds_vg), \
        "ERRO: Número de folds diferente entre os modelos!"

    n_folds = len(folds_rn)
    
    # Tabela por fold
    if mostrar_detalhes:
        print("Tabela de AUC-ROC por Fold:")
        print("-" * 110)
        print(f"{'Fold':<20} | {'ResNet50':<12} | {'EfficientB3':<12} | {'VGG16':<12} | "
              f"{'Soma':<12} | {'Produto':<12} | {'Max Vote':<12} |")
        print("-" * 110)

    # Acumuladores globais
    y_true_global = []
    probs_globais = {
        "resnet": [],
        "efficientnet": [],
        "vgg": [],
    }
    probs_estrategias = {nome: [] for nome in ESTRATEGIAS}

    # Processar cada fold
    for i in range(n_folds):
        y_true = folds_rn[i]["y_true"]
        p_rn = np.array(folds_rn[i]["y_prob"])
        p_ef = np.array(folds_ef[i]["y_prob"])
        p_vg = np.array(folds_vg[i]["y_prob"])

        # Acumular para cálculo global
        y_true_global.extend(y_true)
        probs_globais["resnet"].extend(p_rn.tolist())
        probs_globais["efficientnet"].extend(p_ef.tolist())
        probs_globais["vgg"].extend(p_vg.tolist())

        if mostrar_detalhes:
            # AUC por modelo no fold
            auc_rn = calcular_auc(y_true, p_rn)
            auc_ef = calcular_auc(y_true, p_ef)
            auc_vg = calcular_auc(y_true, p_vg)

            # AUC por estratégia no fold
            p_soma = soma(p_rn, p_ef, p_vg)
            p_prod = produto(p_rn, p_ef, p_vg)
            p_max = max_voting(p_rn, p_ef, p_vg)

            auc_soma = calcular_auc(y_true, p_soma)
            auc_prod = calcular_auc(y_true, p_prod)
            auc_max = calcular_auc(y_true, p_max)

            def fmt(v): return f"{v:.4f}" if v is not None else "N/A"

            print(f"Fold {i+1} (70-15-15)    | {fmt(auc_rn):<12} | {fmt(auc_ef):<12} | "
                  f"{fmt(auc_vg):<12} | {fmt(auc_soma):<12} | {fmt(auc_prod):<12} | "
                  f"{fmt(auc_max):<12} |")

        # Acumular probabilidades das estratégias
        for nome, fn in ESTRATEGIAS.items():
            probs_estrategias[nome].extend(fn(p_rn, p_ef, p_vg).tolist())

    if mostrar_detalhes:
        print("-" * 110)

    # Tabela global
    print(f"\nResultados Globais ({n_folds} folds):")
    print("-" * 60)
    print(f"{'Modelo/Estratégia':<20} {'AUC-ROC':>10} {'F1-macro':>10} {'Acurácia':>10}")
    print("-" * 60)

    resultados = {}

    # Avaliar modelos individuais
    for nome, probs in probs_globais.items():
        metricas = calcular_metricas(y_true_global, probs)
        resultados[nome] = metricas
        auc_str = f"{metricas['auc']:.4f}" if metricas['auc'] is not None else "N/A"
        print(f"{nome:<20} {auc_str:>10} {metricas['f1']:>10.4f} {metricas['acc']:>10.4f}")

    print("-" * 60)

    # Avaliar estratégias de ensemble
    for nome, probs in probs_estrategias.items():
        metricas = calcular_metricas(y_true_global, probs)
        resultados[nome] = metricas
        auc_str = f"{metricas['auc']:.4f}" if metricas['auc'] is not None else "N/A"
        print(f"{nome.upper():<20} {auc_str:>10} {metricas['f1']:>10.4f} {metricas['acc']:>10.4f}")

    print("-" * 60)

    # Melhor estratégia
    aucs = {n: v["auc"] for n, v in resultados.items() if v["auc"] is not None}
    melhor = max(aucs, key=aucs.get) if aucs else "N/A"
    print(f"\nMelhor estratégia por AUC-ROC: {melhor.upper()}")
    print(f"   AUC: {aucs[melhor]:.4f} | F1: {resultados[melhor]['f1']:.4f} | "
          f"Acc: {resultados[melhor]['acc']:.4f}")

    # Salvar resultados
    voting_dir = out_dir / "ensemble_comparison"
    voting_dir.mkdir(parents=True, exist_ok=True)

    resultado_final = {
        "melhor_estrategia": melhor,
        "criterio": "auc_global",
        "n_folds": n_folds,
        "resultados": {
            nome: {
                "auc": metricas["auc"],
                "f1": metricas["f1"],
                "acc": metricas["acc"]
            }
            for nome, metricas in resultados.items()
        }
    }

    with open(voting_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(resultado_final, f, indent=2, ensure_ascii=False)

    print(f"\nResultados salvos em: {voting_dir / 'results.json'}")
    
    return melhor, resultados


def gerar_ensemble(cfg: Config, estrategia: str = "soma"):
    """Gera arquivo de resultados detalhado para uma estratégia específica."""
    out_dir = Path(cfg.output_dir)

    if estrategia not in ESTRATEGIAS:
        raise ValueError(f"Estratégia '{estrategia}' inválida. Opções: {list(ESTRATEGIAS.keys())}")

    print(f"Gerando ensemble com estratégia: {estrategia.upper()}\n")

    # Carregar resultados
    res_resnet = carregar_resultados(out_dir / "resnet" / "results.json")
    res_effnet = carregar_resultados(out_dir / "efficientnet" / "results.json")
    res_vgg = carregar_resultados(out_dir / "vgg" / "results.json")

    folds_rn = res_resnet["folds"]
    folds_ef = res_effnet["folds"]
    folds_vg = res_vgg["folds"]

    assert len(folds_rn) == len(folds_ef) == len(folds_vg), \
        "ERRO: Número de folds diferente entre os modelos!"

    fn_estrategia = ESTRATEGIAS[estrategia]
    
    # Processar cada fold
    todos_y_true = []
    todos_y_prob = []
    folds_ensemble = []

    for i in range(len(folds_rn)):
        y_true = folds_rn[i]["y_true"]
        p_rn = np.array(folds_rn[i]["y_prob"])
        p_ef = np.array(folds_ef[i]["y_prob"])
        p_vg = np.array(folds_vg[i]["y_prob"])

        # Aplicar estratégia
        p_ensemble = fn_estrategia(p_rn, p_ef, p_vg)
        
        # Calcular métricas do fold
        metricas_fold = calcular_metricas(y_true, p_ensemble)

        folds_ensemble.append({
            "fold": i + 1,
            "seed": folds_rn[i].get("seed", "N/A"),
            "acc": metricas_fold["acc"],
            "f1": metricas_fold["f1"],
            "auc": metricas_fold["auc"],
            "cm": metricas_fold["cm"],
            "y_true": y_true,
            "y_prob": p_ensemble.tolist()
        })

        todos_y_true.extend(y_true)
        todos_y_prob.extend(p_ensemble.tolist())

    # Métricas globais
    metricas_globais = calcular_metricas(todos_y_true, todos_y_prob)

    print(f"RESULTADO FINAL - ENSEMBLE ({estrategia.upper()})")
    print(f"   Acurácia: {metricas_globais['acc']:.4f}")
    auc_str = f"{metricas_globais['auc']:.4f}" if metricas_globais['auc'] else "N/A"
    print(f"   AUC-ROC:  {auc_str}")
    print(f"   F1-macro: {metricas_globais['f1']:.4f}")

    print("\nRelatório de Classificação:")
    print(classification_report(
        todos_y_true, metricas_globais['y_pred'],
        target_names=["Benigno", "Maligno"]
    ))

    print("Matriz de Confusão:")
    print(metricas_globais['cm'])

    # Salvar resultados
    ensemble_dir = out_dir / f"ensemble_{estrategia}"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    accs = [f["acc"] for f in folds_ensemble]
    f1s = [f["f1"] for f in folds_ensemble]

    summary = {
        "modelo": f"ensemble_{estrategia}",
        "estrategia": estrategia,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "auc_global": metricas_globais["auc"],
        "folds": folds_ensemble,
    }

    with open(ensemble_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResultados salvos em: {ensemble_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble e comparação de modelos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python ensemble.py                           # Compara todas as estratégias
  python ensemble.py --strategy soma           # Gera ensemble com média
  python ensemble.py --strategy produto        # Gera ensemble com produto
  python ensemble.py --compare-only            # Só compara, não gera arquivos detalhados
        """
    )
    parser.add_argument(
        "--strategy",
        choices=list(ESTRATEGIAS.keys()),
        help="Estratégia específica para gerar ensemble detalhado"
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Apenas compara estratégias sem gerar arquivos detalhados"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretório customizado de saída"
    )

    args = parser.parse_args()

    cfg = Config()
<<<<<<< HEAD
    if args.output_dir: cfg.output_dir  = args.output_dir
    
    rodar_ensemble(cfg)
=======
    if args.output_dir:
        cfg.output_dir = args.output_dir

    if args.compare_only:
        # Só comparar
        comparar_estrategias(cfg, mostrar_detalhes=True)
    elif args.strategy:
        # Gerar ensemble específico
        gerar_ensemble(cfg, args.strategy)
    else:
        # Comparar + gerar melhor
        melhor, resultados = comparar_estrategias(cfg, mostrar_detalhes=True)
        print(f"\nGerando ensemble detalhado para a melhor estratégia: {melhor.upper()}\n")
        gerar_ensemble(cfg, melhor)
>>>>>>> 91746db9359496a375cc10f2c19fc57333a79b5d

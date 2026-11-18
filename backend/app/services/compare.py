"""Comparação estatística entre experimentos.

Quando dois experimentos usam o MESMO dataset, alvo, separação treino/teste,
número de folds e semente, os folds da validação cruzada são idênticos — logo
as métricas por fold são pareadas e podemos testar se a diferença é
estatisticamente significativa.

Usamos o t-test pareado com a correção de Nadeau & Bengio (2003), que
compensa a sobreposição entre os conjuntos de treino dos folds (o t-test
comum subestima a variância e gera falsos positivos).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

from .training import LOWER_IS_BETTER


def _same_folds(a: dict, b: dict) -> tuple[bool, str]:
    if not a.get("cv") or not b.get("cv"):
        return False, "Algum experimento não tem validação cruzada."
    pa, pb = a["preprocessing"], b["preprocessing"]
    keys = ["target", "split_strategy", "test_size", "random_state", "time_column",
            "numeric_imputation", "categorical_imputation"]
    if a["dataset_id"] != b["dataset_id"]:
        return False, "Datasets diferentes."
    for k in keys[:5]:
        if pa.get(k) != pb.get(k):
            return False, f"Configuração de separação diferente ({k})."
    # remoção de linhas muda as amostras
    for k in keys[5:]:
        if (pa.get(k) == "drop_rows") != (pb.get(k) == "drop_rows"):
            return False, "Remoção de linhas com ausentes difere entre os experimentos."
    if a["cv"]["folds"] != b["cv"]["folds"] or a["cv"]["splitter"] != b["cv"]["splitter"]:
        return False, "Número de folds ou estratégia de CV diferentes."
    return True, ""


def paired_test(a: dict, b: dict, metric: str) -> dict:
    ok, reason = _same_folds(a, b)
    if not ok:
        return {"available": False, "reason": reason}
    ma, mb = a["cv"]["metrics"].get(metric), b["cv"]["metrics"].get(metric)
    if not ma or not mb:
        return {"available": False, "reason": f"Métrica '{metric}' indisponível na validação cruzada."}
    fa = np.array([np.nan if v is None else v for v in ma["folds"]], dtype=float)
    fb = np.array([np.nan if v is None else v for v in mb["folds"]], dtype=float)
    d = fa - fb
    d = d[~np.isnan(d)]
    k = len(d)
    if k < 2:
        return {"available": False, "reason": "Folds insuficientes."}
    mean_d = float(d.mean())
    var_d = float(d.var(ddof=1))
    # razão n_teste/n_treino em cada fold de k-fold = 1/(k-1)
    # (em TimeSeriesSplit a razão varia por fold; usamos a mesma aproximação)
    ratio = 1.0 / (k - 1)
    note = ("Em TimeSeriesSplit os folds têm tamanhos diferentes: a correção é aproximada."
            if a["cv"]["splitter"] == "TimeSeriesSplit" else None)
    if var_d == 0:
        t_corr, p_corr = (0.0, 1.0) if mean_d == 0 else (np.inf, 0.0)
        t_std, p_std = t_corr, p_corr
    else:
        t_corr = mean_d / np.sqrt((1 / k + ratio) * var_d)
        p_corr = float(2 * sps.t.sf(abs(t_corr), df=k - 1))
        t_std, p_std = sps.ttest_rel(fa[~np.isnan(fa - fb)], fb[~np.isnan(fa - fb)])
    better = None
    if mean_d != 0:
        a_better = mean_d < 0 if metric in LOWER_IS_BETTER else mean_d > 0
        better = a["id"] if a_better else b["id"]
    return {
        "available": True, "metric": metric, "k": k,
        "mean_diff": mean_d, "folds_diff": d.tolist(),
        "t_corrected": float(t_corr), "p_corrected": float(p_corr),
        "t_standard": float(t_std), "p_standard": float(p_std),
        "significant_5pct": bool(p_corr < 0.05),
        "better": better, "note": note,
    }


def compare(experiments: list[dict], metric: str | None = None) -> dict:
    if not experiments:
        return {"rows": [], "pairs": []}
    tasks = {e["task"] for e in experiments}
    metric = metric or experiments[0]["primary_metric"]
    pairs = []
    if len(tasks) == 1:
        for i, a in enumerate(experiments):
            for b in experiments[i + 1:]:
                pairs.append({"a": a["id"], "b": b["id"], "a_name": a["name"], "b_name": b["name"],
                              **paired_test(a, b, metric)})
    return {"metric": metric, "same_task": len(tasks) == 1, "pairs": pairs}

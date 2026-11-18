"""Reproduz os experimentos documentados em resultados/.

Executa, pela própria API do ML Lab (mesmo código usado pela interface), a
configuração do teste de ponta a ponta feito no navegador:

1. Classificação de `falha_24h` no dataset SINTÉTICO de motores:
   pré-processamento padrão da interface (mediana, padronização, one-hot,
   separação estratificada 80/20, semente 42), removendo as colunas que o
   detector de vazamento aponta — como o botão "Remover atributos suspeitos".
   Os 4 classificadores são treinados com hiperparâmetros padrão e validação
   cruzada de 5 folds ("Treinar todos e comparar").
2. Regressão de `temperatura_rolamento_c` no mesmo dataset, com separação
   temporal por `timestamp`, como no teste de fumaça do backend. As colunas
   `falha_24h` e `modo_falha` são excluídas porque descrevem um evento
   FUTURO (próximas 24 h) e não estariam disponíveis no momento da previsão.

Uso (a partir de backend/):
    python -m scripts.reproduzir_resultados ../resultados
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MLLAB_DATA_DIR", tempfile.mkdtemp(prefix="mllab_repro_"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

CLS_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]
REG_METRICS = ["mae", "rmse", "r2", "mape"]


def _ok(r):
    if r.status_code != 200:
        raise SystemExit(f"Erro {r.status_code}: {r.text}")
    return r.json()


def _row(exp: dict, metrics: list[str]) -> dict:
    cv = exp.get("cv") or {}
    row = {"experimento": exp["model_name"]}
    for m in metrics:
        row[f"teste_{m}"] = exp["metrics"]["test"].get(m)
        row[f"treino_{m}"] = exp["metrics"]["train"].get(m)
        row[f"baseline_{m}"] = exp["metrics"]["baseline"].get(m)
        if cv and m in cv["metrics"]:
            row[f"cv_media_{m}"] = cv["metrics"][m]["mean"]
            row[f"cv_dp_{m}"] = cv["metrics"][m]["std"]
    row["tempo_ajuste_s"] = exp["timing"]["fit_s"]
    row["tempo_cv_s"] = exp["timing"]["cv_s"]
    row["n_treino"] = exp["n_train"]
    row["n_teste"] = exp["n_test"]
    row["n_atributos_finais"] = len(exp["final_features"])
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join("" if r.get(c) is None else (f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]))
                              for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(exp: dict) -> dict:
    """Experimento sem os vetores grandes (probabilidades, previsões)."""
    keep = {k: v for k, v in exp.items() if k not in ("results",)}
    res = exp["results"]
    keep["results"] = {k: res.get(k) for k in ("task", "class_labels", "positive_class", "n_classes",
                                                "confusion_matrix", "per_class")}
    if res.get("roc_curve"):
        keep["results"]["roc_auc_curva"] = res["roc_curve"]["auc"]
    if res.get("residuals"):
        keep["results"]["residuos"] = {"media": res["residuals"]["mean"], "dp": res["residuals"]["std"]}
    return keep


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c = TestClient(app)
    did = _ok(c.post("/api/samples/motor_industrial_sintetico"))["id"]

    # ------------------------------------------------------------------ #
    # 1) Classificação: falha_24h                                         #
    # ------------------------------------------------------------------ #
    info = _ok(c.get(f"/api/datasets/{did}/target-info", params={"target": "falha_24h"}))
    pre = {"target": "falha_24h", "time_column": "timestamp", "features": info["default_features"]}
    prev = _ok(c.post("/api/preprocess/preview", json={"dataset_id": did, "preprocessing": pre}))
    leaky = [x["feature"] for x in prev["leakage"]]
    pre["features"] = [f for f in pre["features"] if f not in leaky]

    cls = []
    for model in ["logistic_regression", "decision_tree", "random_forest", "xgboost"]:
        cls.append(_ok(c.post("/api/train", json={
            "dataset_id": did, "model": model, "preprocessing": pre,
            "cv": {"enabled": True, "folds": 5}})))
    cmp_cls = _ok(c.post("/api/experiments/compare", json={"ids": [e["id"] for e in cls], "metric": "f1"}))

    # ------------------------------------------------------------------ #
    # 2) Regressão: temperatura_rolamento_c                               #
    # ------------------------------------------------------------------ #
    info_r = _ok(c.get(f"/api/datasets/{did}/target-info", params={"target": "temperatura_rolamento_c"}))
    pre_r = {"target": "temperatura_rolamento_c", "split_strategy": "temporal", "time_column": "timestamp",
             "features": [f for f in info_r["default_features"] if f not in ("falha_24h", "modo_falha")]}
    prev_r = _ok(c.post("/api/preprocess/preview", json={"dataset_id": did, "preprocessing": pre_r}))
    reg = []
    for model in ["linear_regression", "decision_tree", "random_forest", "xgboost"]:
        reg.append(_ok(c.post("/api/train", json={
            "dataset_id": did, "model": model, "preprocessing": pre_r,
            "cv": {"enabled": True, "folds": 5}})))
    cmp_reg = _ok(c.post("/api/experiments/compare", json={"ids": [e["id"] for e in reg], "metric": "rmse"}))

    # ------------------------------------------------------------------ #
    # Saídas                                                              #
    # ------------------------------------------------------------------ #
    _write_csv(out / "classificacao_falha_24h.csv", [_row(e, CLS_METRICS) for e in cls])
    _write_csv(out / "regressao_temperatura_rolamento.csv", [_row(e, REG_METRICS) for e in reg])
    pairs = lambda cmp: [{k: p.get(k) for k in ("a_name", "b_name", "metric", "k", "mean_diff", "t_corrected",  # noqa: E731
                                               "p_corrected", "p_standard", "significant_5pct", "reason")}
                         for p in cmp["pairs"]]
    meta = {
        "dataset": "motor_industrial_sintetico (6000 linhas, gerado com seed=7)",
        "classificacao": {
            "config_preprocessamento": pre, "vazamento_detectado_e_removido": prev["leakage"],
            "avisos_preview": prev["warnings"], "n_treino": prev["n_train"], "n_teste": prev["n_test"],
            "distribuicao_classes": prev["class_distribution"], "testes_pareados_f1": pairs(cmp_cls),
        },
        "regressao": {
            "config_preprocessamento": pre_r, "avisos_preview": prev_r["warnings"],
            "vazamento_detectado": prev_r["leakage"], "testes_pareados_rmse": pairs(cmp_reg),
        },
        "ambiente": cls[0]["environment"],
    }
    (out / "resumo_execucao.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "experimentos_completos.json").write_text(
        json.dumps([_summary(e) for e in cls + reg], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultados gravados em {out.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "resultados")

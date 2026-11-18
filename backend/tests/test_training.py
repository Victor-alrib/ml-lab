"""Testes de treino, métricas e — principalmente — ausência de vazamento."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score, r2_score, recall_score, root_mean_squared_error
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.services.models import MODELS
from app.services.preprocessing import PreprocessConfig, build_preprocessor, prepare
from app.services.samples import motor_industrial_sintetico
from app.services.training import MODELS_DIR


def _cancer_df():
    b = load_breast_cancer(as_frame=True)
    df = b.frame.rename(columns={"target": "diagnostico"})
    df["diagnostico"] = df["diagnostico"].map({0: "maligno", 1: "benigno"})
    return df


def test_scaler_learned_only_on_train(client, cancer_id):
    """A média do StandardScaler deve ser a média do TREINO, não do dataset completo."""
    cfg = {"target": "diagnostico", "scaling": "standard", "test_size": 0.25, "random_state": 0}
    r = client.post("/api/preprocess/preview", json={"dataset_id": cancer_id, "preprocessing": cfg}).json()
    df = _cancer_df()
    # positivo = classe minoritária ("maligno")
    y = (df["diagnostico"] == "maligno").astype(int).to_numpy()
    X_tr, X_te, _, _ = train_test_split(df.drop(columns="diagnostico"), y, test_size=0.25,
                                        random_state=0, stratify=y)
    learned = {p["column"]: p for p in r["learned_params"]}
    col = "mean radius"
    assert learned[col]["center"] == pytest.approx(X_tr[col].mean(), rel=1e-12)
    assert learned[col]["center"] != pytest.approx(df[col].mean(), rel=1e-6)
    assert r["n_train"] == len(X_tr) and r["n_test"] == len(X_te)


def test_cv_refits_preprocessing_inside_each_fold():
    """Em cada fold, o scaler precisa ser ajustado só com os dados de treino daquele fold."""
    df = motor_industrial_sintetico()
    cfg = PreprocessConfig(target="temperatura_rolamento_c", scaling="standard", split_strategy="random")
    prep = prepare(df, cfg)
    pipe = Pipeline([("prep", build_preprocessor(cfg, prep.num_cols, prep.cat_cols)),
                     ("model", MODELS["linear_regression"].build("regression", {}, 0))])
    from sklearn.model_selection import KFold
    res = cross_validate(pipe, prep.X_train, prep.y_train, cv=KFold(4, shuffle=True, random_state=0),
                         return_estimator=True, return_indices=True)
    c = "carga_pct"  # coluna sem ausentes (a imputação não altera a média)
    j = prep.num_cols.index(c)
    for est, tr_idx in zip(res["estimator"], res["indices"]["train"]):
        scaler = est.named_steps["prep"].named_transformers_["num"].named_steps["scaler"]
        assert isinstance(scaler, StandardScaler)
        assert scaler.mean_[j] == pytest.approx(prep.X_train.iloc[tr_idx][c].mean(), rel=1e-12)


def test_reported_metrics_match_independent_recomputation(client, cancer_id):
    body = {"dataset_id": cancer_id, "model": "random_forest", "params": {"n_estimators": 50},
            "preprocessing": {"target": "diagnostico", "test_size": 0.3, "random_state": 7}}
    exp = client.post("/api/train", json=body).json()
    saved = joblib.load(MODELS_DIR / f"{exp['id']}.joblib")
    pipe = saved["pipeline"]
    df = _cancer_df()
    y = (df["diagnostico"] == "maligno").astype(int).to_numpy()
    _, X_te, _, y_te = train_test_split(df.drop(columns="diagnostico"), y, test_size=0.3,
                                        random_state=7, stratify=y)
    pred = pipe.predict(X_te[saved["meta"]["features"]])
    assert exp["metrics"]["test"]["f1"] == pytest.approx(f1_score(y_te, pred))
    assert exp["metrics"]["test"]["recall"] == pytest.approx(recall_score(y_te, pred))
    cm = np.array(exp["results"]["confusion_matrix"])
    assert cm.sum() == len(y_te)
    assert exp["results"]["positive_class"] == "maligno"


def test_cv_folds_and_baseline(client, cancer_id):
    body = {"dataset_id": cancer_id, "model": "logistic_regression",
            "preprocessing": {"target": "diagnostico"}, "cv": {"enabled": True, "folds": 5}}
    exp = client.post("/api/train", json=body).json()
    assert exp["cv"]["splitter"] == "StratifiedKFold" and exp["cv"]["folds"] == 5
    assert len(exp["cv"]["metrics"]["f1"]["folds"]) == 5
    # baseline "classe mais frequente" nunca prevê a classe positiva minoritária
    assert exp["metrics"]["baseline"]["recall"] == 0.0
    assert "oof" in exp["results"]["threshold_data"]
    oof = exp["results"]["threshold_data"]["oof"]
    assert len(oof["y"]) == exp["n_train"]  # StratifiedKFold cobre todo o treino


@pytest.mark.parametrize("model", ["linear_regression", "decision_tree", "random_forest", "xgboost"])
def test_regression_models(client, diabetes_id, model):
    body = {"dataset_id": diabetes_id, "model": model, "params": {"n_estimators": 50} if model in
            ("random_forest", "xgboost") else {}, "preprocessing": {"target": "progressao", "random_state": 1}}
    r = client.post("/api/train", json=body)
    assert r.status_code == 200, r.text
    exp = r.json()
    saved = joblib.load(MODELS_DIR / f"{exp['id']}.joblib")
    from sklearn.datasets import load_diabetes
    d = load_diabetes(as_frame=True).frame.rename(columns={"target": "progressao"})
    _, X_te, _, y_te = train_test_split(d.drop(columns="progressao"), d["progressao"].to_numpy(),
                                        test_size=0.2, random_state=1)
    pred = saved["pipeline"].predict(X_te)
    assert exp["metrics"]["test"]["rmse"] == pytest.approx(root_mean_squared_error(y_te, pred))
    assert exp["metrics"]["test"]["r2"] == pytest.approx(r2_score(y_te, pred))


def test_task_model_mismatch_rejected(client, diabetes_id):
    r = client.post("/api/train", json={"dataset_id": diabetes_id, "model": "logistic_regression",
                                        "preprocessing": {"target": "progressao"}})
    assert r.status_code == 422


def test_leakage_detector_flags_target_derived_column(client, motor_id):
    r = client.post("/api/preprocess/preview", json={"dataset_id": motor_id,
                                                     "preprocessing": {"target": "falha_24h"}}).json()
    assert any("modo_falha" in w and "vazamento" in w for w in r["warnings"])
    assert not any("vibracao" in w and "vazamento" in w for w in r["warnings"])


def test_temporal_split_puts_latest_rows_in_test():
    df = motor_industrial_sintetico().sample(frac=1, random_state=3)  # embaralha
    cfg = PreprocessConfig(target="temperatura_rolamento_c", split_strategy="temporal",
                           time_column="timestamp", test_size=0.2)
    prep = prepare(df, cfg)
    ts = pd.to_datetime(df["timestamp"]).sort_values()
    cutoff = ts.iloc[len(ts) - len(prep.X_test)]
    assert len(prep.X_test) == round(len(df) * 0.2)
    # o alvo do teste corresponde às linhas mais recentes
    latest = df.assign(t=pd.to_datetime(df["timestamp"])).sort_values("t", kind="stable")
    np.testing.assert_allclose(prep.y_test, latest["temperatura_rolamento_c"].to_numpy()[-len(prep.X_test):])
    assert "timestamp" not in prep.features and cutoff is not None


def test_xgboost_auto_scale_pos_weight_uses_train_counts(client, motor_id):
    feats = ["carga_pct", "corrente_a", "vibracao_rms_mm_s", "horas_desde_manutencao"]
    exp = client.post("/api/train", json={
        "dataset_id": motor_id, "model": "xgboost", "params": {"scale_pos_weight": "auto", "n_estimators": 30},
        "preprocessing": {"target": "falha_24h", "features": feats}, "cv": {"enabled": False}}).json()
    pipe = joblib.load(MODELS_DIR / f"{exp['id']}.joblib")["pipeline"]
    spw = pipe.named_steps["model"].get_params()["scale_pos_weight"]
    assert spw == pytest.approx(exp["params"]["scale_pos_weight_resolved"])
    assert spw > 5  # ~8% de positivos -> ~11


def test_paired_comparison(client, cancer_id):
    ids = []
    for model in ("logistic_regression", "decision_tree"):
        ids.append(client.post("/api/train", json={
            "dataset_id": cancer_id, "model": model, "preprocessing": {"target": "diagnostico"}}).json()["id"])
    cmp = client.post("/api/experiments/compare", json={"ids": ids, "metric": "f1"}).json()
    pair = cmp["pairs"][0]
    assert pair["available"] and pair["k"] == 5
    assert 0 <= pair["p_corrected"] <= 1
    # a correção de Nadeau-Bengio é sempre mais conservadora que o t-test comum
    assert pair["p_corrected"] >= pair["p_standard"] - 1e-12


def test_experiment_crud(client, cancer_id):
    exp = client.post("/api/train", json={"dataset_id": cancer_id, "model": "decision_tree",
                                          "preprocessing": {"target": "diagnostico"},
                                          "cv": {"enabled": False}}).json()
    assert client.patch(f"/api/experiments/{exp['id']}", json={"name": "árvore teste"}).json()["name"] == "árvore teste"
    assert any(e["id"] == exp["id"] for e in client.get("/api/experiments").json())
    assert client.get(f"/api/experiments/{exp['id']}/model").status_code == 200
    assert client.delete(f"/api/experiments/{exp['id']}").status_code == 200
    assert client.get(f"/api/experiments/{exp['id']}").status_code == 404


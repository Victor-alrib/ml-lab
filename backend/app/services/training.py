"""Treinamento e avaliação estatística.

Fluxo de um experimento (sem vazamento):

1. `prepare()` separa treino/teste ANTES de qualquer ajuste.
2. Um Pipeline [pré-processamento -> seleção -> modelo] é montado.
3. Validação cruzada roda `cross_validate(pipeline, X_train, y_train)`: em cada
   fold, o pipeline inteiro é reajustado só nos dados de treino do fold.
4. O pipeline é ajustado no treino completo e avaliado UMA vez no teste.
5. Um baseline ingênuo (DummyClassifier/DummyRegressor) é avaliado nas mesmas
   condições, como referência mínima.
"""
from __future__ import annotations

import platform
import time
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from pydantic import BaseModel, Field
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_validate, learning_curve
from sklearn.pipeline import Pipeline

from ..core.config import DATA_DIR
from .models import CLASSIFICATION, REGRESSION, build_baseline, default_params, get_model
from .preprocessing import PreprocessConfig, build_preprocessor, build_selector, prepare

MODELS_DIR = DATA_DIR / "models"
MAX_PLOT_POINTS = 2000
MAX_THRESHOLD_POINTS = 20000


class CVConfig(BaseModel):
    enabled: bool = True
    folds: int = Field(5, ge=2, le=20)


class TrainRequest(BaseModel):
    dataset_id: str
    name: str | None = None
    model: str
    params: dict[str, Any] = {}
    preprocessing: PreprocessConfig
    cv: CVConfig = CVConfig()
    average: str = Field("macro", description="macro | weighted (multiclasse)")
    learning_curve: bool = False
    permutation_importance: bool = False
    save_model: bool = True
    tags: list[str] = []
    notes: str | None = None


# --------------------------------------------------------------------------- #
# Métricas                                                                    #
# --------------------------------------------------------------------------- #
def _cls_kwargs(n_classes: int, average: str) -> dict:
    return {"average": "binary", "pos_label": 1} if n_classes == 2 else {"average": average}


def classification_metrics(y_true, y_pred, proba, n_classes: int, average: str) -> dict:
    kw = _cls_kwargs(n_classes, average)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0, **kw),
        "recall": recall_score(y_true, y_pred, zero_division=0, **kw),
        "f1": f1_score(y_true, y_pred, zero_division=0, **kw),
    }
    if proba is not None:
        labels = list(range(n_classes))
        try:
            out["log_loss"] = log_loss(y_true, proba, labels=labels)
        except ValueError:
            pass
        try:
            if n_classes == 2:
                out["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
            elif len(np.unique(y_true)) == n_classes:
                out["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr", average=average, labels=labels)
        except ValueError:
            pass
    return {k: float(v) for k, v in out.items()}


def regression_metrics(y_true, y_pred) -> dict:
    out = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    if np.all(np.abs(y_true) > 1e-12):
        out["mape"] = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {k: float(v) for k, v in out.items()}


def cv_scorers(task: str, n_classes: int, average: str) -> dict:
    if task == REGRESSION:
        return {
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "rmse": make_scorer(root_mean_squared_error, greater_is_better=False),
            "r2": make_scorer(r2_score),
        }
    kw = _cls_kwargs(n_classes, average)
    sc = {
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0, **kw),
        "recall": make_scorer(recall_score, zero_division=0, **kw),
        "f1": make_scorer(f1_score, zero_division=0, **kw),
    }
    if n_classes == 2:
        sc["roc_auc"] = "roc_auc"
    return sc


PRIMARY_METRIC = {CLASSIFICATION: "f1", REGRESSION: "rmse"}
LOWER_IS_BETTER = {"mae", "rmse", "mse", "mape", "log_loss"}


def _cv_splitter(split_strategy: str, task: str, folds: int, seed: int, y: np.ndarray):
    if split_strategy == "temporal":
        return TimeSeriesSplit(n_splits=folds), "TimeSeriesSplit"
    if task == CLASSIFICATION:
        min_count = int(np.bincount(y).min())
        if min_count >= folds:
            return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed), "StratifiedKFold"
        if min_count >= 2:
            return StratifiedKFold(n_splits=min_count, shuffle=True, random_state=seed), "StratifiedKFold"
    return KFold(n_splits=folds, shuffle=True, random_state=seed), "KFold"


def _summarize_cv(res: dict, scorers: dict) -> dict:
    out = {}
    for m in scorers:
        sign = -1.0 if m in LOWER_IS_BETTER else 1.0
        val = sign * np.asarray(res[f"test_{m}"], dtype=float)
        tr = sign * np.asarray(res[f"train_{m}"], dtype=float)
        out[m] = {"mean": float(np.nanmean(val)), "std": float(np.nanstd(val, ddof=1)) if len(val) > 1 else 0.0,
                  "folds": [None if np.isnan(v) else float(v) for v in val],
                  "train_mean": float(np.nanmean(tr)),
                  "train_std": float(np.nanstd(tr, ddof=1)) if len(tr) > 1 else 0.0}
    return out


# --------------------------------------------------------------------------- #
# Importância de atributos                                                    #
# --------------------------------------------------------------------------- #
def _model_importance(model, names: list[str]) -> dict | None:
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
        kind = "Importância por impureza/ganho (do próprio modelo)"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        vals = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        kind = "|Coeficiente| (atributos na escala transformada)"
    else:
        return None
    if len(vals) != len(names):
        return None
    order = np.argsort(-vals)[:30]
    return {"kind": kind, "features": [names[i] for i in order], "values": [float(vals[i]) for i in order]}


def _coefficients(model, names: list[str], class_labels: list[str] | None) -> dict | None:
    if not hasattr(model, "coef_"):
        return None
    coef = np.asarray(model.coef_, dtype=float)
    intercept = np.atleast_1d(getattr(model, "intercept_", 0.0)).astype(float)
    if coef.ndim == 1 or coef.shape[0] == 1:
        c = coef.ravel()
        if len(c) != len(names):
            return None
        return {"rows": [{"label": "coef", "values": c.tolist()}], "features": names,
                "intercept": intercept.tolist()}
    return {"rows": [{"label": class_labels[i] if class_labels else str(i), "values": coef[i].tolist()}
                     for i in range(coef.shape[0])], "features": names, "intercept": intercept.tolist()}


def _sample_idx(n: int, k: int = MAX_PLOT_POINTS) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(0).choice(n, k, replace=False))


# --------------------------------------------------------------------------- #
# Execução                                                                    #
# --------------------------------------------------------------------------- #
def run_experiment(df: pd.DataFrame, req: TrainRequest, dataset_meta: dict) -> dict:
    t_start = time.perf_counter()
    cfg = req.preprocessing
    seed = cfg.random_state
    prep = prepare(df, cfg)
    task = prep.task
    spec = get_model(req.model)
    if task not in spec.tasks:
        raise ValueError(f"{spec.name} não suporta {'classificação' if task == CLASSIFICATION else 'regressão'}.")

    params = {**default_params(req.model, task), **(req.params or {})}
    n_classes = len(prep.class_labels) if task == CLASSIFICATION else 0
    caught: list[str] = []

    if req.model == "xgboost" and task == CLASSIFICATION and str(params.get("scale_pos_weight")) == "auto":
        if n_classes == 2:
            pos = int(prep.y_train.sum())
            params["scale_pos_weight_resolved"] = float((len(prep.y_train) - pos) / max(pos, 1))
            build_params = {**params, "scale_pos_weight": params["scale_pos_weight_resolved"]}
        else:
            caught.append("scale_pos_weight só se aplica a classificação binária; ignorado.")
            build_params = {**params, "scale_pos_weight": 1.0}
    else:
        build_params = params

    def make_pipeline(estimator) -> Pipeline:
        return Pipeline([
            ("prep", build_preprocessor(cfg, prep.num_cols, prep.cat_cols)),
            ("select", build_selector(cfg, task)),
            ("model", estimator),
        ])

    pipe = make_pipeline(spec.build(task, build_params, seed))
    X_train, X_test, y_train, y_test = prep.X_train, prep.X_test, prep.y_train, prep.y_test

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")

        # 1) Validação cruzada no TREINO
        cv_result = None
        cv_time = 0.0
        oof_proba = None
        if req.cv.enabled:
            splitter, splitter_name = _cv_splitter(prep.split_strategy, task, req.cv.folds, seed, y_train)
            scorers = cv_scorers(task, n_classes, req.average)
            t0 = time.perf_counter()
            res = cross_validate(make_pipeline(spec.build(task, build_params, seed)), X_train, y_train,
                                 cv=splitter, scoring=scorers, return_train_score=True,
                                 error_score=np.nan, n_jobs=1, return_estimator=True, return_indices=True)
            cv_time = time.perf_counter() - t0
            # Probabilidades "out-of-fold": cada amostra do treino é prevista pelo modelo
            # que NÃO a viu. Servem para escolher o limiar de decisão sem tocar no teste.
            if task == CLASSIFICATION and n_classes == 2:
                oof = np.full(len(y_train), np.nan)
                try:
                    for est, val_idx in zip(res["estimator"], res["indices"]["test"]):
                        oof[val_idx] = est.predict_proba(X_train.iloc[val_idx])[:, 1]
                    oof_proba = oof
                except (AttributeError, ValueError):
                    oof_proba = None
            cv_result = {
                "splitter": splitter_name, "folds": int(splitter.get_n_splits()),
                "metrics": _summarize_cv(res, scorers),
                "fit_time_mean": float(np.mean(res["fit_time"])),
            }

        # 2) Ajuste final no treino completo
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_test = pipe.predict(X_test)
        predict_time = time.perf_counter() - t0
        pred_train = pipe.predict(X_train)

        proba_test = proba_train = None
        if task == CLASSIFICATION and hasattr(pipe, "predict_proba"):
            try:
                proba_test = pipe.predict_proba(X_test)
                proba_train = pipe.predict_proba(X_train)
            except (AttributeError, NotImplementedError):
                pass

        # 3) Baseline ingênuo
        base = make_pipeline(build_baseline(task, seed)).fit(X_train, y_train)
        base_pred = base.predict(X_test)

        # 4) Curva de aprendizado (opcional)
        lc = None
        if req.learning_curve:
            splitter, _ = _cv_splitter(prep.split_strategy, task, min(req.cv.folds, 5), seed, y_train)
            primary = PRIMARY_METRIC[task]
            scorer = cv_scorers(task, n_classes, req.average)[primary]
            sizes, tr_sc, va_sc = learning_curve(
                make_pipeline(spec.build(task, build_params, seed)), X_train, y_train, cv=splitter,
                scoring=scorer, train_sizes=np.linspace(0.15, 1.0, 6), error_score=np.nan, n_jobs=1,
                shuffle=prep.split_strategy != "temporal", random_state=seed)
            sign = -1.0 if primary in LOWER_IS_BETTER else 1.0
            lc = {"metric": primary, "train_sizes": sizes.tolist(),
                  "train_mean": (sign * np.nanmean(tr_sc, axis=1)).tolist(),
                  "train_std": np.nanstd(tr_sc, axis=1).tolist(),
                  "val_mean": (sign * np.nanmean(va_sc, axis=1)).tolist(),
                  "val_std": np.nanstd(va_sc, axis=1).tolist()}

        # 5) Importância por permutação no TESTE (opcional)
        perm = None
        if req.permutation_importance:
            scoring = "f1" if task == CLASSIFICATION and n_classes == 2 else (
                "f1_macro" if task == CLASSIFICATION else "neg_root_mean_squared_error")
            pi = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=seed,
                                        scoring=scoring, n_jobs=1)
            order = np.argsort(-pi.importances_mean)[:30]
            perm = {"kind": f"Permutação no teste ({scoring}), atributos originais",
                    "features": [prep.features[i] for i in order],
                    "values": [float(pi.importances_mean[i]) for i in order],
                    "std": [float(pi.importances_std[i]) for i in order]}

    for w in wlist:
        msg = str(w.message)
        if "feature_selection" in str(w.filename) and issubclass(w.category, RuntimeWarning):
            continue  # divisão por zero em F-test com atributo constante dentro de um fold
        if any(s in msg for s in ("is greater than n_features", "Features [", "Found unknown categories")):
            if "greater than n_features" in msg:
                caught.append("k da seleção de atributos é maior que o número de atributos; todos foram mantidos.")
            continue
        short = f"{w.category.__name__}: {msg[:220]}"
        if short not in caught:
            caught.append(short)
    caught = caught[:8]

    final_features = list(pipe[:-1].get_feature_names_out())
    model = pipe.named_steps["model"]

    result: dict[str, Any] = {
        "task": task, "class_labels": prep.class_labels, "positive_class": prep.positive_class,
        "n_classes": n_classes,
    }
    if task == CLASSIFICATION:
        m_test = classification_metrics(y_test, pred_test, proba_test, n_classes, req.average)
        m_train = classification_metrics(y_train, pred_train, proba_train, n_classes, req.average)
        m_base = classification_metrics(y_test, base_pred, None, n_classes, req.average)
        cm = confusion_matrix(y_test, pred_test, labels=list(range(n_classes)))
        per_class = []
        p, r, f, s = (precision_score(y_test, pred_test, average=None, labels=list(range(n_classes)), zero_division=0),
                      recall_score(y_test, pred_test, average=None, labels=list(range(n_classes)), zero_division=0),
                      f1_score(y_test, pred_test, average=None, labels=list(range(n_classes)), zero_division=0),
                      np.bincount(y_test, minlength=n_classes))
        for i, lab in enumerate(prep.class_labels):
            per_class.append({"label": lab, "precision": float(p[i]), "recall": float(r[i]),
                              "f1": float(f[i]), "support": int(s[i])})
        result.update(confusion_matrix=cm.tolist(), per_class=per_class)
        if proba_test is not None and n_classes == 2 and len(np.unique(y_test)) == 2:
            fpr, tpr, thr = roc_curve(y_test, proba_test[:, 1])
            prec, rec, _ = precision_recall_curve(y_test, proba_test[:, 1])
            idx = _sample_idx(len(fpr), 400)
            idx2 = _sample_idx(len(prec), 400)
            result["roc_curve"] = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist(), "auc": float(auc(fpr, tpr))}
            result["pr_curve"] = {"precision": prec[idx2].tolist(), "recall": rec[idx2].tolist(),
                                  "prevalence": float(np.mean(y_test))}
            # Distribuição das probabilidades previstas por classe real
            result["proba_hist"] = {
                "negative": proba_test[y_test == 0, 1].tolist()[:MAX_PLOT_POINTS],
                "positive": proba_test[y_test == 1, 1].tolist()[:MAX_PLOT_POINTS]}
            # Dados para o explorador de limiar (calculado no frontend com os valores reais)
            ti = _sample_idx(len(y_test), MAX_THRESHOLD_POINTS)
            result["threshold_data"] = {"test": {"y": y_test[ti].tolist(), "p": proba_test[ti, 1].tolist(),
                                                 "sampled": bool(len(y_test) > MAX_THRESHOLD_POINTS)}}
            if oof_proba is not None:
                ok = ~np.isnan(oof_proba)
                yo, po = y_train[ok], oof_proba[ok]
                oi = _sample_idx(len(yo), MAX_THRESHOLD_POINTS)
                result["threshold_data"]["oof"] = {"y": yo[oi].tolist(), "p": po[oi].tolist(),
                                                   "sampled": bool(len(yo) > MAX_THRESHOLD_POINTS)}
    else:
        m_test = regression_metrics(y_test, pred_test)
        m_train = regression_metrics(y_train, pred_train)
        m_base = regression_metrics(y_test, base_pred)
        idx = _sample_idx(len(y_test))
        resid = y_test - pred_test
        result.update(
            predictions={"y_true": y_test[idx].tolist(), "y_pred": pred_test[idx].tolist(),
                         "sampled": bool(len(y_test) > MAX_PLOT_POINTS), "n": int(len(y_test))},
            residuals={"values": resid[idx].tolist(), "mean": float(resid.mean()), "std": float(resid.std(ddof=1))},
        )
        if prep.split_strategy == "temporal":
            result["temporal_test"] = {"y_true": y_test[-MAX_PLOT_POINTS:].tolist(),
                                       "y_pred": pred_test[-MAX_PLOT_POINTS:].tolist()}

    total_time = time.perf_counter() - t_start
    exp_record: dict[str, Any] = {
        "name": req.name or f"{spec.name} · {dataset_meta['name']}",
        "dataset_id": req.dataset_id, "dataset_name": dataset_meta["name"],
        "task": task, "model": req.model, "model_name": spec.name, "family": spec.family,
        "params": params, "preprocessing": cfg.model_dump(), "cv_config": req.cv.model_dump(),
        "average": req.average if task == CLASSIFICATION and n_classes > 2 else (
            "binary" if task == CLASSIFICATION else None),
        "target": cfg.target,
        "features_used": prep.features, "final_features": final_features,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "split_strategy": prep.split_strategy,
        "metrics": {"test": m_test, "train": m_train, "baseline": m_base},
        "primary_metric": PRIMARY_METRIC[task],
        "cv": cv_result,
        "learning_curve": lc,
        "importance": _model_importance(model, final_features),
        "permutation_importance": perm,
        "coefficients": _coefficients(model, final_features, prep.class_labels),
        "timing": {"fit_s": fit_time, "predict_s": predict_time, "cv_s": cv_time, "total_s": total_time},
        "results": result,
        "notes_preprocessing": prep.notes, "warnings": prep.warnings + caught,
        "tags": req.tags, "notes": req.notes,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    if req.save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        exp_record["_pipeline"] = pipe  # removido antes de salvar o JSON
    return exp_record


def persist_model(exp_id: str, pipe: Pipeline, meta: dict) -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{exp_id}.joblib"
    joblib.dump({"pipeline": pipe, "meta": meta}, path)
    return str(path.name)

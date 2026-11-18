"""Catálogo de modelos e esquema de hiperparâmetros.

O esquema é servido ao frontend, que gera o formulário automaticamente. Para
adicionar um modelo novo, basta registrar um `ModelSpec` aqui.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

CLASSIFICATION = "classification"
REGRESSION = "regression"

_LOGREG_NEW_API = inspect.signature(LogisticRegression).parameters["penalty"].default == "deprecated"


def P(name: str, label: str, type_: str, default: Any, help_: str, **kw: Any) -> dict:
    """Atalho para declarar um hiperparâmetro."""
    return {"name": name, "label": label, "type": type_, "default": default, "help": help_, **kw}


@dataclass
class ModelSpec:
    key: str
    name: str
    tasks: list[str]
    description: str
    params: dict[str, list[dict]]  # por tarefa
    build: Callable[[str, dict, int], Any]
    family: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"key": self.key, "name": self.name, "tasks": self.tasks, "family": self.family,
                "description": self.description, "params": self.params, "notes": self.notes}


def _opt_int(v: Any) -> int | None:
    if v in (None, "", "None", 0, "0"):
        return None
    return int(v)


def _class_weight(v: Any) -> str | None:
    return "balanced" if v in ("balanced", True) else None


# --------------------------------------------------------------------------- #
# Construtores                                                                #
# --------------------------------------------------------------------------- #
def _build_linear(task: str, p: dict, seed: int):
    reg = p.get("regularization", "none")
    fit_intercept = bool(p.get("fit_intercept", True))
    alpha = float(p.get("alpha", 1.0))
    if reg == "ridge":
        return Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=seed)
    if reg == "lasso":
        return Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=10000, random_state=seed)
    return LinearRegression(fit_intercept=fit_intercept)


def _build_logistic(task: str, p: dict, seed: int):
    penalty = p.get("penalty", "l2")
    C = float(p.get("C", 1.0))
    kw: dict = {
        "max_iter": int(p.get("max_iter", 1000)),
        "class_weight": _class_weight(p.get("class_weight")),
        "fit_intercept": bool(p.get("fit_intercept", True)),
        "random_state": seed,
    }
    if _LOGREG_NEW_API:
        if penalty == "none":
            kw.update(C=np.inf, solver="lbfgs")
        elif penalty == "l1":
            kw.update(C=C, l1_ratio=1.0, solver="saga")
        else:
            kw.update(C=C, l1_ratio=0.0, solver="lbfgs")
    else:  # scikit-learn < 1.8
        if penalty == "none":
            kw.update(penalty=None, solver="lbfgs")
        elif penalty == "l1":
            kw.update(penalty="l1", C=C, solver="saga")
        else:
            kw.update(penalty="l2", C=C, solver="lbfgs")
    return LogisticRegression(**kw)


def _build_tree(task: str, p: dict, seed: int):
    common = dict(
        max_depth=_opt_int(p.get("max_depth")),
        min_samples_split=int(p.get("min_samples_split", 2)),
        min_samples_leaf=int(p.get("min_samples_leaf", 1)),
        random_state=seed,
    )
    if task == CLASSIFICATION:
        return DecisionTreeClassifier(criterion=p.get("criterion", "gini"),
                                      class_weight=_class_weight(p.get("class_weight")), **common)
    return DecisionTreeRegressor(criterion=p.get("criterion", "squared_error"), **common)


def _max_features(v: Any):
    if v in ("sqrt", "log2"):
        return v
    if v in (None, "all", "1.0"):
        return 1.0
    return float(v)


def _build_rf(task: str, p: dict, seed: int):
    common = dict(
        n_estimators=int(p.get("n_estimators", 200)),
        max_depth=_opt_int(p.get("max_depth")),
        min_samples_split=int(p.get("min_samples_split", 2)),
        min_samples_leaf=int(p.get("min_samples_leaf", 1)),
        max_features=_max_features(p.get("max_features", "sqrt" if task == CLASSIFICATION else "1.0")),
        bootstrap=bool(p.get("bootstrap", True)),
        n_jobs=-1,
        random_state=seed,
    )
    if task == CLASSIFICATION:
        return RandomForestClassifier(class_weight=_class_weight(p.get("class_weight")), **common)
    return RandomForestRegressor(**common)


def _build_xgb(task: str, p: dict, seed: int):
    common = dict(
        n_estimators=int(p.get("n_estimators", 300)),
        learning_rate=float(p.get("learning_rate", 0.1)),
        max_depth=int(p.get("max_depth", 6)),
        subsample=float(p.get("subsample", 1.0)),
        colsample_bytree=float(p.get("colsample_bytree", 1.0)),
        min_child_weight=float(p.get("min_child_weight", 1.0)),
        reg_lambda=float(p.get("reg_lambda", 1.0)),
        reg_alpha=float(p.get("reg_alpha", 0.0)),
        gamma=float(p.get("gamma", 0.0)),
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
    )
    if task == CLASSIFICATION:
        spw = p.get("scale_pos_weight", 1.0)
        # "auto" é resolvido em training.py a partir do conjunto de TREINO
        if spw not in ("auto", None, ""):
            common["scale_pos_weight"] = float(spw)
        return XGBClassifier(eval_metric="logloss", **common)
    return XGBRegressor(**common)


def build_baseline(task: str, seed: int):
    if task == CLASSIFICATION:
        return DummyClassifier(strategy="most_frequent")
    return DummyRegressor(strategy="mean")


# --------------------------------------------------------------------------- #
# Hiperparâmetros                                                             #
# --------------------------------------------------------------------------- #
_CW = P("class_weight", "Peso das classes", "select", "none",
        "'balanced' dá mais peso às classes raras (útil em dados de falha, que costumam ser desbalanceados).",
        options=[{"value": "none", "label": "Nenhum"}, {"value": "balanced", "label": "Balanceado"}])

_TREE_COMMON = [
    P("max_depth", "Profundidade máxima", "int", None,
      "Limita quantos níveis a árvore pode ter. Vazio = sem limite (maior risco de overfitting).",
      min=1, max=100, nullable=True),
    P("min_samples_split", "Mín. amostras para dividir", "int", 2,
      "Número mínimo de amostras em um nó para que ele possa ser dividido.", min=2, max=500),
    P("min_samples_leaf", "Mín. amostras por folha", "int", 1,
      "Cada folha precisa ter pelo menos esse número de amostras. Valores maiores suavizam o modelo.",
      min=1, max=500),
]

MODELS: dict[str, ModelSpec] = {
    "linear_regression": ModelSpec(
        key="linear_regression", name="Regressão Linear", family="Linear", tasks=[REGRESSION],
        description="Ajusta uma reta (hiperplano) que minimiza o erro quadrático. Rápida, interpretável "
                    "e um ótimo baseline para problemas de regressão.",
        params={REGRESSION: [
            P("regularization", "Regularização", "select", "none",
              "Ridge (L2) encolhe coeficientes; Lasso (L1) pode zerá-los (seleção de atributos).",
              options=[{"value": "none", "label": "Nenhuma (OLS)"}, {"value": "ridge", "label": "Ridge (L2)"},
                       {"value": "lasso", "label": "Lasso (L1)"}]),
            P("alpha", "Alpha (força da regularização)", "float", 1.0,
              "Só é usado com Ridge/Lasso. Quanto maior, mais simples o modelo.",
              min=0.0001, max=1000, step=0.1, log=True, depends_on={"regularization": ["ridge", "lasso"]}),
            P("fit_intercept", "Ajustar intercepto", "bool", True, "Inclui o termo constante (b) no modelo."),
        ]},
        build=_build_linear,
        notes=["Sensível à escala apenas quando há regularização — padronize os dados nesse caso."],
    ),
    "logistic_regression": ModelSpec(
        key="logistic_regression", name="Regressão Logística", family="Linear", tasks=[CLASSIFICATION],
        description="Modelo linear que estima a probabilidade de cada classe via função sigmoide/softmax. "
                    "Interpretável e bom baseline para classificação.",
        params={CLASSIFICATION: [
            P("C", "C (inverso da regularização)", "float", 1.0,
              "Valores pequenos = regularização forte (modelo mais simples).", min=0.0001, max=1000,
              step=0.1, log=True),
            P("penalty", "Penalidade", "select", "l2", "Tipo de regularização aplicada aos coeficientes.",
              options=[{"value": "l2", "label": "L2 (Ridge)"}, {"value": "l1", "label": "L1 (Lasso)"},
                       {"value": "none", "label": "Nenhuma"}]),
            P("max_iter", "Máx. iterações", "int", 1000, "Iterações do otimizador.", min=50, max=20000),
            P("fit_intercept", "Ajustar intercepto", "bool", True, "Inclui o termo constante."),
            _CW,
        ]},
        build=_build_logistic,
        notes=["Padronize os atributos: a regularização e o otimizador são sensíveis à escala."],
    ),
    "decision_tree": ModelSpec(
        key="decision_tree", name="Decision Tree", family="Árvores", tasks=[CLASSIFICATION, REGRESSION],
        description="Divide o espaço de atributos com perguntas do tipo 'x > limiar?'. Totalmente "
                    "interpretável, mas tende ao overfitting sem limitar a profundidade.",
        params={
            CLASSIFICATION: [
                P("criterion", "Critério", "select", "gini", "Medida de impureza usada para escolher as divisões.",
                  options=[{"value": "gini", "label": "Gini"}, {"value": "entropy", "label": "Entropia"},
                           {"value": "log_loss", "label": "Log loss"}]),
                *_TREE_COMMON, _CW],
            REGRESSION: [
                P("criterion", "Critério", "select", "squared_error", "Função de erro usada nas divisões.",
                  options=[{"value": "squared_error", "label": "Erro quadrático"},
                           {"value": "absolute_error", "label": "Erro absoluto"},
                           {"value": "friedman_mse", "label": "Friedman MSE"}]),
                *_TREE_COMMON],
        },
        build=_build_tree,
        notes=["Não precisa de normalização."],
    ),
    "random_forest": ModelSpec(
        key="random_forest", name="Random Forest", family="Ensemble (bagging)",
        tasks=[CLASSIFICATION, REGRESSION],
        description="Conjunto de árvores treinadas em amostras bootstrap e subconjuntos de atributos; "
                    "a média reduz a variância de uma árvore isolada.",
        params={t: [
            P("n_estimators", "Número de árvores", "int", 200, "Mais árvores = mais estável (e mais lento).",
              min=10, max=2000),
            *_TREE_COMMON,
            P("max_features", "Atributos por divisão", "select", "sqrt" if t == CLASSIFICATION else "1.0",
              "Quantos atributos sortear em cada divisão. Menos atributos = árvores mais diversas.",
              options=[{"value": "sqrt", "label": "√n"}, {"value": "log2", "label": "log2(n)"},
                       {"value": "0.5", "label": "50%"}, {"value": "1.0", "label": "Todos"}]),
            P("bootstrap", "Bootstrap", "bool", True, "Treina cada árvore em uma amostra com reposição."),
            *([_CW] if t == CLASSIFICATION else []),
        ] for t in (CLASSIFICATION, REGRESSION)},
        build=_build_rf,
        notes=["Não precisa de normalização."],
    ),
    "xgboost": ModelSpec(
        key="xgboost", name="XGBoost", family="Ensemble (boosting)", tasks=[CLASSIFICATION, REGRESSION],
        description="Gradient boosting: árvores rasas adicionadas em sequência, cada uma corrigindo o erro "
                    "das anteriores. Costuma liderar em dados tabulares.",
        params={t: [
            P("n_estimators", "Número de árvores", "int", 300, "Rodadas de boosting.", min=10, max=5000),
            P("learning_rate", "Learning rate (eta)", "float", 0.1,
              "Peso de cada árvore nova. Menor = mais robusto, mas exige mais árvores.",
              min=0.001, max=1.0, step=0.01, log=True),
            P("max_depth", "Profundidade máxima", "int", 6, "Profundidade de cada árvore.", min=1, max=20),
            P("subsample", "Subamostragem de linhas", "float", 1.0,
              "Fração das linhas usada por árvore (<1 reduz overfitting).", min=0.1, max=1.0, step=0.05),
            P("colsample_bytree", "Subamostragem de colunas", "float", 1.0,
              "Fração dos atributos usada por árvore.", min=0.1, max=1.0, step=0.05),
            P("min_child_weight", "Min child weight", "float", 1.0,
              "Soma mínima de pesos (hessiana) numa folha. Maior = modelo mais conservador.",
              min=0, max=100, step=0.5),
            P("reg_lambda", "Lambda (L2)", "float", 1.0, "Regularização L2 nos pesos das folhas.",
              min=0, max=100, step=0.1),
            P("reg_alpha", "Alpha (L1)", "float", 0.0, "Regularização L1 nos pesos das folhas.",
              min=0, max=100, step=0.1),
            P("gamma", "Gamma", "float", 0.0, "Redução mínima de perda para fazer uma nova divisão.",
              min=0, max=50, step=0.1),
            *([P("scale_pos_weight", "Scale pos weight", "select", "1",
                 "Peso da classe positiva (binária). 'Automático' usa nº negativos / nº positivos do treino.",
                 options=[{"value": "1", "label": "1 (sem ajuste)"}, {"value": "auto", "label": "Automático"}])]
              if t == CLASSIFICATION else []),
        ] for t in (CLASSIFICATION, REGRESSION)},
        build=_build_xgb,
        notes=["Não precisa de normalização. Lida com valores ausentes nativamente, mas aqui a imputação "
               "do pré-processamento é aplicada antes."],
    ),
}


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise ValueError(f"Modelo desconhecido: {key}")
    return MODELS[key]


def default_params(key: str, task: str) -> dict:
    return {p["name"]: p["default"] for p in get_model(key).params.get(task, [])}


def catalog() -> list[dict]:
    return [m.to_dict() for m in MODELS.values()]

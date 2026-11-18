"""Pré-processamento SEM vazamento de dados.

Princípio: toda transformação que *aprende* algo dos dados (média para imputar,
média/desvio para padronizar, categorias do one-hot, limites de outlier,
seleção de atributos) fica dentro de um `sklearn.Pipeline`. O pipeline é
ajustado apenas no conjunto de treino — e, na validação cruzada, apenas nos
folds de treino de cada iteração. O conjunto de teste nunca é visto durante o
ajuste.

Operações que não aprendem estatísticas (remover linhas sem alvo, escolher
colunas, ordenar por tempo) podem ser feitas antes da separação sem vazamento.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from .models import CLASSIFICATION, REGRESSION
from .profiling import BINARY, CATEGORICAL, CONSTANT, DATETIME, TEXT, column_types, is_numeric_type


# --------------------------------------------------------------------------- #
# Configuração                                                                #
# --------------------------------------------------------------------------- #
class FeatureSelectionConfig(BaseModel):
    method: str = Field("none", description="none | kbest_f | kbest_mi | variance")
    k: int = 10
    variance_threshold: float = 0.0


class PreprocessConfig(BaseModel):
    target: str
    task: str = Field("auto", description="auto | classification | regression")
    features: list[str] | None = None
    numeric_imputation: str = Field("median", description="mean | median | most_frequent | constant | drop_rows")
    numeric_fill_value: float = 0.0
    categorical_imputation: str = Field("most_frequent", description="most_frequent | constant | drop_rows")
    add_missing_indicator: bool = False
    outlier_treatment: str = Field("none", description="none | clip_iqr")
    scaling: str = Field("standard", description="none | standard | minmax | robust")
    encoding: str = Field("onehot", description="onehot | ordinal")
    max_categories: int = 20
    feature_selection: FeatureSelectionConfig = FeatureSelectionConfig()
    split_strategy: str = Field("stratified", description="random | stratified | temporal")
    time_column: str | None = None
    test_size: float = Field(0.2, gt=0.05, lt=0.6)
    random_state: int = 42
    positive_class: str | None = None


class PreprocessError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Transformador customizado: corte de outliers por IQR aprendido no treino    #
# --------------------------------------------------------------------------- #
class IQRClipper(BaseEstimator, TransformerMixin):
    """Limita (winsoriza) valores fora de [Q1 - k·IQR, Q3 + k·IQR].

    Os quartis são aprendidos em `fit` (somente treino) e reaplicados em
    `transform` (validação/teste), evitando vazamento.
    """

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            q1 = np.nanpercentile(X, 25, axis=0)
            q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


# --------------------------------------------------------------------------- #
# Preparação                                                                  #
# --------------------------------------------------------------------------- #
def infer_task(y: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(y):
        return CLASSIFICATION
    n_unique = y.nunique()
    is_int = bool(np.all(np.mod(y.dropna().to_numpy(dtype=float), 1) == 0))
    return CLASSIFICATION if (is_int and n_unique <= 20) or n_unique == 2 else REGRESSION


def default_features(df: pd.DataFrame, target: str, time_column: str | None = None) -> list[str]:
    types = column_types(df)
    n = len(df)
    out = []
    for c in df.columns:
        if c in (target, time_column):
            continue
        t = types[c]
        if t in (CONSTANT, TEXT, DATETIME):
            continue
        if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() == n and n > 20:
            continue  # identificador
        out.append(c)
    return out


def split_feature_kinds(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Retorna (numéricas, categóricas, ignoradas)."""
    types = column_types(df[features])
    num, cat, ignored = [], [], []
    for c in features:
        t = types[c]
        if is_numeric_type(t) or (t == BINARY and pd.api.types.is_numeric_dtype(df[c])):
            num.append(c)
        elif pd.api.types.is_bool_dtype(df[c]):
            num.append(c)
        elif t in (CATEGORICAL, BINARY):
            cat.append(c)
        elif t == CONSTANT:
            ignored.append(c)
        else:
            ignored.append(c)
    return num, cat, ignored


def build_preprocessor(cfg: PreprocessConfig, num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if num_cols:
        steps: list[tuple[str, Any]] = []
        if cfg.outlier_treatment == "clip_iqr":
            steps.append(("clip_iqr", IQRClipper(1.5)))
        strategy = cfg.numeric_imputation if cfg.numeric_imputation != "drop_rows" else "median"
        steps.append(("imputer", SimpleImputer(
            strategy=strategy, fill_value=cfg.numeric_fill_value if strategy == "constant" else None,
            add_indicator=cfg.add_missing_indicator)))
        scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}.get(cfg.scaling)
        if scaler is not None:
            steps.append(("scaler", scaler))
        transformers.append(("num", Pipeline(steps), num_cols))
    if cat_cols:
        cstrat = cfg.categorical_imputation if cfg.categorical_imputation != "drop_rows" else "most_frequent"
        steps = [("imputer", SimpleImputer(strategy=cstrat, fill_value="(ausente)" if cstrat == "constant" else None))]
        if cfg.encoding == "ordinal":
            steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        else:
            steps.append(("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                   max_categories=max(2, cfg.max_categories),
                                                   sparse_output=False)))
        transformers.append(("cat", Pipeline(steps), cat_cols))
    if not transformers:
        raise PreprocessError("Nenhum atributo utilizável foi selecionado.")
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
    ct.set_output(transform="pandas")
    return ct


def build_selector(cfg: PreprocessConfig, task: str):
    fs = cfg.feature_selection
    if fs.method == "kbest_f":
        return SelectKBest(score_func=f_classif if task == CLASSIFICATION else f_regression, k=max(1, fs.k))
    if fs.method == "kbest_mi":
        func = mutual_info_classif if task == CLASSIFICATION else mutual_info_regression
        from functools import partial
        return SelectKBest(score_func=partial(func, random_state=cfg.random_state), k=max(1, fs.k))
    if fs.method == "variance":
        return VarianceThreshold(threshold=max(0.0, fs.variance_threshold))
    return "passthrough"


def _cast_cats(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
        df[c] = df[c].map(lambda v: v if (isinstance(v, float) and np.isnan(v)) else str(v))
    return df


def detect_leakage(X_train: pd.DataFrame, y_train: np.ndarray, num_cols: list[str], cat_cols: list[str],
                   task: str, seed: int, max_rows: int = 5000) -> list[dict]:
    """Testa o poder preditivo de CADA atributo isoladamente (só no treino).

    Uma árvore rasa usando uma única coluna é avaliada com validação cruzada
    de 3 folds. Se um único atributo prevê o alvo quase perfeitamente, é forte
    indício de vazamento (variável derivada do alvo ou medida após o evento).
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    n = len(X_train)
    idx = np.random.default_rng(seed).choice(n, min(n, max_rows), replace=False)
    Xs, ys = X_train.iloc[idx], y_train[idx]
    if task == CLASSIFICATION and np.bincount(ys).min() < 3:
        return []
    found = []
    for c in num_cols + cat_cols:
        col = Xs[[c]]
        if c in cat_cols:
            col = pd.DataFrame({c: col[c].astype("category").cat.codes})
        col = col.fillna(-999999)
        if task == CLASSIFICATION:
            est, scoring, metric, thr = DecisionTreeClassifier(max_depth=4, random_state=seed), \
                "balanced_accuracy", "balanced accuracy", 0.985
        else:
            est, scoring, metric, thr = DecisionTreeRegressor(max_depth=6, random_state=seed), "r2", "R²", 0.98
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = float(np.mean(cross_val_score(est, col, ys, cv=3, scoring=scoring)))
        except ValueError:
            continue
        if score >= thr:
            found.append({"feature": c, "score": score, "metric": metric})
    return found


class PreparedData:
    """Resultado da preparação: dados separados + metadados (antes do ajuste)."""

    def __init__(self, **kw: Any):
        self.__dict__.update(kw)


def prepare(df: pd.DataFrame, cfg: PreprocessConfig) -> PreparedData:
    if cfg.target not in df.columns:
        raise PreprocessError(f"Coluna alvo '{cfg.target}' não encontrada.")
    notes: list[str] = []
    warnings_list: list[str] = []

    n0 = len(df)
    df = df[df[cfg.target].notna()].copy()
    if len(df) < n0:
        notes.append(f"{n0 - len(df)} linha(s) sem valor no alvo foram removidas.")

    task = cfg.task if cfg.task in (CLASSIFICATION, REGRESSION) else infer_task(df[cfg.target])
    if task == REGRESSION and not pd.api.types.is_numeric_dtype(df[cfg.target]):
        raise PreprocessError("Regressão exige um alvo numérico.")

    features = cfg.features or default_features(df, cfg.target, cfg.time_column)
    features = [f for f in features if f in df.columns and f not in (cfg.target, cfg.time_column)]
    num_cols, cat_cols, ignored = split_feature_kinds(df, features)
    if ignored:
        notes.append(f"Colunas ignoradas (constantes, texto livre ou data): {', '.join(ignored)}.")
    if not num_cols and not cat_cols:
        raise PreprocessError("Nenhum atributo numérico ou categórico disponível.")

    # Remoção de linhas (determinística, não aprende estatística -> sem vazamento)
    drop_subset = []
    if cfg.numeric_imputation == "drop_rows":
        drop_subset += num_cols
    if cfg.categorical_imputation == "drop_rows":
        drop_subset += cat_cols
    if drop_subset:
        before = len(df)
        df = df.dropna(subset=drop_subset)
        notes.append(f"{before - len(df)} linha(s) com valores ausentes foram removidas.")

    # Alvo
    y_raw = df[cfg.target]
    class_labels: list[str] | None = None
    positive_class = None
    if task == CLASSIFICATION:
        y_str = y_raw.astype(str)
        counts = y_str.value_counts()
        if counts.size < 2:
            raise PreprocessError("O alvo precisa ter pelo menos 2 classes.")
        if counts.size > 50:
            raise PreprocessError(f"O alvo tem {counts.size} classes — parece um problema de regressão.")
        if counts.size == 2:
            labels = sorted(counts.index.tolist())
            if cfg.positive_class is not None and str(cfg.positive_class) in labels:
                positive_class = str(cfg.positive_class)
            elif set(labels) == {"0", "1"}:
                positive_class = "1"
            elif set(labels) in ({"True", "False"}, {"true", "false"}):
                positive_class = [lab for lab in labels if lab.lower() == "true"][0]
            else:
                positive_class = counts.index[-1]  # classe minoritária
            negative = [lab for lab in labels if lab != positive_class][0]
            class_labels = [negative, positive_class]
            y = (y_str == positive_class).astype(int).to_numpy()
        else:
            class_labels = sorted(counts.index.tolist())
            mapping = {c: i for i, c in enumerate(class_labels)}
            y = y_str.map(mapping).to_numpy()
        if counts.min() / counts.sum() < 0.1:
            warnings_list.append(
                f"Classes desbalanceadas: a menor classe representa {counts.min() / counts.sum() * 100:.1f}% "
                "dos dados. Accuracy pode enganar — priorize Recall, F1 e balanced accuracy.")
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)

    X = _cast_cats(df[num_cols + cat_cols], cat_cols)
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)

    # Separação treino/teste
    strategy = cfg.split_strategy
    if strategy == "temporal":
        if cfg.time_column and cfg.time_column in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = pd.to_datetime(df[cfg.time_column], errors="coerce", format="mixed")
            if t.notna().mean() < 0.9:
                t = pd.to_numeric(df[cfg.time_column], errors="coerce")
            order = np.argsort(t.to_numpy(), kind="stable")
            notes.append(f"Dados ordenados por '{cfg.time_column}' para a separação temporal.")
        else:
            order = np.arange(len(df))
            notes.append("Separação temporal pela ordem original das linhas.")
        X = X.iloc[order].reset_index(drop=True)
        y = y[order]
        n_test = max(1, int(round(len(X) * cfg.test_size)))
        X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]
    else:
        stratify = None
        if strategy == "stratified" and task == CLASSIFICATION:
            if np.bincount(y).min() >= 2:
                stratify = y
            else:
                warnings_list.append("Alguma classe tem menos de 2 amostras: separação estratificada "
                                     "desativada.")
                strategy = "random"
        elif strategy == "stratified":
            strategy = "random"
            notes.append("Estratificação só se aplica à classificação; usada separação aleatória.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=stratify, shuffle=True)

    if len(X_train) < 10:
        raise PreprocessError("Conjunto de treino muito pequeno (< 10 linhas).")

    if task == CLASSIFICATION:
        missing_in_test = set(range(len(class_labels))) - set(np.unique(y_test))
        if missing_in_test:
            warnings_list.append("Alguma classe não aparece no conjunto de teste; as métricas por classe "
                                 "podem ficar incompletas.")

    # Alertas de possível vazamento — calculados SÓ no treino
    leakage = detect_leakage(X_train, y_train, num_cols, cat_cols, task, cfg.random_state)
    for item in leakage:
        warnings_list.append(
            f"Possível vazamento: '{item['feature']}' sozinha atinge {item['metric']} = {item['score']:.3f} "
            "no treino. Verifique se ela é derivada do alvo ou registrada depois do evento que se quer prever; "
            "se for, remova-a dos atributos.")

    return PreparedData(
        task=task, features=num_cols + cat_cols, num_cols=num_cols, cat_cols=cat_cols, ignored=ignored,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        class_labels=class_labels, positive_class=positive_class, split_strategy=strategy,
        notes=notes, warnings=warnings_list, leakage=leakage,
    )


def _class_dist(y: np.ndarray, labels: list[str]) -> list[dict]:
    counts = np.bincount(y, minlength=len(labels))
    return [{"label": labels[i], "count": int(c), "pct": float(c / counts.sum() * 100)} for i, c in enumerate(counts)]


def preview(df: pd.DataFrame, cfg: PreprocessConfig) -> dict:
    """Ajusta o pré-processamento APENAS no treino e mostra o que foi aprendido."""
    prep = prepare(df, cfg)
    ct = build_preprocessor(cfg, prep.num_cols, prep.cat_cols)
    selector = build_selector(cfg, prep.task)
    pipe = Pipeline([("prep", ct), ("select", selector)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xt_train = pipe.fit_transform(prep.X_train, prep.y_train)
        Xt_test = pipe.transform(prep.X_test)
    feat_after_prep = list(ct.get_feature_names_out())
    selected = list(pipe.get_feature_names_out()) if selector != "passthrough" else feat_after_prep

    learned: list[dict] = []
    if "num" in ct.named_transformers_:
        num_pipe = ct.named_transformers_["num"]
        imp = num_pipe.named_steps["imputer"]
        scaler = num_pipe.named_steps.get("scaler")
        clip = num_pipe.named_steps.get("clip_iqr")
        for i, c in enumerate(prep.num_cols):
            row = {"column": c, "kind": "numérica", "impute_value": float(imp.statistics_[i])
                   if imp.statistics_[i] is not None and not pd.isna(imp.statistics_[i]) else None,
                   "train_missing": int(prep.X_train[c].isna().sum()),
                   "test_missing": int(prep.X_test[c].isna().sum())}
            if isinstance(scaler, StandardScaler):
                row.update(scaler="StandardScaler", center=float(scaler.mean_[i]), scale=float(scaler.scale_[i]))
            elif isinstance(scaler, MinMaxScaler):
                row.update(scaler="MinMaxScaler", center=float(scaler.data_min_[i]),
                           scale=float(scaler.data_range_[i]))
            elif isinstance(scaler, RobustScaler):
                row.update(scaler="RobustScaler", center=float(scaler.center_[i]), scale=float(scaler.scale_[i]))
            if clip is not None:
                row.update(clip_lower=float(clip.lower_[i]), clip_upper=float(clip.upper_[i]))
            learned.append(row)
    if "cat" in ct.named_transformers_:
        cat_pipe = ct.named_transformers_["cat"]
        imp = cat_pipe.named_steps["imputer"]
        enc = cat_pipe.named_steps["encoder"]
        for i, c in enumerate(prep.cat_cols):
            cats = [str(v) for v in enc.categories_[i]]
            learned.append({"column": c, "kind": "categórica", "impute_value": str(imp.statistics_[i]),
                            "train_missing": int(prep.X_train[c].isna().sum()),
                            "test_missing": int(prep.X_test[c].isna().sum()),
                            "n_categories": len(cats), "categories": cats[:30]})

    selection_scores = None
    if isinstance(selector, SelectKBest):
        scores = selector.scores_
        selection_scores = sorted(
            [{"feature": f, "score": None if pd.isna(s) else float(s), "selected": bool(m)}
             for f, s, m in zip(feat_after_prep, scores, selector.get_support())],
            key=lambda d: -(d["score"] or 0))
    elif isinstance(selector, VarianceThreshold):
        selection_scores = sorted(
            [{"feature": f, "score": float(v), "selected": bool(m)}
             for f, v, m in zip(feat_after_prep, selector.variances_, selector.get_support())],
            key=lambda d: -d["score"])

    sample = pd.DataFrame(np.asarray(Xt_train)[:15], columns=selected)
    out = {
        "task": prep.task,
        "n_train": int(len(prep.X_train)), "n_test": int(len(prep.X_test)),
        "split_strategy": prep.split_strategy,
        "numeric_features": prep.num_cols, "categorical_features": prep.cat_cols, "ignored": prep.ignored,
        "n_features_in": len(prep.features), "n_features_after_encoding": len(feat_after_prep),
        "n_features_selected": len(selected), "selected_features": selected,
        "learned_params": learned, "selection_scores": selection_scores,
        "transformed_sample": {"columns": selected, "rows": sample.round(5).values.tolist()},
        "test_transformed_shape": list(np.asarray(Xt_test).shape),
        "notes": prep.notes, "warnings": prep.warnings, "leakage": prep.leakage,
        "class_labels": prep.class_labels, "positive_class": prep.positive_class,
    }
    if prep.task == CLASSIFICATION:
        out["class_distribution"] = {"train": _class_dist(prep.y_train, prep.class_labels),
                                     "test": _class_dist(prep.y_test, prep.class_labels)}
    else:
        out["target_summary"] = {
            "train": {"mean": float(np.mean(prep.y_train)), "std": float(np.std(prep.y_train, ddof=1))},
            "test": {"mean": float(np.mean(prep.y_test)), "std": float(np.std(prep.y_test, ddof=1))},
        }
    return out

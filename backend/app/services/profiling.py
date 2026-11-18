"""Perfil do dataset: tipos de variáveis, estatísticas, ausentes e outliers.

Tudo é calculado diretamente sobre os dados carregados — nada é estimado.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Tipos semânticos usados em todo o app
NUMERIC_CONT = "numerica_continua"
NUMERIC_DISC = "numerica_discreta"
BINARY = "binaria"
CATEGORICAL = "categorica"
DATETIME = "data_hora"
TEXT = "texto_identificador"
CONSTANT = "constante"

TYPE_LABELS = {
    NUMERIC_CONT: "Numérica contínua",
    NUMERIC_DISC: "Numérica discreta",
    BINARY: "Binária",
    CATEGORICAL: "Categórica",
    DATETIME: "Data/hora",
    TEXT: "Texto / identificador",
    CONSTANT: "Constante",
}


def _looks_datetime(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(500)
    if sample.empty:
        return False
    # Precisa ter dígitos e separadores típicos de data
    if not sample.str.contains(r"\d{1,4}[-/:.]\d{1,2}", regex=True).mean() > 0.9:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=False)
    return parsed.notna().mean() > 0.9


def infer_column_type(s: pd.Series) -> str:
    non_null = s.dropna()
    n_unique = non_null.nunique()
    if n_unique <= 1:
        return CONSTANT
    if pd.api.types.is_bool_dtype(s):
        return BINARY
    if pd.api.types.is_datetime64_any_dtype(s):
        return DATETIME
    if pd.api.types.is_numeric_dtype(s):
        if n_unique == 2:
            return BINARY
        is_integer_like = bool(np.all(np.mod(non_null.to_numpy(dtype=float), 1) == 0))
        if is_integer_like and n_unique <= 20:
            return NUMERIC_DISC
        return NUMERIC_CONT
    # object / string
    if n_unique == 2:
        return BINARY
    if _looks_datetime(s):
        return DATETIME
    ratio = n_unique / max(len(non_null), 1)
    if n_unique <= 50 or ratio < 0.05:
        return CATEGORICAL
    return TEXT


def column_types(df: pd.DataFrame) -> dict[str, str]:
    return {c: infer_column_type(df[c]) for c in df.columns}


def is_numeric_type(t: str) -> bool:
    return t in (NUMERIC_CONT, NUMERIC_DISC)


def outlier_stats(x: pd.Series) -> dict:
    """Outliers por IQR (Tukey, 1,5×IQR) e por z-score (|z| > 3)."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    if n < 4:
        return {"iqr_count": 0, "iqr_pct": 0.0, "z_count": 0, "z_pct": 0.0,
                "lower_bound": None, "upper_bound": None}
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_mask = (x < lo) | (x > hi)
    std = x.std(ddof=1)
    z_mask = (np.abs((x - x.mean()) / std) > 3) if std > 0 else pd.Series(False, index=x.index)
    return {
        "iqr_count": int(iqr_mask.sum()),
        "iqr_pct": float(iqr_mask.mean() * 100),
        "z_count": int(z_mask.sum()),
        "z_pct": float(z_mask.mean() * 100),
        "lower_bound": float(lo),
        "upper_bound": float(hi),
    }


def numeric_summary(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return {}
    q = np.percentile(x, [25, 50, 75])
    return {
        "count": int(x.count()),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else None,
        "min": float(x.min()),
        "q1": float(q[0]),
        "median": float(q[1]),
        "q3": float(q[2]),
        "max": float(x.max()),
        "skewness": float(x.skew()) if len(x) > 2 else None,
        "kurtosis": float(x.kurt()) if len(x) > 3 else None,
        "cv_pct": float(x.std(ddof=1) / abs(x.mean()) * 100) if len(x) > 1 and x.mean() != 0 else None,
    }


def categorical_summary(s: pd.Series, top: int = 10) -> dict:
    s = s.dropna().astype(str)
    if s.empty:
        return {}
    vc = s.value_counts()
    return {
        "count": int(s.count()),
        "unique": int(vc.size),
        "top": str(vc.index[0]),
        "top_freq": int(vc.iloc[0]),
        "top_values": [{"value": str(k), "count": int(v), "pct": float(v / len(s) * 100)}
                       for k, v in vc.head(top).items()],
    }


def profile_dataset(df: pd.DataFrame) -> dict:
    types = column_types(df)
    n = len(df)
    columns = []
    for c in df.columns:
        s = df[c]
        t = types[c]
        missing = int(s.isna().sum())
        info: dict = {
            "name": c,
            "dtype": str(s.dtype),
            "semantic_type": t,
            "semantic_label": TYPE_LABELS[t],
            "missing": missing,
            "missing_pct": float(missing / n * 100) if n else 0.0,
            "unique": int(s.nunique(dropna=True)),
            "possible_id": bool(s.nunique(dropna=True) == n and n > 20 and t in (TEXT, NUMERIC_DISC, NUMERIC_CONT)
                                and (pd.api.types.is_integer_dtype(s) or t == TEXT)),
        }
        if is_numeric_type(t) or (t == BINARY and pd.api.types.is_numeric_dtype(s)):
            info["stats"] = numeric_summary(s)
            info["outliers"] = outlier_stats(s) if is_numeric_type(t) else None
        else:
            info["stats"] = categorical_summary(s)
            info["outliers"] = None
        columns.append(info)

    rows_with_missing = int(df.isna().any(axis=1).sum())
    warnings_list = []
    for col in columns:
        if col["possible_id"]:
            warnings_list.append(f"'{col['name']}' tem um valor único por linha — provavelmente um identificador. "
                                 "Não use como atributo do modelo.")
        if col["semantic_type"] == CONSTANT:
            warnings_list.append(f"'{col['name']}' é constante e não carrega informação.")
        if col["missing_pct"] > 40:
            warnings_list.append(f"'{col['name']}' tem {col['missing_pct']:.1f}% de valores ausentes.")
    dup = int(df.duplicated().sum())
    if dup:
        warnings_list.append(f"O dataset possui {dup} linha(s) duplicada(s).")

    return {
        "n_rows": int(n),
        "n_cols": int(df.shape[1]),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
        "duplicated_rows": dup,
        "rows_with_missing": rows_with_missing,
        "rows_with_missing_pct": float(rows_with_missing / n * 100) if n else 0.0,
        "total_missing": int(df.isna().sum().sum()),
        "type_counts": {TYPE_LABELS[k]: sum(1 for v in types.values() if v == k) for k in TYPE_LABELS
                        if any(v == k for v in types.values())},
        "columns": columns,
        "warnings": warnings_list,
    }

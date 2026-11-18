"""Análise exploratória: filtros, histogramas, boxplots, correlação, dispersão e
análise de distribuição. Os agregados são calculados no servidor; o frontend
apenas desenha com Plotly.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..core.config import MAX_SCATTER_POINTS
from .profiling import column_types, is_numeric_type

MAX_GROUPS = 12


class EDAError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Filtros                                                                     #
# --------------------------------------------------------------------------- #
def apply_filters(df: pd.DataFrame, filters: list[dict] | None) -> pd.DataFrame:
    """Filtros suportados: between, in, eq, neq, gt, gte, lt, lte, notnull, isnull."""
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for f in filters:
        col, op, val = f.get("column"), f.get("op"), f.get("value")
        if col not in df.columns:
            raise EDAError(f"Coluna inexistente no filtro: {col}")
        s = df[col]
        if op == "between":
            lo, hi = val if isinstance(val, (list, tuple)) else (None, None)
            num = pd.to_numeric(s, errors="coerce")
            if lo is not None:
                mask &= num >= float(lo)
            if hi is not None:
                mask &= num <= float(hi)
        elif op == "in":
            vals = [str(v) for v in (val or [])]
            mask &= s.astype(str).isin(vals)
        elif op == "eq":
            mask &= s.astype(str) == str(val)
        elif op == "neq":
            mask &= s.astype(str) != str(val)
        elif op in ("gt", "gte", "lt", "lte"):
            num = pd.to_numeric(s, errors="coerce")
            v = float(val)
            mask &= {"gt": num > v, "gte": num >= v, "lt": num < v, "lte": num <= v}[op]
        elif op == "notnull":
            mask &= s.notna()
        elif op == "isnull":
            mask &= s.isna()
        else:
            raise EDAError(f"Operador de filtro inválido: {op}")
    return df[mask.fillna(False)]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise EDAError(f"Coluna inexistente: {col}")
    return pd.to_numeric(df[col], errors="coerce")


def _groups(df: pd.DataFrame, color_by: str | None) -> list[tuple[str, pd.DataFrame]]:
    if not color_by:
        return [("Todos", df)]
    if color_by not in df.columns:
        raise EDAError(f"Coluna inexistente: {color_by}")
    g = df[color_by].astype(str).where(df[color_by].notna(), "(ausente)")
    top = g.value_counts().index[:MAX_GROUPS]
    return [(str(k), df[g == k]) for k in top]


# --------------------------------------------------------------------------- #
# Histograma                                                                  #
# --------------------------------------------------------------------------- #
def histogram(df: pd.DataFrame, column: str, bins: int = 30, color_by: str | None = None) -> dict:
    x_all = _num(df, column).dropna()
    if x_all.empty:
        # Coluna categórica: contagem por categoria
        s = df[column].dropna().astype(str)
        vc = s.value_counts().head(40)
        return {"kind": "categorical", "column": column,
                "categories": vc.index.tolist(), "counts": vc.values.tolist(), "n": int(len(s))}
    bins = int(np.clip(bins, 2, 200))
    edges = np.histogram_bin_edges(x_all, bins=bins)
    series = []
    for name, g in _groups(df, color_by):
        x = _num(g, column).dropna()
        counts, _ = np.histogram(x, bins=edges)
        series.append({"name": name, "counts": counts.tolist(), "n": int(len(x))})
    return {"kind": "numeric", "column": column, "edges": edges.tolist(),
            "series": series, "n": int(len(x_all))}


# --------------------------------------------------------------------------- #
# Boxplot (estatísticas pré-calculadas)                                       #
# --------------------------------------------------------------------------- #
def _box_stats(x: pd.Series, max_outliers: int = 1500) -> dict:
    x = x.dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return {}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = x[(x >= lo) & (x <= hi)]
    outs = x[(x < lo) | (x > hi)]
    n_out = len(outs)
    if n_out > max_outliers:
        rng = np.random.default_rng(0)
        outs = rng.choice(outs, max_outliers, replace=False)
    return {
        "q1": float(q1), "median": float(med), "q3": float(q3),
        "lowerfence": float(inside.min()) if len(inside) else float(q1),
        "upperfence": float(inside.max()) if len(inside) else float(q3),
        "mean": float(x.mean()), "n": int(len(x)),
        "outliers": outs.tolist(), "n_outliers": int(n_out),
    }


def boxplot(df: pd.DataFrame, columns: list[str], group_by: str | None = None) -> dict:
    boxes = []
    for col in columns:
        for name, g in _groups(df, group_by):
            st = _box_stats(_num(g, col))
            if st:
                boxes.append({"column": col, "group": name, **st})
    return {"boxes": boxes, "group_by": group_by}


# --------------------------------------------------------------------------- #
# Correlação                                                                  #
# --------------------------------------------------------------------------- #
def correlation(df: pd.DataFrame, method: str = "pearson", columns: list[str] | None = None) -> dict:
    if method not in ("pearson", "spearman", "kendall"):
        raise EDAError("Método deve ser pearson, spearman ou kendall.")
    types = column_types(df)
    num_cols = [c for c in (columns or df.columns) if is_numeric_type(types.get(c, ""))
                or (types.get(c) == "binaria" and pd.api.types.is_numeric_dtype(df[c]))]
    if len(num_cols) < 2:
        raise EDAError("São necessárias ao menos 2 colunas numéricas.")
    num_cols = num_cols[:60]
    m = df[num_cols].apply(pd.to_numeric, errors="coerce").corr(method=method)
    pairs = []
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            v = m.loc[a, b]
            if pd.notna(v):
                pairs.append({"a": a, "b": b, "r": float(v)})
    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)
    return {"method": method, "columns": num_cols,
            "matrix": [[None if pd.isna(v) else float(v) for v in row] for row in m.values],
            "top_pairs": pairs[:15]}


# --------------------------------------------------------------------------- #
# Dispersão                                                                   #
# --------------------------------------------------------------------------- #
def scatter(df: pd.DataFrame, x: str, y: str, color_by: str | None = None,
            max_points: int = MAX_SCATTER_POINTS) -> dict:
    cols = [x, y] + ([color_by] if color_by else [])
    sub = df[cols].copy()
    sub[x] = _num(sub, x)
    sub[y] = _num(sub, y)
    sub = sub.dropna(subset=[x, y])
    n_total = len(sub)
    if n_total < 2:
        raise EDAError("Dados insuficientes para o gráfico de dispersão.")
    # Estatísticas calculadas sobre TODOS os pontos válidos
    pr, pp = sps.pearsonr(sub[x], sub[y]) if sub[x].std() > 0 and sub[y].std() > 0 else (np.nan, np.nan)
    sr, _ = sps.spearmanr(sub[x], sub[y]) if sub[x].std() > 0 and sub[y].std() > 0 else (np.nan, np.nan)
    slope, intercept = np.polyfit(sub[x], sub[y], 1) if sub[x].std() > 0 else (np.nan, np.nan)
    sampled = n_total > max_points
    plot_df = sub.sample(max_points, random_state=0) if sampled else sub
    series = []
    for name, g in _groups(plot_df, color_by):
        series.append({"name": name, "x": g[x].tolist(), "y": g[y].tolist()})
    xr = [float(sub[x].min()), float(sub[x].max())]
    return {
        "x": x, "y": y, "series": series, "n_total": int(n_total),
        "n_plotted": int(len(plot_df)), "sampled": bool(sampled),
        "pearson_r": pr, "pearson_p": pp, "spearman_rho": sr,
        "trendline": None if np.isnan(slope) else {
            "slope": float(slope), "intercept": float(intercept),
            "x": xr, "y": [float(slope * v + intercept) for v in xr]},
    }


# --------------------------------------------------------------------------- #
# Distribuição                                                                #
# --------------------------------------------------------------------------- #
def distribution(df: pd.DataFrame, column: str) -> dict[str, Any]:
    x = _num(df, column).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 8:
        raise EDAError("São necessários ao menos 8 valores numéricos para analisar a distribuição.")
    skew = float(sps.skew(x, bias=False))
    kurt = float(sps.kurtosis(x, bias=False))  # excesso de curtose (normal = 0)
    tests = []
    if n <= 5000:
        w, p = sps.shapiro(x)
        tests.append({"name": "Shapiro-Wilk", "statistic": float(w), "p_value": float(p)})
    k2, p2 = sps.normaltest(x)
    tests.append({"name": "D'Agostino-Pearson", "statistic": float(k2), "p_value": float(p2)})
    std = x.std(ddof=1)
    if std > 0:
        ks = sps.kstest((x - x.mean()) / std, "norm")
        tests.append({"name": "Kolmogorov-Smirnov (vs. normal padronizada)",
                      "statistic": float(ks.statistic), "p_value": float(ks.pvalue)})
    # Q-Q plot (no máx. 500 pontos, quantis uniformemente espaçados)
    probs = (np.arange(1, min(n, 500) + 1) - 0.5) / min(n, 500)
    sample_q = np.quantile(x, probs)
    theo_q = sps.norm.ppf(probs, loc=x.mean(), scale=std if std > 0 else 1)
    # KDE para sobrepor ao histograma
    kde = None
    if std > 0 and np.unique(x).size > 2:
        grid = np.linspace(x.min(), x.max(), 200)
        kde = {"x": grid.tolist(), "y": sps.gaussian_kde(x)(grid).tolist()}

    if abs(skew) < 0.5:
        shape = "aproximadamente simétrica"
    elif skew > 0:
        shape = "assimétrica à direita (cauda longa para valores altos)"
    else:
        shape = "assimétrica à esquerda (cauda longa para valores baixos)"
    suggest_log = bool(skew > 1 and x.min() > 0)
    return {
        "column": column, "n": n, "skewness": skew, "excess_kurtosis": kurt,
        "shape": shape, "normality_tests": tests,
        "qq": {"theoretical": theo_q.tolist(), "sample": sample_q.tolist()},
        "kde": kde, "suggest_log_transform": suggest_log,
    }

"""Utilitários de serialização JSON (NaN/Inf -> null, tipos NumPy -> Python)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def clean(obj: Any) -> Any:
    """Converte recursivamente para tipos serializáveis em JSON."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    if obj is None or isinstance(obj, (str, int)):
        return obj
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def df_records(df: pd.DataFrame) -> list[dict]:
    return clean(df.to_dict(orient="records"))

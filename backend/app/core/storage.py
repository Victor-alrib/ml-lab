"""Persistência simples em disco (sem banco de dados).

- Datasets: storage/datasets/<id>/data.csv + meta.json
- Experimentos: storage/experiments/<id>.json

Os DataFrames carregados ficam em cache na memória, invalidado pela data de
modificação do arquivo.
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATASETS_DIR, EXPERIMENTS_DIR, ensure_dirs


class NotFoundError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Leitura de CSV                                                              #
# --------------------------------------------------------------------------- #
def _decode(raw: bytes) -> tuple[str, str]:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise ValueError("Não foi possível decodificar o arquivo (tente UTF-8).")


def _sniff_separator(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        # Heurística: separador mais frequente na primeira linha
        first = sample.splitlines()[0] if sample else ""
        counts = {d: first.count(d) for d in [",", ";", "\t", "|"]}
        return max(counts, key=counts.get) if any(counts.values()) else ","


def parse_csv(raw: bytes, sep: str | None = None, decimal: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Lê bytes de CSV detectando encoding, separador e separador decimal.

    Arquivos brasileiros costumam usar ';' como separador e ',' como decimal;
    isso é detectado automaticamente.
    """
    text, encoding = _decode(raw)
    if not text.strip():
        raise ValueError("Arquivo vazio.")
    sample = text[:20000]
    sep = sep or _sniff_separator(sample)
    if decimal is None:
        decimal = "," if sep == ";" else "."
    df = pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal, low_memory=False)
    # Remove colunas sem nome totalmente vazias (ex.: ';' sobrando no fim da linha)
    unnamed_empty = [
        c for c in df.columns
        if str(c).startswith("Unnamed:") and df[c].isna().all()
    ]
    df = df.drop(columns=unnamed_empty)
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("O CSV possui nomes de colunas duplicados.")
    if df.shape[1] < 2:
        raise ValueError(
            "Foi detectada apenas 1 coluna. Verifique o separador (',' ou ';')."
        )
    return df, {"encoding": encoding, "sep": sep, "decimal": decimal}


# --------------------------------------------------------------------------- #
# Datasets                                                                    #
# --------------------------------------------------------------------------- #
_df_cache: dict[str, tuple[float, pd.DataFrame]] = {}


def _ds_dir(dataset_id: str) -> Path:
    if not dataset_id or "/" in dataset_id or ".." in dataset_id:
        raise NotFoundError(dataset_id)
    return DATASETS_DIR / dataset_id


def save_dataset(df: pd.DataFrame, name: str, source: str, parse_info: dict | None = None,
                 description: str | None = None) -> dict:
    ensure_dirs()
    dataset_id = uuid.uuid4().hex[:12]
    d = _ds_dir(dataset_id)
    d.mkdir(parents=True)
    df.to_csv(d / "data.csv", index=False)
    meta = {
        "id": dataset_id,
        "name": name,
        "source": source,
        "description": description,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "created_at": _now(),
        "parse_info": parse_info or {},
        "size_bytes": (d / "data.csv").stat().st_size,
    }
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def list_datasets() -> list[dict]:
    ensure_dirs()
    metas = []
    for d in DATASETS_DIR.iterdir():
        mf = d / "meta.json"
        if mf.exists():
            metas.append(json.loads(mf.read_text(encoding="utf-8")))
    return sorted(metas, key=lambda m: m["created_at"], reverse=True)


def get_dataset_meta(dataset_id: str) -> dict:
    mf = _ds_dir(dataset_id) / "meta.json"
    if not mf.exists():
        raise NotFoundError(dataset_id)
    return json.loads(mf.read_text(encoding="utf-8"))


def load_dataset(dataset_id: str) -> pd.DataFrame:
    path = _ds_dir(dataset_id) / "data.csv"
    if not path.exists():
        raise NotFoundError(dataset_id)
    mtime = path.stat().st_mtime
    cached = _df_cache.get(dataset_id)
    if cached and cached[0] == mtime:
        return cached[1].copy()
    df = pd.read_csv(path, low_memory=False)
    _df_cache[dataset_id] = (mtime, df)
    return df.copy()


def delete_dataset(dataset_id: str) -> None:
    d = _ds_dir(dataset_id)
    if not d.exists():
        raise NotFoundError(dataset_id)
    shutil.rmtree(d)
    _df_cache.pop(dataset_id, None)


# --------------------------------------------------------------------------- #
# Experimentos                                                                #
# --------------------------------------------------------------------------- #
def _exp_path(exp_id: str) -> Path:
    if not exp_id or "/" in exp_id or ".." in exp_id:
        raise NotFoundError(exp_id)
    return EXPERIMENTS_DIR / f"{exp_id}.json"


def save_experiment(exp: dict[str, Any]) -> dict:
    ensure_dirs()
    exp.setdefault("id", uuid.uuid4().hex[:12])
    exp.setdefault("created_at", _now())
    _exp_path(exp["id"]).write_text(json.dumps(exp, ensure_ascii=False), encoding="utf-8")
    return exp


def get_experiment(exp_id: str) -> dict:
    p = _exp_path(exp_id)
    if not p.exists():
        raise NotFoundError(exp_id)
    return json.loads(p.read_text(encoding="utf-8"))


def list_experiments() -> list[dict]:
    ensure_dirs()
    exps = [json.loads(p.read_text(encoding="utf-8")) for p in EXPERIMENTS_DIR.glob("*.json")]
    return sorted(exps, key=lambda e: e["created_at"], reverse=True)


def update_experiment(exp_id: str, **fields: Any) -> dict:
    exp = get_experiment(exp_id)
    exp.update(fields)
    _exp_path(exp_id).write_text(json.dumps(exp, ensure_ascii=False), encoding="utf-8")
    return exp


def delete_experiment(exp_id: str) -> None:
    p = _exp_path(exp_id)
    if not p.exists():
        raise NotFoundError(exp_id)
    p.unlink()

"""Rotas HTTP do ML Lab."""
from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..core import storage
from ..core.config import MAX_UPLOAD_MB, PREVIEW_ROWS
from ..core.utils import clean, df_records
from ..services import compare as cmp
from ..services import eda, profiling
from ..services.models import catalog
from ..services.preprocessing import PreprocessConfig, PreprocessError, default_features, infer_task, preview
from ..services.samples import SAMPLES, list_samples
from ..services.training import MODELS_DIR, TrainRequest, persist_model, run_experiment

router = APIRouter(prefix="/api")


def _load(dataset_id: str) -> tuple[dict, pd.DataFrame]:
    try:
        return storage.get_dataset_meta(dataset_id), storage.load_dataset(dataset_id)
    except storage.NotFoundError:
        raise HTTPException(404, "Dataset não encontrado.") from None


def _bad(e: Exception):
    raise HTTPException(422, str(e))


# --------------------------------------------------------------------------- #
# Saúde e catálogo                                                            #
# --------------------------------------------------------------------------- #
@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/models")
def models():
    return catalog()


# --------------------------------------------------------------------------- #
# Datasets                                                                    #
# --------------------------------------------------------------------------- #
@router.get("/datasets")
def datasets():
    return storage.list_datasets()


@router.get("/samples")
def samples():
    return list_samples()


@router.post("/samples/{key}")
def load_sample(key: str):
    if key not in SAMPLES:
        raise HTTPException(404, "Exemplo não encontrado.")
    s = SAMPLES[key]
    df = s["loader"]()
    return storage.save_dataset(df, s["name"], source=f"exemplo:{key}", description=s["description"])


@router.post("/datasets/upload")
async def upload(file: UploadFile = File(...), sep: str | None = Form(None), decimal: str | None = Form(None),
                 name: str | None = Form(None)):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Arquivo maior que {MAX_UPLOAD_MB} MB.")
    if not (file.filename or "").lower().endswith((".csv", ".txt", ".tsv")):
        raise HTTPException(422, "Envie um arquivo .csv")
    try:
        df, info = storage.parse_csv(raw, sep=sep or None, decimal=decimal or None)
    except (ValueError, pd.errors.ParserError) as e:
        raise HTTPException(422, f"Não foi possível ler o CSV: {e}") from e
    return storage.save_dataset(df, name or (file.filename or "dataset").rsplit(".", 1)[0],
                                source=f"upload:{file.filename}", parse_info=info)


@router.get("/datasets/{dataset_id}")
def dataset_meta(dataset_id: str):
    meta, df = _load(dataset_id)
    return {**meta, "columns": list(df.columns)}


@router.delete("/datasets/{dataset_id}")
def remove_dataset(dataset_id: str):
    try:
        storage.delete_dataset(dataset_id)
    except storage.NotFoundError:
        raise HTTPException(404, "Dataset não encontrado.") from None
    return {"ok": True}


class RowsQuery(BaseModel):
    offset: int = 0
    limit: int = 50
    sort_by: str | None = None
    ascending: bool = True
    filters: list[dict] | None = None
    search: str | None = None


@router.post("/datasets/{dataset_id}/rows")
def rows(dataset_id: str, q: RowsQuery):
    _, df = _load(dataset_id)
    try:
        df = eda.apply_filters(df, q.filters)
    except eda.EDAError as e:
        _bad(e)
    if q.search:
        m = df.astype(str).apply(lambda s: s.str.contains(q.search, case=False, regex=False)).any(axis=1)
        df = df[m]
    if q.sort_by and q.sort_by in df.columns:
        df = df.sort_values(q.sort_by, ascending=q.ascending, na_position="last")
    limit = max(1, min(q.limit, PREVIEW_ROWS))
    page = df.iloc[q.offset:q.offset + limit]
    return {"total": int(len(df)), "offset": q.offset, "columns": list(df.columns),
            "rows": df_records(page), "index": page.index.tolist()}


@router.get("/datasets/{dataset_id}/profile")
def profile(dataset_id: str):
    _, df = _load(dataset_id)
    return clean(profiling.profile_dataset(df))


@router.get("/datasets/{dataset_id}/target-info")
def target_info(dataset_id: str, target: str):
    _, df = _load(dataset_id)
    if target not in df.columns:
        raise HTTPException(422, "Coluna inexistente.")
    y = df[target].dropna()
    task = infer_task(y)
    out: dict[str, Any] = {"task": task, "default_features": default_features(df, target),
                           "n_unique": int(y.nunique())}
    if task == "classification":
        vc = y.astype(str).value_counts()
        out["classes"] = [{"label": k, "count": int(v)} for k, v in vc.items()][:50]
    return out


# --------------------------------------------------------------------------- #
# Análise exploratória                                                        #
# --------------------------------------------------------------------------- #
class EDARequest(BaseModel):
    kind: str
    filters: list[dict] | None = None
    column: str | None = None
    columns: list[str] | None = None
    x: str | None = None
    y: str | None = None
    color_by: str | None = None
    bins: int = 30
    method: str = "pearson"


@router.post("/datasets/{dataset_id}/eda")
def run_eda(dataset_id: str, req: EDARequest):
    _, df = _load(dataset_id)
    try:
        dff = eda.apply_filters(df, req.filters)
        n_filtered = len(dff)
        if n_filtered == 0:
            raise eda.EDAError("Nenhuma linha atende aos filtros.")
        if req.kind == "histogram":
            out = eda.histogram(dff, req.column, req.bins, req.color_by)
        elif req.kind == "boxplot":
            out = eda.boxplot(dff, req.columns or ([req.column] if req.column else []), req.color_by)
        elif req.kind == "correlation":
            out = eda.correlation(dff, req.method, req.columns)
        elif req.kind == "scatter":
            out = eda.scatter(dff, req.x, req.y, req.color_by)
        elif req.kind == "distribution":
            out = eda.distribution(dff, req.column)
        else:
            raise eda.EDAError("Tipo de análise desconhecido.")
    except (eda.EDAError, KeyError, TypeError) as e:
        _bad(e)
    return clean({**out, "rows_filtered": n_filtered, "rows_total": len(df)})


# --------------------------------------------------------------------------- #
# Pré-processamento                                                           #
# --------------------------------------------------------------------------- #
class PreviewRequest(BaseModel):
    dataset_id: str
    preprocessing: PreprocessConfig


@router.post("/preprocess/preview")
def preprocess_preview(req: PreviewRequest):
    _, df = _load(req.dataset_id)
    try:
        return clean(preview(df, req.preprocessing))
    except (PreprocessError, ValueError) as e:
        _bad(e)


# --------------------------------------------------------------------------- #
# Treinamento e experimentos                                                  #
# --------------------------------------------------------------------------- #
def _summary(e: dict) -> dict:
    keep = ["id", "name", "created_at", "dataset_id", "dataset_name", "task", "model", "model_name", "family",
            "params", "target", "n_train", "n_test", "primary_metric", "timing", "tags", "split_strategy",
            "average", "notes"]
    s = {k: e.get(k) for k in keep}
    s["metrics"] = e["metrics"]
    s["cv"] = {"folds": e["cv"]["folds"], "splitter": e["cv"]["splitter"],
               "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "folds"} for k, v in e["cv"]["metrics"].items()}
               } if e.get("cv") else None
    s["n_features"] = len(e.get("final_features", []))
    s["has_model"] = bool(e.get("model_file"))
    s["preprocessing"] = e.get("preprocessing")
    return s


@router.post("/train")
def train(req: TrainRequest):
    meta, df = _load(req.dataset_id)
    try:
        exp = run_experiment(df, req, meta)
    except (PreprocessError, ValueError) as e:
        _bad(e)
    pipe = exp.pop("_pipeline", None)
    exp = clean(exp)
    exp = storage.save_experiment(exp)
    if pipe is not None:
        exp = storage.update_experiment(exp["id"], model_file=persist_model(exp["id"], pipe, {
            "experiment_id": exp["id"], "features": exp["features_used"], "target": exp["target"],
            "task": exp["task"], "class_labels": exp["results"].get("class_labels")}))
    return exp


@router.get("/experiments")
def experiments():
    return [_summary(e) for e in storage.list_experiments()]


@router.get("/experiments/{exp_id}")
def experiment(exp_id: str):
    try:
        return storage.get_experiment(exp_id)
    except storage.NotFoundError:
        raise HTTPException(404, "Experimento não encontrado.") from None


class ExpPatch(BaseModel):
    name: str | None = None
    tags: list[str] | None = None
    notes: str | None = None


@router.patch("/experiments/{exp_id}")
def patch_experiment(exp_id: str, body: ExpPatch):
    try:
        return storage.update_experiment(exp_id, **body.model_dump(exclude_none=True))
    except storage.NotFoundError:
        raise HTTPException(404, "Experimento não encontrado.") from None


@router.delete("/experiments/{exp_id}")
def delete_experiment(exp_id: str):
    try:
        storage.delete_experiment(exp_id)
    except storage.NotFoundError:
        raise HTTPException(404, "Experimento não encontrado.") from None
    (MODELS_DIR / f"{exp_id}.joblib").unlink(missing_ok=True)
    return {"ok": True}


@router.get("/experiments/{exp_id}/model")
def download_model(exp_id: str):
    path = MODELS_DIR / f"{exp_id}.joblib"
    if not path.exists() or "/" in exp_id:
        raise HTTPException(404, "Modelo não salvo para este experimento.")
    return FileResponse(path, filename=f"mllab_{exp_id}.joblib", media_type="application/octet-stream")


class CompareRequest(BaseModel):
    ids: list[str]
    metric: str | None = None


@router.post("/experiments/compare")
def compare_experiments(req: CompareRequest):
    exps = []
    for i in req.ids[:10]:
        try:
            exps.append(storage.get_experiment(i))
        except storage.NotFoundError:
            raise HTTPException(404, f"Experimento {i} não encontrado.") from None
    return clean({**cmp.compare(exps, req.metric), "experiments": exps})

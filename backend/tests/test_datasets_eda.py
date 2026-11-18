"""Testes de upload, perfil e análise exploratória."""
import io

import numpy as np
import pandas as pd

from app.core.storage import parse_csv
from app.services.samples import motor_industrial_sintetico


def test_parse_csv_brazilian_format():
    raw = "temperatura;corrente;status\n25,5;10,2;ok\n30,1;12,7;falha\n".encode("latin-1")
    df, info = parse_csv(raw)
    assert info["sep"] == ";" and info["decimal"] == ","
    assert df["temperatura"].tolist() == [25.5, 30.1]


def test_upload_and_rows(client):
    csv = "a,b,c\n1,2,x\n3,4,y\n5,,x\n"
    r = client.post("/api/datasets/upload", files={"file": ("t.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 200
    ds = r.json()
    assert ds["n_rows"] == 3 and ds["n_cols"] == 3
    rows = client.post(f"/api/datasets/{ds['id']}/rows", json={"limit": 10}).json()
    assert rows["total"] == 3 and rows["rows"][2]["b"] is None


def test_profile_missing_and_outliers_match_manual(client, motor_id):
    prof = client.get(f"/api/datasets/{motor_id}/profile").json()
    df = motor_industrial_sintetico()
    col = next(c for c in prof["columns"] if c["name"] == "corrente_a")
    assert col["missing"] == int(df["corrente_a"].isna().sum())
    x = df["corrente_a"].dropna()
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    manual = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
    assert col["outliers"]["iqr_count"] == manual
    types = {c["name"]: c["semantic_type"] for c in prof["columns"]}
    assert types["timestamp"] == "data_hora"
    assert types["motor_id"] == "categorica"
    assert types["falha_24h"] == "binaria"


def test_histogram_counts_sum_to_n(client, motor_id):
    r = client.post(f"/api/datasets/{motor_id}/eda",
                    json={"kind": "histogram", "column": "vibracao_rms_mm_s", "bins": 25}).json()
    df = motor_industrial_sintetico()
    assert sum(r["series"][0]["counts"]) == int(df["vibracao_rms_mm_s"].notna().sum())


def test_filters_apply(client, motor_id):
    r = client.post(f"/api/datasets/{motor_id}/eda", json={
        "kind": "histogram", "column": "carga_pct",
        "filters": [{"column": "porte", "op": "in", "value": ["grande"]},
                    {"column": "carga_pct", "op": "between", "value": [50, 80]}]}).json()
    df = motor_industrial_sintetico()
    expected = df[(df.porte == "grande") & df.carga_pct.between(50, 80)]
    assert r["rows_filtered"] == len(expected)


def test_correlation_matches_pandas(client, motor_id):
    r = client.post(f"/api/datasets/{motor_id}/eda", json={"kind": "correlation", "method": "spearman"}).json()
    df = motor_industrial_sintetico()
    cols = r["columns"]
    expected = df[cols].corr(method="spearman").to_numpy()
    np.testing.assert_allclose(np.array(r["matrix"], dtype=float), expected, atol=1e-10)


def test_scatter_stats_use_all_points(client, motor_id):
    r = client.post(f"/api/datasets/{motor_id}/eda", json={
        "kind": "scatter", "x": "vibracao_rms_mm_s", "y": "temperatura_rolamento_c"}).json()
    df = motor_industrial_sintetico().dropna(subset=["vibracao_rms_mm_s", "temperatura_rolamento_c"])
    assert r["n_total"] == len(df)
    assert abs(r["pearson_r"] - df["vibracao_rms_mm_s"].corr(df["temperatura_rolamento_c"])) < 1e-10


def test_distribution(client, diabetes_id):
    r = client.post(f"/api/datasets/{diabetes_id}/eda", json={"kind": "distribution", "column": "progressao"})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 442 and len(body["normality_tests"]) >= 2
    from sklearn.datasets import load_diabetes
    y = load_diabetes().target
    assert abs(body["skewness"] - pd.Series(y).skew()) < 1e-10

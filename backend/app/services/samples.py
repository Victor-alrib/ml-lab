"""Datasets de exemplo.

- `motor_industrial_sintetico`: dados SINTÉTICOS gerados por simulação com
  semente fixa (reprodutível). As relações físicas são simplificadas e servem
  apenas para praticar o fluxo de ML — não representam um equipamento real.
- `breast_cancer`, `diabetes`, `wine`: datasets REAIS distribuídos junto com o
  scikit-learn (não exigem download).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import datasets as skd


def motor_industrial_sintetico(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    """Simula leituras horárias de 10 motores de indução.

    Alvos disponíveis:
    - `falha_24h` (classificação binária): ocorre falha nas próximas 24 h?
    - `modo_falha` (multiclasse): nenhuma / rolamento / sobreaquecimento / sobrecarga
    - `temperatura_rolamento_c` (regressão)
    """
    rng = np.random.default_rng(seed)
    motores = [f"MTR-{i:02d}" for i in range(1, 11)]
    porte_map = {m: p for m, p in zip(motores, rng.choice(["pequeno", "medio", "grande"], 10, p=[.3, .4, .3]))}
    motor = rng.choice(motores, n)
    porte = np.array([porte_map[m] for m in motor])
    porte_fator = pd.Series(porte).map({"pequeno": 0.8, "medio": 1.0, "grande": 1.3}).to_numpy()

    ts = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 365 * 24, n)), unit="h")
    hora = ts.hour.to_numpy()
    turno = np.where((hora >= 6) & (hora < 14), "A", np.where((hora >= 14) & (hora < 22), "B", "C"))

    temp_amb = 24 + 6 * np.sin(2 * np.pi * (ts.dayofyear.to_numpy() - 30) / 365) + rng.normal(0, 2, n)
    carga = np.clip(rng.beta(5, 2.5, n) * 110, 15, 115)  # % da carga nominal
    rpm = 1780 - 0.6 * carga + rng.normal(0, 6, n)
    corrente = porte_fator * (8 + 0.22 * carga) + rng.normal(0, 0.8, n)
    horas_manut = rng.gamma(2.2, 420, n)
    desgaste = np.clip(horas_manut / 2500, 0, 2.5)
    vibracao = np.clip(1.1 + 0.9 * desgaste ** 1.6 + 0.012 * carga + rng.lognormal(-1.2, 0.5, n), 0.3, None)
    temp_rol = (temp_amb + 0.0022 * corrente ** 2 * 10 + 6.5 * vibracao + 0.004 * (rpm - 1700)
                + 4 * (porte == "grande") + rng.normal(0, 2.2, n))
    umidade = np.clip(rng.normal(62, 12, n), 20, 98)
    fp = np.clip(0.72 + 0.0016 * carga + rng.normal(0, 0.02, n), 0.6, 0.97)

    # Risco latente de falha (logístico), cada modo com seu mecanismo
    z_rol = -7.6 + 1.35 * vibracao + 0.0009 * horas_manut
    z_sup = -8.0 + 0.12 * (temp_rol - 40) + 0.03 * (temp_amb - 24) + 0.01 * umidade
    z_sob = -6.5 + 0.08 * (carga - 60) + 0.15 * (corrente - 25 * porte_fator)
    probs = np.stack([1 / (1 + np.exp(-z)) for z in (z_rol, z_sup, z_sob)], axis=1)
    sorteio = rng.uniform(size=(n, 3)) < probs
    modo = np.full(n, "nenhuma", dtype=object)
    for j, nome in enumerate(["rolamento", "sobreaquecimento", "sobrecarga"]):
        modo[(modo == "nenhuma") & sorteio[:, j]] = nome

    df = pd.DataFrame({
        "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
        "motor_id": motor, "porte": porte, "turno": turno,
        "temperatura_ambiente_c": temp_amb.round(2), "umidade_pct": umidade.round(1),
        "carga_pct": carga.round(2), "rotacao_rpm": rpm.round(1), "corrente_a": corrente.round(3),
        "fator_potencia": fp.round(3), "vibracao_rms_mm_s": vibracao.round(3),
        "horas_desde_manutencao": horas_manut.round(0),
        "temperatura_rolamento_c": temp_rol.round(2),
        "modo_falha": modo, "falha_24h": (modo != "nenhuma").astype(int),
    })
    # Imperfeições realistas: sensores sem leitura e picos espúrios
    for col, frac in [("vibracao_rms_mm_s", 0.03), ("umidade_pct", 0.05), ("corrente_a", 0.015),
                      ("temperatura_ambiente_c", 0.02)]:
        df.loc[rng.uniform(size=n) < frac, col] = np.nan
    spikes = rng.uniform(size=n) < 0.004
    df.loc[spikes, "corrente_a"] = df.loc[spikes, "corrente_a"] * rng.uniform(2.5, 4, spikes.sum())
    df.loc[rng.uniform(size=n) < 0.02, "turno"] = np.nan
    return df


def _from_sklearn(loader, target_name: str, rename_target: dict | None = None) -> pd.DataFrame:
    b = loader(as_frame=True)
    df = b.frame.copy()
    if "target" in df.columns:
        df = df.rename(columns={"target": target_name})
    if rename_target:
        df[target_name] = df[target_name].map(rename_target)
    return df


SAMPLES = {
    "motor_industrial_sintetico": {
        "name": "Motores industriais (sintético)",
        "description": "Simulação SINTÉTICA e reprodutível (seed=7) de 6.000 leituras de 10 motores de indução, "
                       "com dados ausentes e picos de sensor. Alvos: falha_24h (binária), modo_falha "
                       "(multiclasse) e temperatura_rolamento_c (regressão). Atenção: modo_falha descreve o mesmo "
                       "evento de falha_24h — usá-la como atributo para prever falha_24h é vazamento de dados "
                       "(armadilha proposital para estudo). Não representa um equipamento real.",
        "task_hint": "Classificação: falha_24h · Regressão: temperatura_rolamento_c",
        "loader": motor_industrial_sintetico,
    },
    "breast_cancer": {
        "name": "Breast Cancer Wisconsin (real)",
        "description": "Dataset real (UCI) distribuído com o scikit-learn: 569 amostras, 30 atributos de "
                       "imagens de biópsia. Alvo: diagnostico (maligno/benigno).",
        "task_hint": "Classificação binária: diagnostico",
        "loader": lambda: _from_sklearn(skd.load_breast_cancer, "diagnostico", {0: "maligno", 1: "benigno"}),
    },
    "diabetes": {
        "name": "Diabetes (real)",
        "description": "Dataset real distribuído com o scikit-learn: 442 pacientes, 10 atributos já "
                       "padronizados. Alvo: progressao (medida quantitativa após 1 ano).",
        "task_hint": "Regressão: progressao",
        "loader": lambda: _from_sklearn(skd.load_diabetes, "progressao"),
    },
    "wine": {
        "name": "Wine (real)",
        "description": "Dataset real (UCI) distribuído com o scikit-learn: 178 vinhos, 13 medições químicas. "
                       "Alvo: cultivar (3 classes).",
        "task_hint": "Classificação multiclasse: cultivar",
        "loader": lambda: _from_sklearn(skd.load_wine, "cultivar", {0: "cultivar_0", 1: "cultivar_1",
                                                                    2: "cultivar_2"}),
    },
}


def list_samples() -> list[dict]:
    return [{"key": k, "name": v["name"], "description": v["description"], "task_hint": v["task_hint"]}
            for k, v in SAMPLES.items()]

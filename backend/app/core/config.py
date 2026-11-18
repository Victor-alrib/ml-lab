"""Configurações globais do ML Lab."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("MLLAB_DATA_DIR", BASE_DIR / "storage"))
DATASETS_DIR = DATA_DIR / "datasets"
EXPERIMENTS_DIR = DATA_DIR / "experiments"

# Limite de upload (MB) e de pontos enviados para gráficos de dispersão.
MAX_UPLOAD_MB = int(os.getenv("MLLAB_MAX_UPLOAD_MB", "100"))
MAX_SCATTER_POINTS = int(os.getenv("MLLAB_MAX_SCATTER_POINTS", "5000"))
PREVIEW_ROWS = 200

CORS_ORIGINS = os.getenv(
    "MLLAB_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

RANDOM_STATE_DEFAULT = 42


def ensure_dirs() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

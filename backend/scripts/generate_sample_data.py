"""Exporta os datasets de exemplo para ../sample_data/*.csv.

Uso (a partir de backend/):
    python -m scripts.generate_sample_data
"""
from __future__ import annotations

from pathlib import Path

from app.services.samples import SAMPLES

OUT = Path(__file__).resolve().parents[2] / "sample_data"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for key, spec in SAMPLES.items():
        df = spec["loader"]()
        path = OUT / f"{key}.csv"
        df.to_csv(path, index=False)
        print(f"{path.name}: {df.shape[0]} linhas x {df.shape[1]} colunas")


if __name__ == "__main__":
    main()

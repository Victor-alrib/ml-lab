import os
import tempfile

import pytest

# Armazenamento isolado para os testes (definido antes de importar o app)
_tmp = tempfile.mkdtemp(prefix="mllab_test_")
os.environ["MLLAB_DATA_DIR"] = _tmp

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(scope="session")
def motor_id(client):
    return client.post("/api/samples/motor_industrial_sintetico").json()["id"]


@pytest.fixture(scope="session")
def cancer_id(client):
    return client.post("/api/samples/breast_cancer").json()["id"]


@pytest.fixture(scope="session")
def diabetes_id(client):
    return client.post("/api/samples/diabetes").json()["id"]

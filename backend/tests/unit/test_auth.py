import time
from collections.abc import Iterator

import jwt
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from ragflow.auth import security_middleware
from ragflow.settings import get_settings

SECRET = "test-secret-with-enough-length-for-hs256"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("JWT_SECRET", SECRET)
    get_settings.cache_clear()
    app = FastAPI()
    app.middleware("http")(security_middleware)

    @app.get("/whoami")
    def whoami(request: Request) -> dict[str, object]:
        return dict(request.state.user)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "UP"}

    yield TestClient(app)
    get_settings.cache_clear()


def _token(**claims: object) -> str:
    return jwt.encode({"id": "u1", **claims}, SECRET, algorithm="HS256")


def test_missing_header_is_401(client: TestClient) -> None:
    r = client.get("/whoami")
    assert r.status_code == 401
    assert r.json() == {"detail": "Missing or invalid Authorization header"}


def test_valid_token_sets_user(client: TestClient) -> None:
    r = client.get("/whoami", headers={"Authorization": f"Bearer {_token()}"})
    assert r.status_code == 200
    assert r.json()["id"] == "u1"


def test_expired_token(client: TestClient) -> None:
    r = client.get(
        "/whoami", headers={"Authorization": f"Bearer {_token(exp=int(time.time()) - 10)}"}
    )
    assert r.status_code == 401
    assert r.json() == {"detail": "Token has expired"}


def test_bad_signature(client: TestClient) -> None:
    bad = jwt.encode({"id": "u1"}, "another-secret-of-sufficient-length!!", algorithm="HS256")
    r = client.get("/whoami", headers={"Authorization": f"Bearer {bad}"})
    assert r.status_code == 401
    assert r.json()["detail"].startswith("Invalid token:")


def test_exempt_paths_and_preflight(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    assert client.options("/whoami").status_code != 401

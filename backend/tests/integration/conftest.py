"""Real-database fixtures (run with ``uv run pytest -m integration``; needs Docker)."""

import shutil
from collections.abc import Iterator

import pytest

PGVECTOR_IMAGE = "pgvector/pgvector:pg17"


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        import docker

        docker.from_env().ping()
    except Exception:
        return False
    return True


@pytest.fixture(scope="session")
def pg_url() -> Iterator[str]:
    """SQLAlchemy psycopg3 URL of a throwaway pgvector container."""
    if not _docker_available():
        pytest.skip("Docker is not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(PGVECTOR_IMAGE, driver="psycopg") as pg:
        yield pg.get_connection_url()

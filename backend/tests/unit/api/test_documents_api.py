"""Contract tests: same paths, fields and response shapes as LibreChat's rag-api."""

from pathlib import Path

from tests.unit.api.conftest import Api

TEXT = "\n\n".join(f"Section {i}: the quick brown fox number {i} jumps." for i in range(5))


def _upload(
    api: Api,
    file_id: str = "f1",
    field: str = "file",
    route: str = "/embed",
    headers: dict[str, str] | None = None,
    entity_id: str | None = None,
) -> dict:
    data = {"file_id": file_id}
    if entity_id:
        data["entity_id"] = entity_id
    r = api.client.post(
        route,
        data=data,
        files={field: ("notes.txt", TEXT.encode(), "text/plain")},
        headers=headers or {},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_health(api: Api) -> None:
    assert api.client.get("/health").json() == {"status": "UP"}


def test_health_down_is_a_real_503(api: Api, monkeypatch) -> None:
    async def down() -> bool:
        return False

    monkeypatch.setattr(api.services.store, "health", down)
    r = api.client.get("/health")
    assert r.status_code == 503
    assert r.json() == {"status": "DOWN"}


def test_embed_then_read_back(api: Api) -> None:
    body = _upload(api)
    assert body == {
        "status": True,
        "message": "File processed successfully.",
        "file_id": "f1",
        "filename": "notes.txt",
        "known_type": True,
    }
    assert api.client.get("/ids").json() == ["f1"]

    docs = api.client.get("/documents", params={"ids": ["f1"]}).json()
    assert docs
    assert set(docs[0]) == {"page_content", "metadata"}
    assert docs[0]["metadata"]["user_id"] == "public"

    files = api.client.get("/files").json()
    assert [(f["file_id"], f["chunks"]) for f in files] == [("f1", len(docs))]


def test_embed_upload_uses_uploaded_file_field(api: Api) -> None:
    assert _upload(api, field="uploaded_file", route="/embed-upload")["status"] is True


def test_temp_upload_is_removed(api: Api) -> None:
    _upload(api)
    leftovers = [p for p in api.services.settings.rag_upload_dir.rglob("*") if p.is_file()]
    assert leftovers == []


def test_query_shapes(api: Api) -> None:
    _upload(api)
    r = api.client.post("/query", json={"query": "brown fox", "file_id": "f1", "k": 2})
    assert r.status_code == 200
    hits = r.json()
    assert len(hits) == 2
    doc, score = hits[0]
    assert {"page_content", "metadata", "type"} <= set(doc)
    assert isinstance(score, float)

    # /query returns [] for no match; /query_multiple returns 404 (rag-api parity)
    assert api.client.post("/query", json={"query": "x", "file_id": "nope"}).json() == []
    r = api.client.post("/query_multiple", json={"query": "x", "file_ids": ["nope"]})
    assert r.status_code == 404
    r = api.client.post("/query_multiple", json={"query": "fox", "file_ids": ["f1"], "k": 3})
    assert len(r.json()) == 3


def test_documents_404_and_delete(api: Api) -> None:
    _upload(api)
    assert api.client.get("/documents", params={"ids": ["f1", "zzz"]}).status_code == 404
    r = api.client.request("DELETE", "/documents", json=["f1", "zzz"])
    assert r.status_code == 404
    assert api.client.get("/ids").json() == ["f1"]  # nothing half-deleted
    r = api.client.request("DELETE", "/documents", json=["f1"])
    assert r.json() == {"message": "Documents for 1 file deleted successfully"}
    assert api.client.get("/ids").json() == []


def test_context_is_ordered_text(api: Api) -> None:
    _upload(api)
    text = api.client.get("/documents/f1/context").json()
    assert text.index("Section 0") < text.index("Section 4")
    assert api.client.get("/documents/zzz/context").status_code == 404


def test_text_extraction_does_not_store(api: Api) -> None:
    r = api.client.post(
        "/text",
        data={"file_id": "t1"},
        files={"file": ("a.md", b"# Hi\n\n*there*", "text/markdown")},
    )
    assert r.json() == {
        "text": "# Hi\n\n*there*",
        "file_id": "t1",
        "filename": "a.md",
        "known_type": True,
    }
    assert api.client.get("/ids").json() == []


def test_local_embed_path_traversal(api: Api) -> None:
    body = {
        "filepath": "../../etc/passwd",
        "filename": "p",
        "file_content_type": "text/plain",
        "file_id": "x",
    }
    r = api.client.post("/local/embed", json=body)
    assert r.status_code == 404


def test_local_embed_reports_store_failure(api: Api, monkeypatch) -> None:
    upload_dir: Path = api.services.settings.rag_upload_dir
    (upload_dir / "doc.txt").write_text(TEXT)

    async def boom(_docs: object) -> list[str]:
        raise RuntimeError("db down")

    monkeypatch.setattr(api.services.store, "aadd_documents", boom)
    body = {
        "filepath": "doc.txt",
        "filename": "doc.txt",
        "file_content_type": "text/plain",
        "file_id": "x",
    }
    r = api.client.post("/local/embed", json=body)
    assert r.status_code == 200
    assert r.json()["status"] is False
    assert "db down" in r.json()["message"]


def test_validation_error_shape(api: Api) -> None:
    r = api.client.post("/query", json={"file_id": "f1"})
    assert r.status_code == 422
    assert r.json()["message"] == "Request validation failed"


# ------------------------------------------------------------------ auth + isolation
def test_requires_token(auth_api: Api) -> None:
    assert auth_api.client.get("/ids").status_code == 401
    assert auth_api.client.get("/health").status_code == 200


def test_users_are_isolated(auth_api: Api) -> None:
    alice, bob = auth_api.token("alice"), auth_api.token("bob")
    _upload(auth_api, headers=alice)
    assert auth_api.client.get("/ids", headers=bob).json() == []
    r = auth_api.client.post("/query", json={"query": "fox", "file_id": "f1"}, headers=bob)
    assert r.json() == []
    assert auth_api.client.get("/documents/f1/context", headers=bob).status_code == 404
    r = auth_api.client.request("DELETE", "/documents", json=["f1"], headers=bob)
    assert r.status_code == 404
    assert auth_api.client.get("/ids", headers=alice).json() == ["f1"]


def test_entity_upload_is_visible_with_entity(auth_api: Api) -> None:
    alice = auth_api.token("alice")
    _upload(auth_api, headers=alice, entity_id="agent1")
    assert auth_api.client.get("/ids", headers=alice).json() == []
    assert auth_api.client.get("/ids", params={"entity_id": "agent1"}, headers=alice).json() == [
        "f1"
    ]

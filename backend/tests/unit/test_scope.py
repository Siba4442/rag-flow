from types import SimpleNamespace
from typing import Any

import pytest

from ragflow.scope import (
    PUBLIC_OWNER,
    ScopeFilter,
    file_clause,
    files_clause,
    resolve_scope,
    resolve_write_owner,
)


def _req(user: dict[str, Any] | None) -> Any:
    state = SimpleNamespace()
    if user is not None:
        state.user = user
    return SimpleNamespace(state=state)


def test_unauthenticated_scope_is_entity_or_public() -> None:
    assert resolve_scope(_req(None)).owners == (PUBLIC_OWNER,)
    assert resolve_scope(_req(None), "agent1").owners == ("agent1",)


def test_authenticated_scope_widens_with_entity(policy: Any) -> None:
    policy("trust")
    assert resolve_scope(_req({"id": "u1"})).owners == ("u1",)
    assert resolve_scope(_req({"id": "u1"}), "agent1").owners == ("agent1", "u1")


def test_token_without_id_falls_back_to_public() -> None:
    assert resolve_scope(_req({})).owners == (PUBLIC_OWNER,)


@pytest.mark.parametrize(
    ("user", "entity", "expected"),
    [
        (None, None, PUBLIC_OWNER),
        (None, "e", "e"),
        ({"id": "u1"}, None, "u1"),
        ({"id": "u1"}, "e", "e"),
    ],
)
def test_write_owner_replaces_rather_than_widens(
    user: dict[str, Any] | None, entity: str | None, expected: str, policy: Any
) -> None:
    policy("trust")
    assert resolve_write_owner(_req(user), entity) == expected


def test_predicate_always_includes_owner_clause() -> None:
    scope = ScopeFilter(owners=("a", "b"))
    assert scope.predicate(file_clause("f")) == {
        "$and": [{"file_id": {"$eq": "f"}}, {"user_id": {"$in": ["a", "b"]}}]
    }
    assert files_clause(["x", "y"]) == {"file_id": {"$in": ["x", "y"]}}


def test_row_without_owner_is_out_of_scope() -> None:
    assert not ScopeFilter(owners=("a",)).owns(None)
    assert ScopeFilter(owners=("a",)).owns("a")


@pytest.fixture
def policy(monkeypatch: pytest.MonkeyPatch) -> Any:
    from ragflow.settings import get_settings

    def set_policy(value: str) -> None:
        monkeypatch.setenv("ENTITY_ID_POLICY", value)
        get_settings.cache_clear()

    yield set_policy
    get_settings.cache_clear()


def test_default_policy_is_claim() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        resolve_scope(_req({"id": "u1"}), "agent1")


def test_deny_policy_ignores_entity(policy: Any) -> None:
    policy("deny")
    assert resolve_scope(_req({"id": "u1"}), "agent1").owners == ("u1",)
    assert resolve_write_owner(_req({"id": "u1"}), "agent1") == "u1"


def test_claim_policy_requires_entity_in_token(policy: Any) -> None:
    from fastapi import HTTPException

    policy("claim")
    user = {"id": "u1", "entities": ["agent1"]}
    assert resolve_scope(_req(user), "agent1").owners == ("agent1", "u1")
    with pytest.raises(HTTPException) as err:
        resolve_scope(_req(user), "agent2")
    assert err.value.status_code == 403

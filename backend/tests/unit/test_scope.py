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


def test_authenticated_scope_widens_with_entity() -> None:
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
    user: dict[str, Any] | None, entity: str | None, expected: str
) -> None:
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

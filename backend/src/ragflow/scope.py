"""The one place a request's scope becomes a store predicate.

Ported from ``rag-api/app/scope.py`` with the same semantics:

* Reads and deletes widen the owner set: ``{token user id or "public"} | {entity_id}``.
* Writes stamp a single owner: ``entity_id`` if given, else the token user id
  (see :func:`resolve_write_owner`).
* The owner predicate always goes into the store query *before* ranking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fastapi import Request

PUBLIC_OWNER = "public"
OWNER_METADATA_KEY = "user_id"


@dataclass(frozen=True)
class ScopeFilter:
    """The owners a request may read, as a store-query clause."""

    owners: tuple[str, ...]

    def owner_clause(self) -> dict[str, Any]:
        return {OWNER_METADATA_KEY: {"$in": list(self.owners)}}

    def predicate(self, *clauses: dict[str, Any]) -> dict[str, Any]:
        """Combine caller clauses (e.g. file id) with the owner clause."""
        return {"$and": [*clauses, self.owner_clause()]}

    def owns(self, owner: str | None) -> bool:
        """A row with no recorded owner is *not* in scope."""
        return owner in self.owners


def file_clause(file_id: str) -> dict[str, Any]:
    return {"file_id": {"$eq": file_id}}


def files_clause(file_ids: Sequence[str]) -> dict[str, Any]:
    return {"file_id": {"$in": list(file_ids)}}


def request_user(request: Request) -> dict[str, Any] | None:
    user = getattr(request.state, "user", None)
    return user if isinstance(user, dict) else None


def resolve_scope(request: Request, entity_id: str | None = None) -> ScopeFilter:
    """Resolve the owners this request may read.

    The caller's own identity comes from the verified token only. ``entity_id`` is
    caller-asserted and widens the set; deployments exposing this API to untrusted
    callers must authorize entity access upstream (same caveat as rag-api).
    """
    user = request_user(request)
    if user is None:
        return ScopeFilter(owners=(entity_id or PUBLIC_OWNER,))

    owners = {str(user.get("id") or PUBLIC_OWNER)}
    if entity_id:
        owners.add(entity_id)
    return ScopeFilter(owners=tuple(sorted(owners)))


def resolve_write_owner(request: Request, entity_id: str | None = None) -> str:
    """The owner stamped on newly embedded chunks (rag-api ``get_user_id``)."""
    user = request_user(request)
    if user is None:
        return entity_id or PUBLIC_OWNER
    return entity_id or str(user.get("id") or PUBLIC_OWNER)

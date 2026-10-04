"""The one place a request's scope becomes a store predicate.

Ported from ``rag-api/app/scope.py`` with the same semantics:

* Reads and deletes widen the owner set: ``{token user id or "public"} | {entity_id}``.
* Writes stamp a single owner: ``entity_id`` if given, else the token user id
  (see :func:`resolve_write_owner`).
* The owner predicate always goes into the store query *before* ranking.

``entity_id`` is caller-asserted. With JWT auth on, ``ENTITY_ID_POLICY`` decides
whether it must appear in the token's ``entities`` claim (default), is trusted
(rag-api behaviour, for LibreChat), or is ignored.
Without JWT auth there is no identity to protect.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request

from ragflow.settings import EntityIdPolicy, get_settings

logger = logging.getLogger(__name__)

PUBLIC_OWNER = "public"
OWNER_METADATA_KEY = "user_id"
ENTITIES_CLAIM = "entities"


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


def _authorized_entity(user: dict[str, Any], entity_id: str | None) -> str | None:
    """Apply ENTITY_ID_POLICY to a caller-supplied entity id."""
    if not entity_id:
        return None
    policy = get_settings().entity_id_policy
    if policy is EntityIdPolicy.TRUST:
        return entity_id
    if policy is EntityIdPolicy.DENY:
        return None
    allowed = user.get(ENTITIES_CLAIM) or []
    if isinstance(allowed, list) and entity_id in allowed:
        return entity_id
    logger.info("entity_id %r not in token %s claim", entity_id, ENTITIES_CLAIM)
    raise HTTPException(status_code=403, detail="entity_id not permitted for this token")


def resolve_scope(request: Request, entity_id: str | None = None) -> ScopeFilter:
    """Resolve the owners this request may read (own id, widened by entity_id)."""
    user = request_user(request)
    if user is None:
        return ScopeFilter(owners=(entity_id or PUBLIC_OWNER,))

    owners = {str(user.get("id") or PUBLIC_OWNER)}
    entity = _authorized_entity(user, entity_id)
    if entity:
        owners.add(entity)
    return ScopeFilter(owners=tuple(sorted(owners)))


def resolve_write_owner(request: Request, entity_id: str | None = None) -> str:
    """The owner stamped on newly embedded chunks (rag-api ``get_user_id``)."""
    user = request_user(request)
    if user is None:
        return entity_id or PUBLIC_OWNER
    return _authorized_entity(user, entity_id) or str(user.get("id") or PUBLIC_OWNER)

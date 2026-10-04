"""JWT bearer authentication, ported from ``rag-api/app/middleware.py``.

Behaviour kept: HS256, ``request.state.user = payload``, open access with a warning
when ``JWT_SECRET`` is unset, and the same 401 bodies. Changed: CORS preflight
(``OPTIONS``) is exempt so browsers can reach the API when a secret is set.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

import jwt
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from ragflow.settings import get_settings

logger = logging.getLogger(__name__)

EXEMPT_PATHS = frozenset({"/docs", "/openapi.json", "/health"})

CallNext = Callable[[Request], Awaitable[Response]]


def _is_exempt(request: Request) -> bool:
    path = request.url.path
    if request.method == "OPTIONS" or path in EXEMPT_PATHS:
        return True
    # The bundled frontend's static files are public; its API calls are not.
    return get_settings().serve_frontend and (path in ("/", "/design") or path.startswith("/ui"))


async def security_middleware(request: Request, call_next: CallNext) -> Response:
    if _is_exempt(request):
        return await call_next(request)

    secret = get_settings().jwt_secret
    if not secret:
        logger.warning("JWT_SECRET not set: requests are not authenticated")
        return await call_next(request)

    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        logger.info("Missing or invalid Authorization header: %s", request.url.path)
        return JSONResponse(
            status_code=401, content={"detail": "Missing or invalid Authorization header"}
        )

    token = authorization.split(" ", 1)[1]
    try:
        # PyJWT verifies ``exp`` itself and raises ExpiredSignatureError.
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return JSONResponse(status_code=401, content={"detail": "Token has expired"})
    except jwt.PyJWTError as exc:
        logger.info("Invalid token for %s: %s", request.url.path, exc)
        return JSONResponse(status_code=401, content={"detail": f"Invalid token: {exc}"})

    request.state.user = payload
    return await call_next(request)

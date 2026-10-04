"""Logging configuration (``DEBUG_RAG_API`` / ``CONSOLE_JSON``) and request logging."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from ragflow.settings import Settings

HTTP_REQ = "http_req"
HTTP_RES = "http_res"

logger = logging.getLogger("ragflow")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, object] = {
            "message": record.getMessage(),
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        for key in (HTTP_REQ, HTTP_RES):
            if key in record.__dict__:
                data[key] = record.__dict__[key]
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data)


def configure_logging(settings: Settings) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if settings.debug_rag_api else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter()
        if settings.console_json
        else logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root.handlers[:] = [handler]
    logging.getLogger("uvicorn.access").disabled = True


class LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        log = logger.debug if request.url.path == "/health" else logger.info
        log(
            "Request %s %s - %s",
            request.method,
            request.url,
            response.status_code,
            extra={
                HTTP_REQ: {"method": request.method, "url": str(request.url)},
                HTTP_RES: {"status_code": response.status_code},
            },
        )
        return response

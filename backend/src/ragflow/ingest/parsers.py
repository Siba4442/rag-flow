"""Optional PDF parsers (``PDF_PARSER``).

* ``pypdf`` (default): handled by the regular loader table; :func:`parse_pdf` returns None.
* ``llamaparse``: LlamaIndex's hosted LlamaParse via the ``llama-cloud`` SDK (extra
  ``llamaparse``; needs ``LLAMA_CLOUD_API_KEY``). Best for scanned PDFs, tables and
  complex layouts. This is the only place LlamaIndex adds something LangChain lacks;
  the rest of its framework (indices, query engines) would duplicate LangGraph.
* ``docling``: local, free layout-aware parsing (extra ``docling``).

Each parser yields one Document per page with ``source`` and ``page`` metadata so the
splitter and ``/documents/{id}/context`` treat them like PyPDF output.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from ragflow.settings import PdfParser, Settings

logger = logging.getLogger(__name__)


def parse_pdf(filepath: str, settings: Settings) -> list[Document] | None:
    match settings.pdf_parser:
        case PdfParser.PYPDF:
            return None
        case PdfParser.LLAMAPARSE:
            return _llamaparse(filepath, settings)
        case PdfParser.DOCLING:
            return _docling(filepath)


def _llamaparse(filepath: str, settings: Settings) -> list[Document]:
    from llama_cloud import LlamaCloud

    if not settings.llama_cloud_api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY is required when PDF_PARSER=llamaparse")
    client = LlamaCloud(api_key=settings.llama_cloud_api_key)
    with open(filepath, "rb") as fh:
        uploaded = client.files.create(file=fh, purpose="parse")
    result: Any = client.parsing.parse(
        file_id=uploaded.id, tier="agentic", version="latest", expand=["markdown"]
    )
    return [
        Document(page_content=page.markdown or "", metadata={"source": filepath, "page": i})
        for i, page in enumerate(result.markdown.pages)
    ]


def _docling(filepath: str) -> list[Document]:
    from langchain_docling import DoclingLoader
    from langchain_docling.loader import ExportType

    docs: list[Document] = DoclingLoader(
        file_path=filepath, export_type=ExportType.DOC_CHUNKS
    ).load()
    for doc in docs:
        meta = doc.metadata
        page = _docling_page(meta)
        doc.metadata = {"source": filepath, **({"page": page} if page is not None else {})}
    return docs


def _docling_page(meta: dict[str, Any]) -> int | None:
    try:
        return int(meta["dl_meta"]["doc_items"][0]["prov"][0]["page_no"]) - 1
    except (KeyError, IndexError, TypeError, ValueError):
        return None

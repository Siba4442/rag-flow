"""File loaders, ported from ``rag-api/app/utils/document_loader.py``.

The extension / MIME dispatch table, CSV encoding conversion, PDF image-extraction
fallback and ``process_documents`` context rendering keep rag-api's behaviour.
"""

from __future__ import annotations

import codecs
import contextlib
import logging
import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import chardet
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredRSTLoader,
    UnstructuredXMLLoader,
)
from langchain_core.documents import Document

from ragflow.settings import KNOWN_SOURCE_EXT

logger = logging.getLogger(__name__)

# Binary formats with dedicated loaders: a conflicting multipart Content-Type such as
# ``text/markdown`` must not hijack them into a text loader.
BINARY_FILE_EXTENSIONS = frozenset({"pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "epub"})

MARKDOWN_TYPES = frozenset(
    {"text/markdown", "text/x-markdown", "application/markdown", "application/x-markdown"}
)
XML_TYPES = frozenset({"application/xml", "text/xml", "application/xhtml+xml"})
PPT_TYPES = frozenset(
    {
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
)
DOC_TYPES = frozenset(
    {
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
)
XLS_TYPES = frozenset(
    {
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
)


class Loader(Protocol):
    def lazy_load(self) -> Iterator[Document]: ...


@dataclass
class LoadedFile:
    loader: Loader
    known_type: bool
    file_ext: str
    temp_paths: list[str] = field(default_factory=list)
    #: The caller's file when the loader reads a converted temp copy (e.g. re-encoded CSV).
    original_path: str | None = None

    def load(self) -> list[Document]:
        temps = set(self.temp_paths)  # cleanup() empties the list
        try:
            docs = list(self.loader.lazy_load())
        finally:
            self.cleanup()
        if self.original_path:
            # Never let an internal temp path become the cited source.
            for doc in docs:
                if doc.metadata.get("source") in temps:
                    doc.metadata["source"] = self.original_path
        return docs

    def cleanup(self) -> None:
        for path in self.temp_paths:
            try:
                os.remove(path)
            except OSError as exc:
                logger.warning("Failed to remove temporary file %s: %s", path, exc)
        self.temp_paths.clear()


def detect_file_encoding(filepath: str) -> str:
    """BOM markers first, then chardet; defaults to utf-8."""
    with open(filepath, "rb") as f:
        raw = f.read(4096)

    boms = (
        (codecs.BOM_UTF32_LE, "utf-32-le"),
        (codecs.BOM_UTF32_BE, "utf-32-be"),
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
        (codecs.BOM_UTF8, "utf-8-sig"),
    )
    for bom, name in boms:
        if raw.startswith(bom):
            return name
    encoding = chardet.detect(raw).get("encoding")
    return encoding.lower() if encoding else "utf-8"


def _csv_loader(filepath: str) -> LoadedFile:
    encoding = detect_file_encoding(filepath)
    if encoding in ("utf-8", "ascii"):
        return LoadedFile(CSVLoader(filepath), True, "csv")

    # Stream-convert to UTF-8 rather than holding the whole file as one string.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".csv", delete=False
    ) as tmp:
        try:
            with open(filepath, encoding=encoding, errors="replace") as src:
                while chunk := src.read(64 * 1024):
                    tmp.write(chunk)
        except Exception:
            tmp.close()
            os.unlink(tmp.name)
            raise
    return LoadedFile(CSVLoader(tmp.name), True, "csv", [tmp.name], original_path=filepath)


class SafePyPDFLoader:
    """PyPDFLoader that falls back to text-only when image extraction hits ``/Filter``.

    See https://github.com/langchain-ai/langchain/issues/26652.
    """

    def __init__(self, filepath: str, extract_images: bool = False) -> None:
        self.filepath = filepath
        self.extract_images = extract_images

    def lazy_load(self) -> Iterator[Document]:
        loader = PyPDFLoader(self.filepath, extract_images=self.extract_images)
        if not self.extract_images:
            yield from loader.lazy_load()
            return
        # Collect eagerly so a mid-stream failure can't duplicate already-yielded pages.
        try:
            pages = list(loader.lazy_load())
        except KeyError as exc:
            if "/Filter" not in str(exc):
                raise
            logger.warning("PDF image extraction failed for %s, text only: %s", self.filepath, exc)
            pages = list(PyPDFLoader(self.filepath, extract_images=False).lazy_load())
        yield from pages


def get_loader(
    filename: str,
    content_type: str | None,
    filepath: str,
    *,
    raw_text: bool = False,
    pdf_extract_images: bool = False,
) -> LoadedFile:
    """Pick a loader by extension and/or MIME type (same precedence as rag-api).

    ``raw_text`` loads text formats such as Markdown verbatim (for ``/text``); the
    embedding path keeps semantic loaders that strip formatting.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else filename.lower()
    ct = content_type or ""

    def text(known: bool = True) -> LoadedFile:
        return LoadedFile(TextLoader(filepath, autodetect_encoding=True), known, ext)

    if ext == "pdf" or ct == "application/pdf":
        return LoadedFile(SafePyPDFLoader(filepath, pdf_extract_images), True, ext)
    if ext == "csv" or ct == "text/csv":
        return _csv_loader(filepath)
    if ext == "rst":
        return LoadedFile(UnstructuredRSTLoader(filepath, mode="elements"), True, ext)
    if ext == "xml" or ct in XML_TYPES:
        return LoadedFile(UnstructuredXMLLoader(filepath), True, ext)
    if ext in ("ppt", "pptx") or ct in PPT_TYPES:
        return LoadedFile(UnstructuredPowerPointLoader(filepath), True, ext)
    if ext == "md" or (ct in MARKDOWN_TYPES and ext not in BINARY_FILE_EXTENSIONS):
        return text() if raw_text else LoadedFile(UnstructuredMarkdownLoader(filepath), True, ext)
    if ext == "epub" or ct == "application/epub+zip":
        return LoadedFile(UnstructuredEPubLoader(filepath), True, ext)
    if ext in ("doc", "docx") or ct in DOC_TYPES:
        return LoadedFile(Docx2txtLoader(filepath), True, ext)
    if ext in ("xls", "xlsx") or ct in XLS_TYPES:
        return LoadedFile(UnstructuredExcelLoader(filepath), True, ext)
    if ext == "json" or ct == "application/json":
        return text()
    if ext in KNOWN_SOURCE_EXT or "text/" in ct:
        return text()
    return text(known=False)


def clean_text(value: str) -> str:
    """Strip NUL bytes and invalid UTF-8 (surrogates) from PDF text."""
    value = value.replace("\x00", "")
    with contextlib.suppress(UnicodeError):
        value = value.encode("utf-8", "ignore").decode("utf-8")
    return value


def process_documents(documents: Sequence[Document], chunk_overlap: int) -> str:
    """Render ordered chunks as one text: basename, ``# PAGE n`` headers, de-overlapped."""
    basename = next(
        (str(d.metadata["source"]).split("/")[-1] for d in documents if "source" in d.metadata),
        "",
    )
    out = f"{basename}\n"
    last_page: object = None
    for doc in documents:
        page = doc.metadata.get("page")
        if page and page != last_page:
            out += f"\n# PAGE {page}\n\n"
            last_page = page
        content = doc.page_content
        if chunk_overlap and out.endswith(content[:chunk_overlap]):
            out += content[chunk_overlap:]
        else:
            out += content
    return out.strip()

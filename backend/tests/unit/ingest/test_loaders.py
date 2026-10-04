import os
from pathlib import Path

import pytest
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from ragflow.ingest.loaders import (
    SafePyPDFLoader,
    clean_text,
    detect_file_encoding,
    get_loader,
    process_documents,
)


@pytest.mark.parametrize(
    ("filename", "content_type", "loader_cls", "known"),
    [
        ("a.pdf", None, SafePyPDFLoader, True),
        ("blob", "application/pdf", SafePyPDFLoader, True),
        ("a.docx", None, Docx2txtLoader, True),
        # binary extension wins over a conflicting markdown content type
        ("a.docx", "text/markdown", Docx2txtLoader, True),
        ("a.md", None, UnstructuredMarkdownLoader, True),
        ("a.py", None, TextLoader, True),
        ("a.json", None, TextLoader, True),
        ("a.weird", "application/octet-stream", TextLoader, False),
    ],
)
def test_dispatch(
    tmp_path: Path, filename: str, content_type: str | None, loader_cls: type, known: bool
) -> None:
    path = tmp_path / "f"
    path.write_text("x")
    loaded = get_loader(filename, content_type, str(path))
    assert isinstance(loaded.loader, loader_cls)
    assert loaded.known_type is known


def test_markdown_raw_text_is_verbatim(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("# Title\n\n**bold**")
    docs = get_loader("a.md", None, str(path), raw_text=True).load()
    assert docs[0].page_content == "# Title\n\n**bold**"


def test_non_utf8_csv_is_converted_and_cleaned_up(tmp_path: Path) -> None:
    path = tmp_path / "latin.csv"
    path.write_bytes("name,city\nJosé,Zürich\nRenée,Genève\n".encode("utf-16"))
    assert detect_file_encoding(str(path)).startswith("utf-16")
    loaded = get_loader("latin.csv", "text/csv", str(path))
    assert isinstance(loaded.loader, CSVLoader)
    temp = loaded.temp_paths[0]
    docs = loaded.load()
    assert "Zürich" in docs[0].page_content
    assert not os.path.exists(temp)
    assert docs[0].metadata["source"] == str(path)  # not the temp copy


def test_clean_text_strips_nul_and_surrogates() -> None:
    assert clean_text("a\x00b\udc80c") == "abc"


def test_process_documents_pages_and_overlap() -> None:
    docs = [
        Document(page_content="hello world", metadata={"source": "/up/u/report.pdf", "page": 0}),
        Document(page_content="world again", metadata={"page": 1}),
    ]
    text = process_documents(docs, chunk_overlap=5)
    assert text == "report.pdf\nhello world\n# PAGE 1\n\nworld again"

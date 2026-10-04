from langchain_core.documents import Document

from ragflow.stores.pinecone import _unq_file, vector_id


def test_vector_id_encodes_scope_and_roundtrips_file_id() -> None:
    doc = Document(
        page_content="x",
        metadata={
            "user_id": "a#b",
            "file_id": "f/1#x",
            "ingestion_attempt_id": "t",
            "chunk_index": 3,
        },
    )
    vid = vector_id(doc)
    assert vid == "a%23b#f%2F1%23x#t#3"
    assert _unq_file(vid) == "f/1#x"

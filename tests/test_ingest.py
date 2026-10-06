from types import SimpleNamespace

import pytest


def test_ingestion_creates_index_and_upserts_stable_document_ids(monkeypatch, tmp_path):
    import langchain_openai
    import langchain_pinecone
    import pinecone

    import ingest

    source = tmp_path / "notes.md"
    source.write_text("# Checkpoints\nSQLite saves each research step.")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "portfolio-test")

    class IndexClient:
        def __init__(self):
            self.created = None

        def list_indexes(self):
            return SimpleNamespace(names=lambda: [] if self.created is None else ["portfolio-test"])

        def create_index(self, **options):
            self.created = options

        def describe_index(self, name):
            return SimpleNamespace(status={"ready": True}, dimension=1536)

    batches = []

    class VectorStore:
        @classmethod
        def from_documents(cls, documents, embedding, index_name, **options):
            batches.append((documents, options))
            return cls()

    client = IndexClient()
    monkeypatch.setattr(pinecone, "Pinecone", lambda **kwargs: client)
    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr(langchain_pinecone, "PineconeVectorStore", VectorStore)

    count = ingest.ingest_data(source)
    assert count == 1
    assert client.created["dimension"] == 1536
    assert batches[0][0][0].metadata["source"] == str(source)
    assert "SQLite saves" in batches[0][0][0].page_content
    assert batches[0][1]["namespace"] == "research-assistant"
    ingest.ingest_data(source)
    assert batches[0][1]["ids"] == batches[1][1]["ids"]


def test_empty_document_directory_is_rejected(monkeypatch, tmp_path):
    import ingest

    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    with pytest.raises(ValueError, match="No non-empty"):
        ingest.ingest_data(tmp_path)

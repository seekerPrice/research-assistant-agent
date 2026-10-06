"""Index UTF-8 text and Markdown documents in an optional Pinecone knowledge base."""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def ingest_data(path=Path("data")):
    """Create a compatible index if needed and upsert document chunks with stable IDs."""
    missing = [key for key in ("OPENAI_API_KEY", "PINECONE_API_KEY") if not os.getenv(key)]
    if missing:
        raise ValueError("Document ingestion requires " + " and ".join(missing) + ".")
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Document path does not exist: {path}")
    files = (
        [path]
        if path.is_file()
        else sorted(
            file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in (".txt", ".md")
        )
    )
    if path.is_file() and path.suffix.lower() not in (".txt", ".md"):
        raise ValueError("Supported document formats are .txt and .md.")
    texts = [(file, file.read_text(encoding="utf-8")) for file in files]
    texts = [(file, text) for file, text in texts if text.strip()]
    if not texts:
        raise ValueError("No non-empty .txt or .md documents found.")
    try:
        from langchain_core.documents import Document
        from langchain_openai import OpenAIEmbeddings
        from langchain_pinecone import PineconeVectorStore
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from pinecone import Pinecone, ServerlessSpec
    except ImportError as exc:
        raise RuntimeError('Install document support with pip install -e ".[rag]".') from exc

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks, ids = [], []
    for file, text in texts:
        documents = splitter.split_documents([Document(page_content=text, metadata={"source": str(file)})])
        for index, document in enumerate(documents):
            document.metadata["chunk"] = index
            chunks.append(document)
            ids.append(hashlib.sha256(f"{file}:{index}".encode()).hexdigest())
    client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.getenv("PINECONE_INDEX_NAME", "research-assistant")
    if index_name not in client.list_indexes().names():
        client.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"), region=os.getenv("PINECONE_REGION", "us-east-1")
            ),
        )
    deadline = time.monotonic() + 60
    description = client.describe_index(index_name)
    while not description.status["ready"]:
        if time.monotonic() >= deadline:
            raise TimeoutError("Pinecone index is not ready after 60 seconds. Retry ingestion later.")
        time.sleep(1)
        description = client.describe_index(index_name)
    if description.dimension != 1536:
        raise ValueError("This embedding model needs an index with 1536 dimensions; use a new index name.")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small", request_timeout=30),
        index_name=index_name,
        ids=ids,
        namespace=os.getenv("PINECONE_NAMESPACE", "research-assistant"),
    )
    return len(chunks)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Index text and Markdown documents in Pinecone")
    parser.add_argument("--path", type=Path, default=Path("data"), help="Document file or directory")
    args = parser.parse_args(argv)
    load_dotenv(Path.cwd() / ".env")
    try:
        count = ingest_data(args.path)
        print(f"Indexed {count} document chunks.")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

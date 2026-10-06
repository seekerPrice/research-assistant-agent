"""Read-only retrieval tools with explicit source links and bounded network requests."""

import os

USER_AGENT = "research-assistant-agent/0.2 (https://github.com/seekerPrice/research-assistant-agent)"


def search_web(query: str) -> str:
    """Search the web for current information; return titles, URLs, and snippets."""
    from ddgs import DDGS

    with DDGS(timeout=10) as client:
        results = list(client.text(query, max_results=5))
    if not results:
        raise RuntimeError("No web results returned.")
    return "\n\n".join(
        f"Title: {item.get('title', '')}\nURL: {item.get('href', '')}\nExcerpt: {item.get('body', '')[:2000]}"
        for item in results
    )


def search_wikipedia(query: str) -> str:
    """Search English Wikipedia for background; include canonical page URLs."""
    import requests

    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": 3,
            "prop": "extracts|info",
            "exintro": 1,
            "explaintext": 1,
            "inprop": "url",
            "format": "json",
            "formatversion": 2,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", [])
    if not pages:
        raise RuntimeError("No Wikipedia results returned.")
    return "\n\n".join(
        f"Title: {page['title']}\nURL: {page['fullurl']}\nExcerpt: {page.get('extract', '')[:3000]}"
        for page in sorted(pages, key=lambda item: item.get("index", 0))
    )


def search_arxiv(query: str) -> str:
    """Search arXiv paper titles and abstracts; return links without downloading PDFs."""
    import feedparser
    import requests

    response = requests.get(
        "https://export.arxiv.org/api/query",
        params={
            "search_query": query,
            "start": 0,
            "max_results": 3,
            "sortBy": "relevance",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    response.raise_for_status()
    entries = feedparser.parse(response.content).entries
    if not entries:
        raise RuntimeError("No arXiv results returned.")
    for entry in entries:
        if "/api/errors" in entry.get("id", ""):
            raise RuntimeError(f"arXiv query failed: {entry.get('summary', 'invalid query')}")
    return "\n\n".join(
        f"Title: {entry.title.strip()}\n"
        f"URL: {entry.id.replace('http://', 'https://')}\n"
        f"Excerpt: {entry.summary.strip()[:3000]}"
        for entry in entries
    )


def retrieve_documents(query: str) -> str:
    """Retrieve relevant text from the user's Pinecone-indexed local knowledge base."""
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        raise RuntimeError("Document retrieval requires OPENAI_API_KEY and PINECONE_API_KEY.")
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_pinecone import PineconeVectorStore
    except ImportError as exc:
        raise RuntimeError('Install document support with pip install -e ".[rag]".') from exc
    store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME", "research-assistant"),
        embedding=OpenAIEmbeddings(model="text-embedding-3-small", request_timeout=30),
        namespace=os.getenv("PINECONE_NAMESPACE", "research-assistant"),
    )
    documents = store.similarity_search(query, k=3)
    if not documents:
        raise RuntimeError("No relevant local documents found; run the ingestion command first.")
    return "\n\n".join(
        f"Local source: {doc.metadata.get('source', 'unknown')}\nExcerpt: {doc.page_content}"
        for doc in documents
    )


tools = [search_web, search_wikipedia, search_arxiv, retrieve_documents]

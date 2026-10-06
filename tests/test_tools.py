import pytest


def test_web_search_keeps_clickable_source_urls(monkeypatch):
    import ddgs

    import tools

    class SearchClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def text(self, *args, **kwargs):
            return [
                {
                    "title": "Checkpoints",
                    "href": "https://example.org/checkpoints",
                    "body": "External snippet",
                }
            ]

    monkeypatch.setattr(ddgs, "DDGS", lambda **kwargs: SearchClient())
    result = tools.search_web("LangGraph checkpoints")
    assert "https://example.org/checkpoints" in result
    assert "External snippet" in result


def test_wikipedia_search_keeps_clickable_source_urls(monkeypatch):
    import requests

    import tools

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "query": {
                    "pages": [
                        {
                            "title": "LangGraph",
                            "index": 1,
                            "fullurl": "https://en.wikipedia.org/wiki/LangGraph",
                            "extract": "A workflow library.",
                        }
                    ]
                }
            }

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    result = tools.search_wikipedia("LangGraph")
    assert "https://en.wikipedia.org/wiki/LangGraph" in result
    assert "A workflow library" in result


def test_empty_web_search_is_a_failure_not_evidence(monkeypatch):
    import ddgs

    import tools

    class SearchClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def text(self, *args, **kwargs):
            return []

    monkeypatch.setattr(ddgs, "DDGS", lambda **kwargs: SearchClient())
    with pytest.raises(RuntimeError, match="No web results"):
        tools.search_web("Nothing")


def test_arxiv_results_preserve_paper_links(monkeypatch):
    from types import SimpleNamespace

    import requests

    import tools

    feed = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/abs/2501.12345v1</id><title>Example paper</title>
      <summary>Evidence from an abstract.</summary></entry></feed>"""
    monkeypatch.setattr(
        requests, "get", lambda *args, **kwargs: SimpleNamespace(content=feed, raise_for_status=lambda: None)
    )
    result = tools.search_arxiv("Example")
    assert "https://arxiv.org/abs/2501.12345v1" in result
    assert "Evidence from an abstract" in result


def test_arxiv_error_feed_is_not_retrieved_evidence(monkeypatch):
    from types import SimpleNamespace

    import requests

    import tools

    feed = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/api/errors#incorrect_query</id><title>Error</title>
      <summary>Incorrect query syntax.</summary></entry></feed>"""
    monkeypatch.setattr(
        requests, "get", lambda *args, **kwargs: SimpleNamespace(content=feed, raise_for_status=lambda: None)
    )
    with pytest.raises(RuntimeError, match="arXiv query failed"):
        tools.search_arxiv("Bad query")


def test_document_retrieval_preserves_the_local_source(monkeypatch):
    import langchain_openai
    import langchain_pinecone
    from langchain_core.documents import Document

    import tools

    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")

    class Store:
        def similarity_search(self, query, k):
            return [Document(page_content="Persist every step.", metadata={"source": "/sample/notes.md"})]

    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr(langchain_pinecone, "PineconeVectorStore", lambda **kwargs: Store())
    result = tools.retrieve_documents("Persistence")
    assert "/sample/notes.md" in result
    assert "Persist every step" in result

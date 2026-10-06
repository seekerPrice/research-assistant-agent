import pytest
from langchain_core.messages import AIMessage, HumanMessage


def test_primary_failure_returns_the_configured_fallback_response(monkeypatch):
    import nodes

    class Unavailable:
        def invoke(self, messages):
            raise TimeoutError("Provider unavailable")

    class Available:
        def invoke(self, messages):
            return AIMessage(content="Fallback result")

    monkeypatch.setattr(nodes, "ChatOpenAI", lambda **kwargs: Unavailable())
    monkeypatch.setattr(nodes, "ChatGoogleGenerativeAI", lambda **kwargs: Available())
    result = nodes.invoke_llm_with_fallback([HumanMessage(content="Topic")])
    assert result.content == "Fallback result"


def test_google_only_configuration_can_make_a_request(monkeypatch):
    import nodes

    monkeypatch.delenv("OPENAI_API_KEY")

    class Available:
        def invoke(self, messages):
            return AIMessage(content="Google-only result")

    monkeypatch.setattr(nodes, "ChatGoogleGenerativeAI", lambda **kwargs: Available())
    assert nodes.invoke_llm_with_fallback([HumanMessage(content="Topic")]).content == "Google-only result"


def test_all_provider_failures_are_actionable(monkeypatch):
    import nodes

    class Unavailable:
        def invoke(self, messages):
            raise TimeoutError("Provider unavailable")

    monkeypatch.setattr(nodes, "ChatOpenAI", lambda **kwargs: Unavailable())
    monkeypatch.setattr(nodes, "ChatGoogleGenerativeAI", lambda **kwargs: Unavailable())
    with pytest.raises(RuntimeError, match="All configured models failed"):
        nodes.invoke_llm_with_fallback([HumanMessage(content="Topic")])

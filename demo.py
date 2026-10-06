"""Fixed illustrative evidence and model responses; never call a network service."""

from langchain_core.messages import AIMessage

DEMO_TOPIC = "How does this research assistant work?"
EVIDENCE = {
    "LangGraph workflow": (
        "https://docs.langchain.com/oss/python/langgraph/graph-api",
        "LangGraph models workflows with shared state, nodes, and conditional edges.",
    ),
    "Persistent research sessions": (
        "https://docs.langchain.com/oss/python/langgraph/persistence",
        "Checkpointers persist graph state for a thread, allowing recovery and continuity.",
    ),
    "Retrieval augmented generation": (
        "https://docs.langchain.com/oss/python/langchain/retrieval",
        "Retrieval supplies relevant external documents as context for generation.",
    ),
}


def search_demo(query: str) -> str:
    """Look up a fixed architecture excerpt; this is not live research."""
    url, excerpt = EVIDENCE[query]
    return f"Title: {query}\nURL: {url}\nExcerpt: {excerpt}"


def invoke_demo(messages, tools=None):
    """Supply deterministic responses through the live workflow's model interface."""
    prompt = messages[-1].content
    if tools:
        query = prompt.removeprefix("Query: ")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_demo",
                    "args": {"query": query},
                    "id": "demo-search",
                    "type": "tool_call",
                }
            ],
        )
    if prompt.startswith("You are a research planner"):
        return AIMessage(content="\n".join(EVIDENCE))
    selected = [(query, url, excerpt) for query, (url, excerpt) in EVIDENCE.items() if url in prompt]
    findings = "\n\n".join(f"- **{query}.** {excerpt}\n  [Source]({url})" for query, url, excerpt in selected)
    return AIMessage(
        content=f"""# Research Assistant Agent

> Offline demonstration using fixed illustrative excerpts. No live searches or model calls
> were made. This report demonstrates the workflow; it is not an answer to an arbitrary topic.

## Summary

The assistant plans a topic, retrieves evidence for each step, and produces a Markdown report.
The same LangGraph workflow runs in both demo and live modes.

## Key findings

{findings}

## Limitations

The demo uses three fixed excerpts to make the project easy to inspect without credentials.
It does not exercise paid providers, live search availability, or Pinecone integration.
Live reports still require human review of the cited evidence.
"""
    )

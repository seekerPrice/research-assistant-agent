# Research Assistant Agent

> Offline demonstration using fixed illustrative excerpts. No live searches or model calls
> were made. This report demonstrates the workflow; it is not an answer to an arbitrary topic.

## Summary

The assistant plans a topic, retrieves evidence for each step, and produces a Markdown report.
The same LangGraph workflow runs in both demo and live modes.

## Key findings

- **LangGraph workflow.** LangGraph models workflows with shared state, nodes, and conditional edges.
  [Source](https://docs.langchain.com/oss/python/langgraph/graph-api)

- **Persistent research sessions.** Checkpointers persist graph state for a thread, allowing recovery and continuity.
  [Source](https://docs.langchain.com/oss/python/langgraph/persistence)

- **Retrieval augmented generation.** Retrieval supplies relevant external documents as context for generation.
  [Source](https://docs.langchain.com/oss/python/langchain/retrieval)

## Limitations

The demo uses three fixed excerpts to make the project easy to inspect without credentials.
It does not exercise paid providers, live search availability, or Pinecone integration.
Live reports still require human review of the cited evidence.

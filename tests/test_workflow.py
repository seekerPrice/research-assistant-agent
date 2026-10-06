from langchain_core.messages import AIMessage


def test_all_research_steps_reach_the_final_report(monkeypatch):
    """Replacing findings instead of accumulating them loses two evidence sources."""
    import nodes
    from graph import create_graph

    def search_web(query: str) -> str:
        """Return deterministic evidence instead of making a network request."""
        return f"Evidence for {query}: https://example.org/{query.lower()}"

    def respond(messages, tools=None):
        prompt = messages[-1].content
        if tools:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_web",
                        "args": {"query": prompt.split("Query: ")[-1]},
                        "id": "search-1",
                        "type": "tool_call",
                    }
                ],
            )
        if "Synthesize" in prompt:
            return AIMessage(content=prompt)
        return AIMessage(content="First\nSecond\nThird")

    monkeypatch.setattr(nodes, "tools", [search_web])
    monkeypatch.setattr(nodes, "invoke_llm_with_fallback", respond)
    app = create_graph()
    result = app.invoke({"topic": "Evidence aggregation"}, {"configurable": {"thread_id": "aggregation"}})

    assert len(result["findings"]) == 3
    for query in ("First", "Second", "Third"):
        assert f"Evidence for {query}" in result["messages"][-1]


def test_a_new_topic_does_not_reuse_old_findings(monkeypatch):
    import nodes
    from graph import create_graph

    def search_web(query: str) -> str:
        """Produce topic-specific evidence."""
        return f"Evidence: {query}"

    def respond(messages, tools=None):
        prompt = messages[-1].content
        if tools:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_web",
                        "args": {"query": prompt},
                        "id": "search-1",
                        "type": "tool_call",
                    }
                ],
            )
        if prompt.startswith("You are a research planner"):
            return AIMessage(content="NewEvidence" if "NewTopic" in prompt else "OldEvidence")
        return AIMessage(content=prompt)

    monkeypatch.setattr(nodes, "tools", [search_web])
    monkeypatch.setattr(nodes, "invoke_llm_with_fallback", respond)
    app = create_graph()
    config = {"configurable": {"thread_id": "two-topics"}}
    app.invoke({"topic": "OldTopic"}, config)
    result = app.invoke({"topic": "NewTopic"}, config)

    assert len(result["findings"]) == 1
    assert "NewEvidence" in result["findings"][0]
    assert "OldEvidence" not in result["findings"][0]


def test_failed_search_is_not_treated_as_evidence(monkeypatch):
    import nodes

    def search_web(query: str) -> str:
        """Simulate an unavailable external search service."""
        raise TimeoutError("Search unavailable")

    monkeypatch.setattr(nodes, "tools", [search_web])
    monkeypatch.setattr(
        nodes,
        "invoke_llm_with_fallback",
        lambda *args, **kwargs: AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_web",
                    "args": {"query": "Topic"},
                    "id": "search-1",
                    "type": "tool_call",
                }
            ],
        ),
    )
    result = nodes.researcher_node({"topic": "Topic", "plan": ["Topic"], "current_step": 0, "findings": []})
    assert result["findings"] == []
    assert "Search unavailable" in result["errors"][0]


def test_unretrieved_model_knowledge_is_not_evidence(monkeypatch):
    import nodes

    monkeypatch.setattr(
        nodes,
        "invoke_llm_with_fallback",
        lambda *args, **kwargs: AIMessage(content="A plausible but unsupported claim."),
    )
    result = nodes.researcher_node({"topic": "Topic", "plan": ["Topic"], "current_step": 0, "findings": []})
    assert result["findings"] == []
    assert result["errors"]


def test_plan_is_bounded_even_if_model_returns_too_many_queries(monkeypatch):
    import nodes

    monkeypatch.setattr(
        nodes,
        "invoke_llm_with_fallback",
        lambda *args, **kwargs: AIMessage(content="\n".join(f"Query {i}" for i in range(20))),
    )
    result = nodes.planner_node({"topic": "Topic", "max_steps": 3})
    assert result["plan"] == ["Query 0", "Query 1", "Query 2"]


def test_empty_evidence_produces_an_honest_report(monkeypatch):
    import nodes

    monkeypatch.setattr(
        nodes, "invoke_llm_with_fallback", lambda *args, **kwargs: AIMessage(content="An unsupported answer.")
    )
    result = nodes.responder_node({"topic": "Topic", "findings": [], "errors": ["Search unavailable"]})
    assert "No usable evidence" in result["messages"][-1]
    assert "Search unavailable" in result["messages"][-1]


def test_planner_accepts_provider_text_blocks(monkeypatch):
    import nodes

    monkeypatch.setattr(
        nodes,
        "invoke_llm_with_fallback",
        lambda *args, **kwargs: AIMessage(content=[{"type": "text", "text": "First\nSecond"}]),
    )
    assert nodes.planner_node({"topic": "Topic"})["plan"] == ["First", "Second"]


def test_responder_normalizes_provider_text_blocks(monkeypatch):
    import nodes

    monkeypatch.setattr(
        nodes,
        "invoke_llm_with_fallback",
        lambda *args, **kwargs: AIMessage(content=[{"type": "text", "text": "# Report\nEvidence summary"}]),
    )
    result = nodes.responder_node({"topic": "Topic", "findings": ["Retrieved evidence"]})
    assert result["answer"] == "# Report\nEvidence summary"


def test_interrupted_session_resumes_at_synthesis(monkeypatch, tmp_path):
    import pytest
    from langgraph.checkpoint.sqlite import SqliteSaver

    import demo
    from graph import create_graph

    original = demo.invoke_demo

    def interrupted(messages, tools=None):
        if messages[-1].content.startswith("Synthesize"):
            raise TimeoutError("Provider temporarily unavailable")
        return original(messages, tools=tools)

    db = str(tmp_path / "resume.sqlite")
    config = {"configurable": {"thread_id": "interrupted"}}
    monkeypatch.setattr(demo, "invoke_demo", interrupted)
    with SqliteSaver.from_conn_string(db) as saver:
        app = create_graph(checkpointer=saver, demo=True)
        with pytest.raises(TimeoutError):
            app.invoke({"topic": "Architecture", "demo": True}, config)
        assert app.get_state(config).next == ("responder",)
    monkeypatch.setattr(demo, "invoke_demo", original)
    with SqliteSaver.from_conn_string(db) as saver:
        app = create_graph(checkpointer=saver, demo=True)
        result = app.invoke(None, config)
        assert len(result["findings"]) == 3
        assert "Offline demonstration" in result["answer"]

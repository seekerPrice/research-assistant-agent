"""The same three-node LangGraph workflow powers live research and the offline demo."""

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from nodes import planner_node, researcher_node, responder_node
from state import AgentState


def create_graph(checkpointer=None, *, demo=False):
    """Compile a graph. Pass a SQLite saver for durable, caller-owned persistence."""
    workflow = StateGraph(AgentState)
    if demo:
        from demo import invoke_demo, search_demo

        workflow.add_node("planner", partial(planner_node, llm=invoke_demo))
        workflow.add_node(
            "researcher", partial(researcher_node, llm=invoke_demo, research_tools=[search_demo])
        )
        workflow.add_node("responder", partial(responder_node, llm=invoke_demo))
    else:
        workflow.add_node("planner", planner_node)
        workflow.add_node("researcher", researcher_node)
        workflow.add_node("responder", responder_node)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_conditional_edges(
        "researcher",
        lambda state: "researcher" if state["current_step"] < len(state["plan"]) else "responder",
        ["researcher", "responder"],
    )
    workflow.add_edge("responder", END)
    return workflow.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())

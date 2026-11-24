"""The LangGraph agent graph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from repomind.app.agent import nodes
from repomind.app.agent.state import AgentState


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("decompose", nodes.decompose)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("synthesize", nodes.synthesize)
    g.add_node("verify_citations", nodes.verify_citations)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "decompose")
    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", "verify_citations")
    g.add_edge("verify_citations", "finalize")
    g.add_edge("finalize", END)
    return g

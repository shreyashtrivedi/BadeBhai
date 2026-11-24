r"""The LangGraph agent graph.

load_memory -> decompose -> retrieve -> grade_evidence --(weak, retries left)--> rewrite_queries -> retrieve
                                                     \--(done)--> synthesize -> verify_citations -> write_memory
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from repomind.app.agent import nodes
from repomind.app.agent.state import AgentState


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("load_memory", nodes.load_memory)
    g.add_node("decompose", nodes.decompose)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade_evidence", nodes.grade_evidence)
    g.add_node("rewrite_queries", nodes.rewrite_queries)
    g.add_node("synthesize", nodes.synthesize)
    g.add_node("verify_citations", nodes.verify_citations)
    g.add_node("write_memory", nodes.write_memory)

    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "decompose")
    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "grade_evidence")
    g.add_conditional_edges(
        "grade_evidence",
        nodes.route_after_grade,
        {"rewrite_queries": "rewrite_queries", "synthesize": "synthesize"},
    )
    g.add_conditional_edges(
        "rewrite_queries", nodes.route_after_rewrite, {"retrieve": "retrieve", "synthesize": "synthesize"}
    )
    g.add_edge("synthesize", "verify_citations")
    g.add_edge("verify_citations", "write_memory")
    g.add_edge("write_memory", END)
    return g

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from rag_app.config import AppConfig
from rag_app.copilot import run_financial_copilot_with_mode


StateDict = dict[str, Any]


def run_query_pipeline(config: AppConfig, question: str, top_k: int) -> dict[str, Any]:
    return run_financial_copilot_with_mode(config, question, top_k, "legacy")

def build_rag_graph(config: AppConfig):
    def copilot_node(state: StateDict) -> StateDict:
        question = state["question"]
        top_k = int(state.get("top_k", config.retrieval_k))
        result = run_financial_copilot_with_mode(config, question, top_k, "legacy")
        next_state = dict(state)
        next_state.update(result)
        return next_state

    graph = StateGraph(dict)
    graph.add_node("run_copilot", copilot_node)
    graph.add_edge(START, "run_copilot")
    graph.add_edge("run_copilot", END)
    return graph.compile()

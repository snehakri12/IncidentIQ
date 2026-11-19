"""LangGraph workflow wiring for an incident investigation.

The graph is a linear five-step pipeline:

    classify -> retrieve_history -> gather_external -> analyze -> recommend

``sources_checked`` uses an additive reducer so every node can report which
evidence sources it consulted.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.models.schema import FailureType


@dataclass
class InvestigationRequest:
    """A failure report submitted for investigation."""

    ticket_id: str
    job_name: str
    description: str
    log_excerpt: str
    script_name: Optional[str] = None


class InvestigationState(TypedDict, total=False):
    request: InvestigationRequest
    failure_type: FailureType
    historical_incidents: list[dict]
    github_file: Optional[dict]
    jira_issues: list[dict]
    sources_checked: Annotated[list[str], operator.add]
    analysis: Optional[str]
    recommendations: list[str]


def build_investigation_graph(service: Any):
    """Compile the investigation workflow around an ``InvestigationService``."""
    graph = StateGraph(InvestigationState)
    graph.add_node("classify", service.classify_node)
    graph.add_node("retrieve_history", service.retrieve_history_node)
    graph.add_node("gather_external", service.external_evidence_node)
    graph.add_node("analyze", service.analyze_node)
    graph.add_node("recommend", service.recommend_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve_history")
    graph.add_edge("retrieve_history", "gather_external")
    graph.add_edge("gather_external", "analyze")
    graph.add_edge("analyze", "recommend")
    graph.add_edge("recommend", END)
    return graph.compile()


async def run_investigation(service: Any, request: InvestigationRequest) -> dict:
    """Run the full workflow and shape the public result payload."""
    state: InvestigationState = await service.graph.ainvoke(
        {"request": request, "sources_checked": []}
    )
    return {
        "ticket_id": request.ticket_id,
        "job_name": request.job_name,
        "failure_type": state["failure_type"],
        "analysis": state.get("analysis"),
        "recommendations": state.get("recommendations", []),
        "evidence": {
            "historical_incidents": state.get("historical_incidents", []),
            "github_file": state.get("github_file"),
            "jira_issues": state.get("jira_issues", []),
        },
        "sources_checked": state.get("sources_checked", []),
    }

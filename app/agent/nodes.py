"""Investigation nodes: classification, retrieval, external evidence, analysis.

``InvestigationService`` owns the long-lived dependencies (retriever, GitHub
and Jira clients, compiled LangGraph) and exposes one async method per graph
node.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.agent.graph import (
    InvestigationRequest,
    InvestigationState,
    build_investigation_graph,
)
from app.agent.tools import GitHubClient, JiraClient
from app.models.schema import FailureType
from app.rag.pipeline import IncidentRetriever

logger = logging.getLogger(__name__)

_CLASSIFICATION_RULES: list[tuple[FailureType, tuple[str, ...]]] = [
    (
        FailureType.PERMISSION_ERROR,
        ("permission denied", "access denied", "forbidden", "unauthorized", "not authorized", "403"),
    ),
    (
        FailureType.RESOURCE_EXHAUSTION,
        (
            "out of memory",
            "oom",
            "memory limit",
            "memoryerror",
            "disk full",
            "no space left",
            "quota exceeded",
            "resource exhausted",
            "too many open files",
        ),
    ),
    (
        FailureType.TIMEOUT,
        ("timed out", "timeout", "deadline exceeded"),
    ),
    (
        FailureType.DEPENDENCY_FAILURE,
        (
            "connection refused",
            "connection reset",
            "service unavailable",
            "bad gateway",
            "name resolution",
            "dns",
            "502",
            "503",
        ),
    ),
    (
        FailureType.CONFIGURATION_ERROR,
        ("missing environment variable", "misconfigured", "invalid credentials", "not configured"),
    ),
    (
        FailureType.DATA_ERROR,
        ("schema mismatch", "parse error", "invalid json", "constraint violation", "unexpected null"),
    ),
]

_PLAYBOOK: dict[FailureType, list[str]] = {
    FailureType.PERMISSION_ERROR: [
        "Verify the job's service account and its role bindings against the resource it touched.",
        "Check for recently rotated or expired credentials and secrets.",
    ],
    FailureType.RESOURCE_EXHAUSTION: [
        "Compare the job's memory/disk usage against its configured limits and recent input sizes.",
        "Process large inputs in streaming batches or raise the resource limit with capacity sign-off.",
    ],
    FailureType.TIMEOUT: [
        "Check latency dashboards for the downstream dependency during the failure window.",
        "Tune the client timeout and add retry with exponential backoff if the call is idempotent.",
    ],
    FailureType.DEPENDENCY_FAILURE: [
        "Confirm the health and recent deploys of the upstream service the job depends on.",
        "Add a circuit breaker or fallback so one dependency outage does not fail the whole job.",
    ],
    FailureType.CONFIGURATION_ERROR: [
        "Diff the job's runtime configuration against the last known-good deploy.",
        "Validate required environment variables at startup so misconfiguration fails fast.",
    ],
    FailureType.DATA_ERROR: [
        "Inspect the first failing record and validate the upstream schema contract.",
        "Add input validation and quarantine malformed records instead of failing the batch.",
    ],
    FailureType.RUNTIME_ERROR: [
        "Reproduce with the captured input and inspect the full stack trace around the failure.",
        "Check recent code or dependency changes to the job for regressions.",
    ],
}


def classify_failure(text: str) -> FailureType:
    """Rule-based first-pass classification of a failure description/log."""
    lowered = text.lower()
    for failure_type, keywords in _CLASSIFICATION_RULES:
        if any(keyword in lowered for keyword in keywords):
            return failure_type
    return FailureType.RUNTIME_ERROR


def _format_evidence(
    historical_incidents: list[dict],
    github_file: Optional[dict],
    jira_issues: list[dict],
) -> str:
    sections = []
    if historical_incidents:
        lines = [
            f"- [{incident.get('id')}] {incident.get('summary')} | resolution: {incident.get('resolution')}"
            for incident in historical_incidents
        ]
        sections.append("Similar historical incidents:\n" + "\n".join(lines))
    else:
        sections.append("Similar historical incidents: none found.")
    if jira_issues:
        lines = [
            f"- {issue.get('key')}: {issue.get('summary')} (status: {issue.get('status')})"
            for issue in jira_issues
        ]
        sections.append("Related Jira issues:\n" + "\n".join(lines))
    else:
        sections.append("Related Jira issues: none found.")
    if github_file:
        snippet = str(github_file.get("content", ""))[:2_000]
        sections.append(
            f"Repository file {github_file.get('path')}:\n```\n{snippet}\n```"
        )
    else:
        sections.append("Repository evidence: none fetched.")
    return "\n\n".join(sections)


class InvestigationService:
    """Owns retriever, integration clients, and the compiled LangGraph."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self.retriever = IncidentRetriever(config)
        self.github = GitHubClient(config.get("github", {}))
        self.jira = JiraClient(config.get("jira", {}))
        self.graph = build_investigation_graph(self)

    # ------------------------------------------------------------- graph nodes
    async def classify_node(self, state: InvestigationState) -> dict:
        request = state["request"]
        failure_type = classify_failure(f"{request.description}\n{request.log_excerpt}")
        return {"failure_type": failure_type}

    async def retrieve_history_node(self, state: InvestigationState) -> dict:
        request = state["request"]
        query = f"{request.job_name} {request.description} {request.log_excerpt}"
        incidents, source = self.retriever.search(query)
        return {"historical_incidents": incidents, "sources_checked": [source]}

    async def external_evidence_node(self, state: InvestigationState) -> dict:
        request = state["request"]
        file_path = request.script_name or self._config.get("github", {}).get("default_path")
        failure_terms = state["failure_type"].value.replace("_", " ")

        github_file, jira_issues = await asyncio.gather(
            self.github.fetch_file(file_path),
            self.jira.search(f"{request.job_name} {failure_terms}"),
            return_exceptions=True,
        )
        if isinstance(github_file, BaseException):
            logger.warning("GitHub evidence fetch failed: %s", github_file)
            github_file = None
        if isinstance(jira_issues, BaseException):
            logger.warning("Jira search failed: %s", jira_issues)
            jira_issues = []

        sources = []
        if self.github.enabled:
            sources.append(f"github repository evidence ({self.github.mode})")
        if self.jira.enabled:
            sources.append(f"jira incident history ({self.jira.mode})")
        return {
            "github_file": github_file,
            "jira_issues": jira_issues,
            "sources_checked": sources,
        }

    async def analyze_node(self, state: InvestigationState) -> dict:
        analysis = await self._optional_llm_analysis(
            state["request"],
            state["failure_type"],
            state.get("historical_incidents", []),
            state.get("github_file"),
            state.get("jira_issues", []),
        )
        sources = ["langchain root-cause analysis"] if analysis else []
        return {"analysis": analysis, "sources_checked": sources}

    async def recommend_node(self, state: InvestigationState) -> dict:
        recommendations = list(_PLAYBOOK.get(state["failure_type"], []))
        for incident in state.get("historical_incidents", [])[:2]:
            resolution = incident.get("resolution")
            if resolution:
                recommendations.append(
                    f"Previously resolved in {incident.get('id')}: {resolution}"
                )
        return {"recommendations": recommendations}

    # ------------------------------------------------------------ LLM analysis
    async def _optional_llm_analysis(
        self,
        request: InvestigationRequest,
        failure_type: FailureType,
        historical_incidents: list[dict],
        github_file: Optional[dict],
        jira_issues: list[dict],
    ) -> Optional[str]:
        """Generate an evidence-grounded diagnosis when an LLM is configured."""
        llm = self._config.get("llm", {})
        if not llm.get("enabled") or not os.getenv("OPENAI_API_KEY"):
            return None

        prompt = ChatPromptTemplate.from_messages(
            [("system", llm["system_prompt"]), ("human", llm["user_prompt"])]
        )
        model = ChatOpenAI(
            model=llm.get("model", "gpt-4o-mini"),
            temperature=llm.get("temperature", 0),
            timeout=llm.get("timeout_seconds", 30),
        )
        chain = prompt | model
        try:
            response = await chain.ainvoke(
                {
                    "ticket_id": request.ticket_id,
                    "job_name": request.job_name,
                    "description": request.description,
                    "log_excerpt": request.log_excerpt,
                    "failure_type": failure_type.value,
                    "evidence": _format_evidence(
                        historical_incidents, github_file, jira_issues
                    ),
                }
            )
        except Exception as error:
            logger.warning("LLM analysis unavailable: %s", error)
            return None
        return response.content

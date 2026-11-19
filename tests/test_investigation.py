import asyncio

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from fastapi.testclient import TestClient

from app.agent.graph import InvestigationRequest, run_investigation
from app.agent.nodes import InvestigationService, classify_failure
from app.agent.tools import MCPHTTPClient, JiraClient, load_config
from app.models.schema import FailureType


def test_classifies_permission_failure():
    assert classify_failure("permission denied") is FailureType.PERMISSION_ERROR


def test_local_investigation_uses_synthetic_evidence(monkeypatch):
    monkeypatch.setenv("GITHUB_MCP_URL", "")
    monkeypatch.setenv("JIRA_MCP_URL", "")
    config = load_config()
    service = InvestigationService(config)
    result = asyncio.run(run_investigation(service, InvestigationRequest("T-1", "worker", "large import", "out of memory")))
    assert result["failure_type"] is FailureType.RESOURCE_EXHAUSTION
    assert result["evidence"]["historical_incidents"]
    assert "local synthetic incident corpus" in result["sources_checked"]


def test_langchain_analysis_uses_configured_prompt_and_model(monkeypatch):
    config = load_config()
    config["llm"]["enabled"] = True
    service = InvestigationService(config)
    observed_prompts = []

    def fake_model(**_kwargs):
        return RunnableLambda(
            lambda prompt: (
                observed_prompts.append(prompt.to_string())
                or AIMessage(content="Diagnosis grounded in evidence.")
            )
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("app.agent.nodes.ChatOpenAI", fake_model)
    result = asyncio.run(
        service._optional_llm_analysis(
            InvestigationRequest("T-1", "worker", "test description", "test log"),
            FailureType.RUNTIME_ERROR,
            [],
            None,
            [],
        )
    )

    assert result == "Diagnosis grounded in evidence."
    assert "test description" in observed_prompts[0]
    assert "test log" in observed_prompts[0]


def test_mcp_arguments_are_limited_to_server_tool_schema():
    arguments = MCPHTTPClient._arguments_for_tool(
        {
            "properties": {"owner": {"type": "string"}, "path": {"type": "string"}},
            "required": ["owner", "path"],
        },
        {"owner": "team", "path": "app/main.py", "repo": "RootCauseAI"},
    )
    assert arguments == {"owner": "team", "path": "app/main.py"}


def test_mcp_argument_mapping_reports_missing_required_values():
    try:
        MCPHTTPClient._arguments_for_tool(
            {"properties": {"cloudId": {"type": "string"}}, "required": ["cloudId"]},
            {},
        )
    except RuntimeError as error:
        assert "cloudId" in str(error)
    else:
        raise AssertionError("Expected missing required MCP arguments to be reported")


def test_jira_mcp_result_normalization():
    assert JiraClient._normalize_mcp_issues(
        {"issues": [{"key": "OPS-1", "fields": {"summary": "Job failed", "status": {"name": "Open"}}}]}
    ) == [{"key": "OPS-1", "summary": "Job failed", "status": "Open"}]


def test_mcp_endpoint_uses_configured_read_tools(monkeypatch):
    monkeypatch.setenv("GITHUB_MCP_URL", "https://github-mcp.example/mcp")
    monkeypatch.setenv("JIRA_MCP_URL", "https://jira-mcp.example/mcp")
    config = load_config()
    github = config["github"]
    jira_config = config["jira"]
    from app.agent.tools import GitHubClient

    github_client = GitHubClient(github)
    jira_client = JiraClient(jira_config)
    calls = []

    async def fake_call_tool(_client, tool_name, arguments):
        calls.append((tool_name, arguments))
        if tool_name == "get_file_contents":
            return {"path": "app/main.py", "content": "source"}
        return {"issues": [{"key": "OPS-1", "summary": "Prior incident", "status": "Open"}]}

    monkeypatch.setattr(MCPHTTPClient, "call_tool", fake_call_tool)
    github_file, issues = asyncio.run(
        _fetch_mcp_evidence(github_client, jira_client)
    )

    assert github_file == {"path": "app/main.py", "content": "source"}
    assert issues == [{"key": "OPS-1", "summary": "Prior incident", "status": "Open"}]
    assert [call[0] for call in calls] == ["get_file_contents", "search_issues"]


async def _fetch_mcp_evidence(github_client, jira_client):
    return await asyncio.gather(
        github_client.fetch_file("main.py"),
        jira_client.search("batch worker"),
    )


def test_investigation_api_contract(monkeypatch):
    monkeypatch.setenv("GITHUB_MCP_URL", "")
    monkeypatch.setenv("JIRA_MCP_URL", "")
    monkeypatch.setenv("JIRA_URL", "")
    monkeypatch.setenv("JIRA_EMAIL", "")
    monkeypatch.setenv("JIRA_API_TOKEN", "")
    from app.main import app
    with TestClient(app) as client:
        response = client.post("/api/v1/investigations", json={
            "ticket_id": "T-2", "job_name": "worker", "description": "large import",
            "log_excerpt": "out of memory",
        })
    assert response.status_code == 200
    assert response.json()["failure_type"] == "resource_exhaustion"


def test_readiness_and_basic_security_headers():
    from app.main import app
    with TestClient(app) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-request-id"]


def test_rejects_oversized_requests():
    from app.main import app
    with TestClient(app) as client:
        response = client.post("/api/v1/investigations", headers={"Content-Length": "50001"})
    assert response.status_code == 413


def test_edge_fixture_rejects_path_traversal():
    from edge_node_service.edge_agent import app
    response = app.test_client().get("/api/logs?name=../../config")
    assert response.status_code == 404
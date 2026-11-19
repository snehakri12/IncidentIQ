"""Evidence-gathering integrations: configuration, MCP client, GitHub and Jira clients.

GitHub and Jira are reached through MCP servers when ``GITHUB_MCP_URL`` /
``JIRA_MCP_URL`` are configured, and fall back to their REST APIs when only
tokens are available. Everything degrades gracefully to local-only mode so the
demo runs without any credentials.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"

_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in _TRUTHY


def load_config() -> dict:
    """Load YAML settings and overlay environment-provided secrets/endpoints."""
    with _CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    github = config.setdefault("github", {})
    github["mcp_url"] = os.getenv("GITHUB_MCP_URL", "")
    github["token"] = os.getenv("GITHUB_TOKEN", "")
    github["owner"] = os.getenv("GITHUB_OWNER", github.get("owner", ""))
    github["repo"] = os.getenv("GITHUB_REPO", github.get("repo", ""))

    jira = config.setdefault("jira", {})
    jira["mcp_url"] = os.getenv("JIRA_MCP_URL", "")
    jira["base_url"] = os.getenv("JIRA_URL", "")
    jira["email"] = os.getenv("JIRA_EMAIL", "")
    jira["api_token"] = os.getenv("JIRA_API_TOKEN", "")

    llm = config.setdefault("llm", {})
    llm["enabled"] = _env_flag("LLM_ENABLED", bool(llm.get("enabled", False)))

    retrieval = config.setdefault("retrieval", {})
    retrieval["use_chromadb"] = _env_flag(
        "RAG_USE_CHROMADB", bool(retrieval.get("use_chromadb", False))
    )

    return config


class MCPHTTPClient:
    """Minimal client for MCP servers speaking streamable HTTP.

    ``call_tool`` discovers the named tool's input schema from the server and
    maps the caller's available values onto it, so each deployment's tool
    schema drives what gets sent.
    """

    def __init__(self, url: str, auth_token: Optional[str] = None) -> None:
        self._url = url
        self._headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else None

    async def call_tool(self, tool_name: str, available_values: dict[str, Any]) -> Any:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        values = {key: value for key, value in available_values.items() if value is not None}
        async with streamablehttp_client(self._url, headers=self._headers) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                listing = await session.list_tools()
                tool = next((item for item in listing.tools if item.name == tool_name), None)
                if tool is None:
                    raise RuntimeError(
                        f"MCP server at {self._url} does not expose tool '{tool_name}'"
                    )
                arguments = self._arguments_for_tool(tool.inputSchema or {}, values)
                result = await session.call_tool(tool_name, arguments)
                return self._parse_result(result)

    @staticmethod
    def _arguments_for_tool(schema: dict, available_values: dict[str, Any]) -> dict[str, Any]:
        """Keep only values the tool's schema declares; fail loudly on gaps."""
        properties = (schema or {}).get("properties", {})
        arguments = {
            key: value for key, value in available_values.items() if key in properties
        }
        missing = [key for key in (schema or {}).get("required", []) if key not in arguments]
        if missing:
            raise RuntimeError(
                "Missing required MCP arguments: " + ", ".join(sorted(missing))
            )
        return arguments

    @staticmethod
    def _parse_result(result: Any) -> Any:
        if getattr(result, "isError", False):
            raise RuntimeError(f"MCP tool call failed: {result}")
        structured = getattr(result, "structuredContent", None)
        if structured:
            return structured
        for content in getattr(result, "content", []) or []:
            if getattr(content, "type", "") == "text":
                try:
                    return json.loads(content.text)
                except ValueError:
                    return {"text": content.text}
        return {}


class GitHubClient:
    """Fetch repository-level evidence via an MCP server or the GitHub REST API."""

    def __init__(self, config: dict) -> None:
        self._config = config
        mcp_url = config.get("mcp_url")
        self._mcp = MCPHTTPClient(mcp_url, config.get("token") or None) if mcp_url else None

    @property
    def enabled(self) -> bool:
        return bool(
            self._mcp
            or (self._config.get("token") and self._config.get("owner") and self._config.get("repo"))
        )

    @property
    def mode(self) -> str:
        if self._mcp:
            return "mcp"
        return "rest" if self.enabled else "disabled"

    async def fetch_file(self, path: Optional[str]) -> Optional[dict]:
        if not path or not self.enabled:
            return None
        if self._mcp:
            return await self._mcp.call_tool(
                self._config.get("mcp", {}).get("tool", "get_file_contents"),
                {
                    "owner": self._config.get("owner"),
                    "repo": self._config.get("repo"),
                    "path": path,
                    "ref": self._config.get("ref"),
                },
            )
        return await self._fetch_rest(path)

    async def _fetch_rest(self, path: str) -> Optional[dict]:
        url = (
            "https://api.github.com/repos/"
            f"{self._config['owner']}/{self._config['repo']}/contents/{path}"
        )
        headers = {
            "Accept": "application/vnd.github.raw+json",
            "Authorization": f"Bearer {self._config['token']}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        params = {"ref": self._config["ref"]} if self._config.get("ref") else None
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return {"path": path, "content": response.text[:8_000]}


class JiraClient:
    """Search historical Jira issues via an MCP server or the Jira Cloud REST API."""

    def __init__(self, config: dict) -> None:
        self._config = config
        mcp_url = config.get("mcp_url")
        self._mcp = MCPHTTPClient(mcp_url, config.get("api_token") or None) if mcp_url else None

    @property
    def enabled(self) -> bool:
        return bool(
            self._mcp
            or (
                self._config.get("base_url")
                and self._config.get("email")
                and self._config.get("api_token")
            )
        )

    @property
    def mode(self) -> str:
        if self._mcp:
            return "mcp"
        return "rest" if self.enabled else "disabled"

    async def search(self, query: str) -> list[dict]:
        if not self.enabled:
            return []
        jql = self._build_jql(query)
        if self._mcp:
            result = await self._mcp.call_tool(
                self._config.get("mcp", {}).get("tool", "search_issues"),
                {
                    "jql": jql,
                    "query": query,
                    "maxResults": self._config.get("max_results", 5),
                    "cloudId": self._config.get("cloud_id"),
                },
            )
            return self._normalize_mcp_issues(result)
        return await self._search_rest(jql)

    def _build_jql(self, query: str) -> str:
        sanitized = query.replace('"', " ").strip()
        project = self._config.get("project")
        clauses = [f'text ~ "{sanitized}"']
        if project:
            clauses.insert(0, f'project = "{project}"')
        return " AND ".join(clauses) + " ORDER BY created DESC"

    async def _search_rest(self, jql: str) -> list[dict]:
        url = self._config["base_url"].rstrip("/") + "/rest/api/3/search/jql"
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                url,
                params={
                    "jql": jql,
                    "maxResults": self._config.get("max_results", 5),
                    "fields": "summary,status",
                },
                auth=(self._config["email"], self._config["api_token"]),
            )
            response.raise_for_status()
            return self._normalize_mcp_issues(response.json())

    @staticmethod
    def _normalize_mcp_issues(result: Any) -> list[dict]:
        """Flatten Jira issue payloads (raw REST or MCP-shaped) to key/summary/status."""
        if isinstance(result, dict):
            issues = result.get("issues", [])
        elif isinstance(result, list):
            issues = result
        else:
            return []
        normalized = []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            fields = issue.get("fields", {}) or {}
            summary = fields.get("summary", issue.get("summary", ""))
            status = fields.get("status", issue.get("status", ""))
            if isinstance(status, dict):
                status = status.get("name", "")
            normalized.append(
                {"key": issue.get("key", ""), "summary": summary, "status": status}
            )
        return normalized

# IncidentIQ

AI-powered incident analysis system that classifies software/job failures, investigates them against historical incidents, and generates evidence-grounded root-cause recommendations through a structured LangGraph workflow.

Built with **Python, FastAPI, LangGraph, LangChain, ChromaDB, OpenAI embeddings, MCP, and the Jira/GitHub APIs**.

## How it works

Each investigation runs through a five-node LangGraph state machine:

```
POST /api/v1/investigations
        │
        ▼
┌───────────────┐   ┌────────────────────┐   ┌─────────────────────┐
│   classify     │ → │  retrieve_history   │ → │   gather_external    │
│ rule-based     │   │ RAG over historical │   │ Jira + GitHub fetched│
│ failure typing │   │ incidents (Chroma)  │   │ in parallel (asyncio)│
└───────────────┘   └────────────────────┘   └─────────────────────┘
                                                        │
        ┌────────────────────┐   ┌─────────────────┐    │
        │     recommend       │ ← │     analyze      │ ◄──┘
        │ playbook + prior    │   │ LangChain +      │
        │ incident resolutions│   │ ChatOpenAI (opt.)│
        └────────────────────┘   └─────────────────┘
```

1. **Classify** — a rule-based first pass types the failure (`resource_exhaustion`, `permission_error`, `timeout`, `dependency_failure`, `configuration_error`, `data_error`, `runtime_error`).
2. **Retrieve history (RAG)** — historical incidents are chunked, embedded with OpenAI `text-embedding-3-small`, and indexed in a persistent **ChromaDB** collection; the query is answered by semantic similarity search. Without an API key, a keyword retriever searches the same local synthetic corpus, so the service always runs.
3. **Gather external evidence** — **Jira** incident history and **GitHub** repository files are fetched concurrently with `asyncio.gather`, each through an **MCP** server when configured (tool schemas are discovered from the server and arguments mapped onto them) or through the REST API as a fallback.
4. **Analyze** — a **LangChain** prompt pipes all gathered evidence into **ChatOpenAI** for an evidence-grounded root-cause diagnosis (optional; off by default).
5. **Recommend** — actionable steps from a per-failure-type playbook, plus resolutions that fixed the most similar past incidents.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000 for the demo UI, or call the API directly:

```bash
curl -X POST http://localhost:8000/api/v1/investigations \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "DEMO-001",
    "job_name": "reconciliation-worker",
    "description": "Batch job exceeded its memory limit while processing a large import.",
    "log_excerpt": "ERROR: out of memory while processing records"
  }'
```

No credentials are required — the service runs fully locally against a synthetic incident corpus. To enable the full pipeline, copy `.env.example` to `.env` and set:

| Variable | Enables |
| --- | --- |
| `OPENAI_API_KEY` + `LLM_ENABLED=true` | LangChain + ChatOpenAI root-cause analysis |
| `OPENAI_API_KEY` + `RAG_USE_CHROMADB=true` | ChromaDB semantic retrieval with `text-embedding-3-small` |
| `GITHUB_MCP_URL` (or `GITHUB_TOKEN` + `GITHUB_OWNER` + `GITHUB_REPO`) | Repository evidence via MCP or the GitHub REST API |
| `JIRA_MCP_URL` (or `JIRA_URL` + `JIRA_EMAIL` + `JIRA_API_TOKEN`) | Historical issue search via MCP or the Jira REST API |

### Docker

```bash
docker compose up --build
```

## API

| Endpoint | Description |
| --- | --- |
| `POST /api/v1/investigations` | Run a full investigation; returns failure type, evidence, analysis, and recommendations |
| `GET /ready`, `GET /health` | Readiness/liveness probes |
| `GET /` | Minimal demo UI |

Example response (local mode):

```json
{
  "ticket_id": "DEMO-001",
  "job_name": "reconciliation-worker",
  "failure_type": "resource_exhaustion",
  "analysis": null,
  "recommendations": [
    "Compare the job's memory/disk usage against its configured limits and recent input sizes.",
    "Process large inputs in streaming batches or raise the resource limit with capacity sign-off.",
    "Previously resolved in INC-1001: Switched the import to streaming batches of 10k records and raised the container memory limit to 6GiB."
  ],
  "evidence": { "historical_incidents": ["..."], "github_file": null, "jira_issues": [] },
  "sources_checked": ["local synthetic incident corpus"]
}
```

All requests get `X-Request-ID` and security headers; bodies over 50 KB are rejected with 413.

## Project structure

```
app/
  main.py            FastAPI app, middleware, routes
  agent/
    graph.py         LangGraph state machine + run_investigation
    nodes.py         InvestigationService: graph nodes, classifier, LLM analysis
    tools.py         load_config, MCPHTTPClient, GitHubClient, JiraClient
  rag/pipeline.py    Chunking + ChromaDB/OpenAI-embeddings retriever with local fallback
  models/schema.py   FailureType enum and API schemas
config/settings.yaml Prompts, retrieval, and integration defaults
data/sample_incidents.json  Synthetic historical incident corpus
edge_node_service/   Simulated edge node serving job logs (Flask)
static/index.html    Demo UI
tests/               End-to-end contract tests
```

## Tests

```bash
pytest
```

The suite covers failure classification, the full local investigation flow, LangChain prompt/model wiring, MCP argument mapping against server tool schemas, Jira payload normalization, parallel MCP evidence retrieval, the API contract, security headers, request-size limits, and path-traversal rejection in the edge log service.

## Notes

- All incident data in this repository is synthetic and for demonstration only. Do not submit production secrets or personal data.
- The MCP client discovers each server's tool schema at call time and only sends arguments the schema declares, failing loudly when required arguments are missing.

"""FastAPI entrypoint for IncidentIQ."""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent.graph import InvestigationRequest, run_investigation
from app.agent.nodes import InvestigationService
from app.agent.tools import load_config
from app.models.schema import InvestigationIn, InvestigationOut

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("incidentiq")

MAX_BODY_BYTES = 50_000
_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.service = InvestigationService(load_config())
    logger.info("IncidentIQ ready")
    yield


app = FastAPI(
    title="IncidentIQ",
    description="AI-powered incident analysis: classification, RAG over historical incidents, and evidence-grounded recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def guardrails(request: Request, call_next):
    """Reject oversized bodies and stamp basic security/trace headers."""
    content_length = request.headers.get("content-length")
    too_large = False
    if content_length:
        try:
            too_large = int(content_length) > MAX_BODY_BYTES
        except ValueError:
            too_large = True
    if too_large:
        response = JSONResponse({"detail": "Request body too large"}, status_code=413)
    else:
        response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Request-ID"] = request.headers.get("x-request-id", str(uuid.uuid4()))
    return response


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict:
    return {"status": "ready"}


@app.post("/api/v1/investigations", response_model=InvestigationOut)
async def create_investigation(payload: InvestigationIn, request: Request):
    service: InvestigationService = request.app.state.service
    result = await run_investigation(
        service,
        InvestigationRequest(
            ticket_id=payload.ticket_id,
            job_name=payload.job_name,
            description=payload.description,
            log_excerpt=payload.log_excerpt,
            script_name=payload.script_name,
        ),
    )
    return result


if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")

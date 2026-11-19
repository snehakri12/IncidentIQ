"""Shared enums and API schemas."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FailureType(str, Enum):
    """Coarse classification of a software/job failure."""

    PERMISSION_ERROR = "permission_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT = "timeout"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION_ERROR = "configuration_error"
    DATA_ERROR = "data_error"
    RUNTIME_ERROR = "runtime_error"


class InvestigationIn(BaseModel):
    """Inbound payload for POST /api/v1/investigations."""

    ticket_id: str = Field(min_length=1, max_length=64)
    job_name: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=8_000)
    log_excerpt: str = Field(min_length=1, max_length=20_000)
    script_name: Optional[str] = Field(default=None, max_length=256)


class EvidenceOut(BaseModel):
    historical_incidents: list = Field(default_factory=list)
    github_file: Optional[dict] = None
    jira_issues: list = Field(default_factory=list)


class InvestigationOut(BaseModel):
    """Response contract for an investigation."""

    ticket_id: str
    job_name: str
    failure_type: FailureType
    analysis: Optional[str] = None
    recommendations: list[str] = Field(default_factory=list)
    evidence: EvidenceOut
    sources_checked: list[str] = Field(default_factory=list)

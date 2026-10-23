"""Wire-level values and stable error codes for the private media endpoint.

Plain dataclasses only. The transport will serialise them; no Pydantic model
sits in front of :func:`distribute_agent.plan.parse_plan`, because that parser
must see raw numbers, booleans and strings before any coercion. Diagnostic and
error dictionaries are deliberately bounded: counts and timings, never generated
code, provider bodies, credentials or local paths.
"""

from dataclasses import dataclass, field
from typing import Optional

# Stable prefixes: a client can branch on ``code`` without parsing prose.
INVALID_INPUT = "invalid_input"
PLAN_FAILED = "plan_failed"
GENERATION_DISABLED = "generation_disabled"
BUSY = "busy"
CAPACITY = "capacity"
STALE_DIGEST = "stale_digest"
REQUEST_CONFLICT = "request_conflict"
UNKNOWN_JOB = "unknown_job"
UNKNOWN_ARTIFACT = "unknown_artifact"
ARTIFACT_TOO_LARGE = "artifact_too_large"
ARTIFACT_CORRUPT = "artifact_corrupt"
STORAGE_ERROR = "storage_error"
STATE_LOCKED = "state_locked"
SHUTTING_DOWN = "shutting_down"
SNAPSHOT_UNAVAILABLE = "snapshot_unavailable"
SUBMISSION_UNCERTAIN = "submission_uncertain"
SUBMISSION_FAILED = "submission_failed"
INVALID_OUTPUT = "invalid_output"
INTERRUPTED = "interrupted"
RUNTIME_ERROR = "runtime_error"

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_NEEDS_ATTENTION = "needs_attention"
JOB_STATUSES = (
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    STATUS_FAILED,
    STATUS_NEEDS_ATTENTION,
)

PLAN_FIELD_ORDER = ("template", "prompt", "negative", "seed", "width", "height")


class ServiceError(RuntimeError):
    """A stable, redacted failure raised at the runtime boundary.

    ``code`` is one of the module constants. ``may_have_submitted`` is true only
    when a billable submission might exist, so a caller can look it up instead of
    resubmitting.
    """

    def __init__(self, code: str, message: str, *, may_have_submitted: bool = False):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.may_have_submitted = bool(may_have_submitted)

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "may_have_submitted": self.may_have_submitted,
        }


@dataclass(frozen=True)
class PlanResult:
    """One strict, validated plan plus the package it was planned against."""

    plan: dict
    package_digest: str
    model_id: str
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "plan": dict(self.plan),
            "package_digest": self.package_digest,
            "model_id": self.model_id,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class ArtifactMetadata:
    """Opaque artifact identity: no file name, path or vendor payload."""

    artifact_id: str
    job_id: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ArtifactContent:
    metadata: ArtifactMetadata
    data: bytes


@dataclass(frozen=True)
class JobView:
    """The persisted view returned by ``start_generation`` / ``get_generation``."""

    job_id: str
    request_id: str
    status: str
    plan: dict
    package_digest: str
    vendor_job_id: Optional[str]
    artifacts: tuple
    error: Optional[dict]
    created_at: str
    updated_at: str
    started_at: Optional[str]
    finished_at: Optional[str]

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            STATUS_SUCCEEDED,
            STATUS_FAILED,
            STATUS_NEEDS_ATTENTION,
        )

    def as_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "request_id": self.request_id,
            "status": self.status,
            "plan": dict(self.plan),
            "package_digest": self.package_digest,
            "vendor_job_id": self.vendor_job_id,
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
            "error": dict(self.error) if self.error else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def canonical_plan(plan) -> dict:
    """The validated plan with omitted fields left out, in declaration order.

    Identity is this dictionary plus the package digest. ``None`` and an omitted
    field mean the same thing to the runner, so they must mean the same thing to
    idempotency.
    """
    canonical = {}
    for name in PLAN_FIELD_ORDER:
        value = getattr(plan, name)
        if value is not None:
            canonical[name] = value
    return canonical


def error_record(code: str, message: str, *, may_have_submitted: bool = False) -> dict:
    return {
        "code": code,
        "message": message,
        "may_have_submitted": bool(may_have_submitted),
    }

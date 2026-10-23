"""FastMCP transport for the private media runtime.

The server is a loopback Streamable-HTTP endpoint with JWT auth on every MCP
request. The runtime's blocking API is called through ``asyncio.to_thread`` so a
plan or a SQLite read never stalls the event loop. Custom HTTP routes are added
with ``server.custom_route`` and are *not* covered by FastMCP's MCP auth
middleware, so the artifact download route verifies the same JWT verifier itself.
Only ``GET /health`` is public.

The runtime is started and stopped by the FastMCP lifespan. It pins its package
snapshot and takes the state lock on startup; stopping rejects new work and
drains the single worker within the configured bound.
"""

import asyncio
from typing import Optional, TypedDict

from fastmcp import FastMCP
from fastmcp.exceptions import ResourceError, ToolError
from fastmcp.resources import ResourceContent, ResourceResult
from fastmcp.server.lifespan import lifespan
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..artifacts import MAX_ARTIFACT_BYTES
from .types import (
    ARTIFACT_CORRUPT,
    ARTIFACT_TOO_LARGE,
    RUNTIME_ERROR,
    SHUTTING_DOWN,
    STORAGE_ERROR,
    UNKNOWN_ARTIFACT,
    UNKNOWN_JOB,
    ServiceError,
)
from .runtime import MediaRuntime

SERVICE_NAME = "mediagen"
MCP_PATH = "/mcp"
ARTIFACT_ROUTE = "/artifacts/{job_id}/{artifact_id}"
ARTIFACT_URI = "media://jobs/{job_id}/artifacts/{artifact_id}"
HEALTH_PATH = "/health"
BEARER_CHALLENGE = 'Bearer realm="mediagen"'

INSTRUCTIONS = (
    "Private media planning and generation endpoint. Authenticate with a bearer "
    "token that carries the media:access scope. Use plan_media to produce a "
    "validated plan, start_generation to submit it, and get_generation to poll "
    "the job. Artifact metadata is on the job; binary artifacts are available at "
    "the media:// resource for inline reads or GET /artifacts/{job_id}/"
    "{artifact_id} for files above the inline cap. Generation is disabled unless "
    "the operator enabled it and supplied the RunPod key."
)

_ARTIFACT_HTTP_STATUS = {
    UNKNOWN_JOB: 404,
    UNKNOWN_ARTIFACT: 404,
    ARTIFACT_TOO_LARGE: 413,
    ARTIFACT_CORRUPT: 410,
    SHUTTING_DOWN: 503,
    STORAGE_ERROR: 500,
    RUNTIME_ERROR: 500,
}


class PlanOut(TypedDict):
    plan: dict
    package_digest: str
    model_id: str
    diagnostics: dict


class ArtifactOut(TypedDict):
    artifact_id: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    sha256: str


class ErrorOut(TypedDict):
    code: str
    message: str
    may_have_submitted: bool


class JobOut(TypedDict):
    job_id: str
    request_id: str
    status: str
    plan: dict
    package_digest: str
    vendor_job_id: Optional[str]
    artifacts: list[ArtifactOut]
    error: Optional[ErrorOut]
    created_at: str
    updated_at: str
    started_at: Optional[str]
    finished_at: Optional[str]


def build_server(
    runtime: MediaRuntime,
    *,
    verifier,
) -> FastMCP:
    """Register the three tools, the artifact resource and the HTTP routes."""

    @lifespan
    async def runtime_lifespan(_server):
        await asyncio.to_thread(runtime.start)
        try:
            yield {"runtime": runtime}
        finally:
            released = await asyncio.to_thread(runtime.stop)
            if not released:
                raise RuntimeError(
                    "the media runtime did not release its state lock; active planner "
                    "or bridge work may remain and remote work was not cancelled"
                )

    server = FastMCP(
        SERVICE_NAME,
        instructions=INSTRUCTIONS,
        auth=verifier,
        lifespan=runtime_lifespan,
        mask_error_details=True,
    )

    @server.tool
    async def plan_media(request: str) -> PlanOut:
        """Plan one image from a natural-language brief (no generation)."""
        try:
            result = await asyncio.to_thread(runtime.plan, request)
        except ServiceError as exc:
            raise _tool_error(exc) from exc
        return result.as_dict()

    @server.tool
    async def start_generation(
        plan: dict, package_digest: str, request_id: str
    ) -> JobOut:
        """Validate and enqueue one paid generation; idempotent per request_id."""
        try:
            view = await asyncio.to_thread(
                runtime.start_generation, plan, package_digest, request_id
            )
        except ServiceError as exc:
            raise _tool_error(exc) from exc
        return view.as_dict()

    @server.tool
    async def get_generation(job_id: str) -> JobOut:
        """Read the persisted status, plan and artifact metadata for one job."""
        try:
            view = await asyncio.to_thread(runtime.get_generation, job_id)
        except ServiceError as exc:
            raise _tool_error(exc) from exc
        return view.as_dict()

    @server.resource(
        ARTIFACT_URI,
        name="job-artifact",
        description="One validated PNG or JPEG artifact, capped at the inline read limit.",
    )
    async def job_artifact(job_id: str, artifact_id: str) -> ResourceResult:
        try:
            content = await asyncio.to_thread(
                runtime.read_artifact, job_id, artifact_id
            )
        except ServiceError as exc:
            raise ResourceError(f"{exc.code}: {exc.message}") from exc
        return ResourceResult(
            [
                ResourceContent(
                    content.data,
                    mime_type=content.metadata.mime_type,
                )
            ]
        )

    @server.custom_route(HEALTH_PATH, methods=["GET"])
    async def health(_request: Request) -> Response:
        return JSONResponse({"status": "ok"}, headers={"Cache-Control": "no-store"})

    @server.custom_route(ARTIFACT_ROUTE, methods=["GET"])
    async def download_artifact(request: Request) -> Response:
        if not await authorize_request(request, verifier):
            return Response(
                "Unauthorized",
                status_code=401,
                headers={
                    "WWW-Authenticate": BEARER_CHALLENGE,
                    "Cache-Control": "no-store",
                },
            )
        job_id = request.path_params["job_id"]
        artifact_id = request.path_params["artifact_id"]
        try:
            content = await asyncio.to_thread(
                runtime.read_artifact,
                job_id,
                artifact_id,
                max_bytes=MAX_ARTIFACT_BYTES,
            )
        except ServiceError as exc:
            return Response(
                f"{exc.code}: {exc.message}",
                status_code=_ARTIFACT_HTTP_STATUS.get(exc.code, 500),
                media_type="text/plain",
                headers={"Cache-Control": "no-store"},
            )
        return Response(
            content.data,
            media_type=content.metadata.mime_type,
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return server


def build_http_app(
    server: FastMCP,
    *,
    host_origin_protection="auto",
    allowed_hosts=(),
    allowed_origins=(),
):
    """The same HTTP app the CLI serves, for in-process ASGI tests."""
    return server.http_app(
        path=MCP_PATH,
        transport="http",
        stateless_http=True,
        host_origin_protection=host_origin_protection,
        allowed_hosts=list(allowed_hosts) or None,
        allowed_origins=list(allowed_origins) or None,
    )


async def run_server(
    server: FastMCP,
    *,
    host: str,
    port: int,
    allowed_hosts=(),
    allowed_origins=(),
) -> None:
    """Serve the real HTTP transport on a TCP socket until shutdown."""
    await server.run_http_async(
        transport="http",
        path=MCP_PATH,
        host=host,
        port=port,
        stateless_http=True,
        show_banner=False,
        host_origin_protection="auto",
        allowed_hosts=list(allowed_hosts) or None,
        allowed_origins=list(allowed_origins) or None,
    )


async def authorize_request(request: Request, verifier) -> bool:
    """Verify a bearer token with the same verifier MCP uses.

    The custom route does not inherit FastMCP's MCP auth middleware, so this is
    the explicit check for that route. Scope validation happens inside the
    verifier, which returns ``None`` for any invalid or under-scoped token.
    """
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return False
    try:
        return await verifier.verify_token(token.strip()) is not None
    except Exception:
        # A malformed token must read as unauthorized, never as a server error
        # with a traceback or verifier internals.
        return False


def _tool_error(exc: ServiceError) -> ToolError:
    message = f"{exc.code}: {exc.message}"
    if exc.may_have_submitted:
        message += " (may_have_submitted=true)"
    return ToolError(message)

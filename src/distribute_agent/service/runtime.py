"""Durable media runtime: planning, admission, one worker, opaque artifacts.

The transport is intentionally absent. :class:`MediaRuntime` exposes four
blocking calls that a server wraps in threads:

* ``start_generation(prompt, specs)`` — validate the specs, plan the brief
  against the serving snapshot, apply the pixel and seed overrides, dry-run the
  real runner without credentials, then enqueue and return. This is the only
  mutating call.
* ``get_generation(job_id)`` — persisted status, plan and artifact metadata.
* ``read_artifact(job_id, artifact_id, max_bytes=...)`` — a bounded, no-follow
  read of one validated image.
* ``start()`` / ``stop()`` — state lock, snapshot, recovery, one worker.

``plan(request, template=None)`` is the planner step on its own, used by
``start_generation`` and by tests; no transport exposes it.

Model construction and submission are plain callables injected at construction
(``model_factory`` and ``submission``), so tests can drive the runtime offline.
The defaults are the real :class:`smolagents.InferenceClientModel` on the tested
proxy and the real :func:`distribute_agent.bridge.submit_plan`.

Package snapshots are content-addressed copies of only manifest, endpoints,
runner, workflows and skills. A job stores its digest and is processed against
that snapshot, so a later live-package edit cannot change a queued or retried
job. The live package is never mutated; the sandbox mounts the snapshot
read-only.
"""

import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

from ..agent import AgentRunError, build_agent, redact, run_agent
from ..artifacts import (
    IMAGE_SUFFIXES,
    MAX_ARTIFACT_BYTES,
    MAX_ARTIFACT_DIMENSION,
    MAX_ARTIFACT_PIXELS,
    ArtifactError,
    sniff_image,
)
from ..bridge import BridgeError, submit_plan
from ..package import Package, PackageError
from ..plan import PlanError, parse_plan
from ..skill import SkillError, load_skill
from .config import ServiceConfig
from .jobs import JobStore, StoredArtifact
from .types import (
    ARTIFACT_CORRUPT,
    ARTIFACT_TOO_LARGE,
    BUSY,
    GENERATION_DISABLED,
    INVALID_INPUT,
    INVALID_OUTPUT,
    PLAN_FAILED,
    REQUEST_CONFLICT,
    RUNTIME_ERROR,
    SHUTTING_DOWN,
    SNAPSHOT_UNAVAILABLE,
    STATE_LOCKED,
    STORAGE_ERROR,
    SUBMISSION_FAILED,
    SUBMISSION_UNCERTAIN,
    UNKNOWN_ARTIFACT,
    UNKNOWN_JOB,
    ArtifactContent,
    ArtifactMetadata,
    JobView,
    PlanResult,
    ServiceError,
    canonical_plan,
)

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SIMPLE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
SNAPSHOT_FILES = ("manifest.json", "endpoints.json", "bin/rp_oneshot.py")
MAX_ERROR_MESSAGE_CHARS = 400
# The client-facing category is stable; the template it resolves to is an
# internal implementation detail that may change between releases.
CATEGORY_TEMPLATES = {"sd-semireal": "unholy-sdxl"}
DEFAULT_CATEGORY = "sd-semireal"
SPEC_FIELDS = ("id", "category", "width", "height", "seed")
OVERRIDE_FIELDS = ("width", "height", "seed")
_ABSOLUTE_PATH = re.compile(r"(?<![\w:/])/(?:[^\s:'\"<>]+)")
_SENSITIVE_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


@dataclass(frozen=True)
class PackageSnapshot:
    digest: str
    package: Package


@dataclass(frozen=True)
class _Specs:
    """The validated client specs, with the category resolved to a template."""

    id: str
    category: str
    template: str
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None

    def canonical(self) -> dict:
        """The client's own spec, normalized: stored for replay matching."""
        canonical = {"id": self.id, "category": self.category}
        for name in OVERRIDE_FIELDS:
            value = getattr(self, name)
            if value is not None:
                canonical[name] = value
        return canonical

    def overrides(self) -> dict:
        return {
            name: getattr(self, name)
            for name in OVERRIDE_FIELDS
            if getattr(self, name) is not None
        }


class _ArtifactProblem(Exception):
    """A vendor-published file is not a complete, expected image."""


class _OutcomeUncertain(Exception):
    """A billable submission may exist but its identity was not reported."""


class MediaRuntime:
    def __init__(
        self,
        config: ServiceConfig,
        *,
        model_factory=None,
        submission=None,
        env=None,
        runner_prefix=None,
    ):
        self.config = config
        self._model_factory = model_factory
        self._submission = submission or submit_plan
        self._env = dict(os.environ if env is None else env)
        self._runner_prefix = tuple(runner_prefix) if runner_prefix else None
        self._state = "new"
        self._snapshot = None
        self._store = None
        self._lock_fd = None
        self._worker = None
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._planner_lock = threading.Lock()
        self._planner_idle = threading.Event()
        self._planner_idle.set()
        self._active_planners = 0
        self._admission_lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._state != "new":
            raise ServiceError(
                RUNTIME_ERROR, f"runtime cannot start from state {self._state!r}"
            )
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._acquire_state_lock()
            self._snapshot = self._pin_snapshot(self.config.package_root)
            self._store = JobStore(self.config.database_path)
            self._store.interrupt_running()
        except BaseException:
            self._release_resources()
            raise
        self._state = "running"
        self._worker = threading.Thread(
            target=self._worker_loop, name="mediagen-worker", daemon=True
        )
        self._worker.start()

    def stop(self, timeout=None) -> bool:
        """Stop admissions and claims, then drain bounded active work.

        Returns true when the worker and any admitted planner operation finished
        and the OS lock was released. False means the shared bound expired: a
        planner or bridge call may still be in flight, so the database and lock
        stay held for this process. Planner operations are tracked by admission,
        not by their executor-owned thread, so an idle thread pool cannot block
        shutdown. A forced process exit may leave remote work; neither
        exactly-once submission nor remote cancellation is claimed.
        """
        with self._admission_lock:
            if self._state == "new":
                self._state = "stopped"
                return True
            if self._state in ("stopped", "abandoned"):
                return self._state == "stopped"
            if self._state != "running":
                return False
            self._state = "stopping"
            worker = self._worker
            planner_running = self._active_planners > 0
        self._stopping.set()
        self._wake.set()
        deadline = time.monotonic() + (
            self.config.shutdown_timeout_seconds
            if timeout is None
            else max(0.0, timeout)
        )
        if worker is not None:
            worker.join(max(0.0, deadline - time.monotonic()))
        if planner_running:
            self._planner_idle.wait(max(0.0, deadline - time.monotonic()))
        with self._admission_lock:
            planner_running = self._active_planners > 0
        active = (worker is not None and worker.is_alive()) or planner_running
        if active:
            self._state = "abandoned"
            return False
        self._worker = None
        self._release_resources()
        self._state = "stopped"
        return True

    def _acquire_state_lock(self) -> None:
        fd = os.open(self.config.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise ServiceError(
                STATE_LOCKED,
                "another mediagen runtime already holds this state directory; stop "
                "that process before starting a second one",
            ) from exc
        self._lock_fd = fd
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        except OSError:
            pass

    def _release_resources(self) -> None:
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                pass
            self._store = None
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None

    # -- planning -------------------------------------------------------

    def _validated_request(self, request) -> str:
        """Reject a brief the planner cannot be given, before any model call."""
        if not isinstance(request, str):
            raise ServiceError(INVALID_INPUT, "prompt must be a string")
        if not request.strip():
            raise ServiceError(INVALID_INPUT, "prompt must not be empty")
        if len(request) > self.config.max_request_chars:
            raise ServiceError(
                INVALID_INPUT,
                f"prompt is {len(request)} characters; the limit is "
                f"{self.config.max_request_chars}",
            )
        return request

    def plan(self, request: str, *, template: str = None) -> PlanResult:
        request = self._validated_request(request)
        # Admission and shutdown share one lock: stop() either sees this planner
        # and drains it, or plan() sees the stopping state and rejects the call.
        with self._admission_lock:
            self._require_running()
            if not self._planner_lock.acquire(blocking=False):
                raise ServiceError(
                    BUSY,
                    "another planning request is in progress; one planner runs at a "
                    "time, so retry after it finishes",
                )
            self._active_planners += 1
            self._planner_idle.clear()
        workspace = None
        try:
            snapshot = self._snapshot
            self.config.workspace_dir.mkdir(parents=True, exist_ok=True)
            workspace = Path(
                tempfile.mkdtemp(prefix="plan-", dir=self.config.workspace_dir)
            )
            try:
                model = self._create_model()
                skill = load_skill(snapshot.package.skill_path())
                agent = build_agent(
                    model=model,
                    skill=skill,
                    package=snapshot.package,
                    workspace_dir=workspace,
                    max_steps=self.config.planner_max_steps,
                    deadline_seconds=self.config.planner_deadline_seconds,
                    per_call_timeout=self.config.sandbox_action_timeout_seconds,
                    max_model_requests=self.config.planner_max_model_requests,
                    plan_manifest=snapshot.package.manifest,
                    plan_template=template,
                )
                result = run_agent(
                    agent, request, deadline_seconds=self.config.planner_deadline_seconds
                )
                plan = parse_plan(snapshot.package.manifest, result.answer)
            except ServiceError:
                raise
            except AgentRunError as exc:
                # A provider exception can carry its raw response body. The
                # planner's diagnostics are deliberately not exposed, so report
                # a fixed operational summary instead of the exception text.
                raise ServiceError(
                    PLAN_FAILED,
                    "planning failed: the agent run did not return a valid plan "
                    "within its budget",
                ) from exc
            except (PlanError, PackageError, SkillError) as exc:
                raise ServiceError(
                    PLAN_FAILED, f"planning failed: {self._message(exc)}"
                ) from exc
            except Exception as exc:
                raise ServiceError(
                    PLAN_FAILED, "planning failed: the planner raised an unexpected error"
                ) from exc
            if self._model_factory is None:
                model_id = self.config.model_id
            else:
                model_id = str(
                    getattr(model, "model_id", None) or self.config.model_id
                )
            return PlanResult(
                plan=canonical_plan(plan),
                package_digest=snapshot.digest,
                model_id=model_id,
                diagnostics=_bounded_diagnostics(result),
            )
        finally:
            if workspace is not None:
                shutil.rmtree(workspace, ignore_errors=True)
            self._planner_lock.release()
            with self._admission_lock:
                self._active_planners -= 1
                if self._active_planners == 0:
                    self._planner_idle.set()

    def _create_model(self):
        if self._model_factory is not None:
            model = self._model_factory()
            if model is None:
                raise ServiceError(
                    PLAN_FAILED, "the injected model factory returned no model"
                )
            return model
        key = self._env.get(self.config.model_api_key_env)
        if not key:
            raise ServiceError(
                PLAN_FAILED,
                f"{self.config.model_api_key_env} is not set; planning needs the "
                "OpenAI-compatible proxy key in the service environment",
            )
        try:
            from smolagents import InferenceClientModel
        except ModuleNotFoundError as exc:
            raise ServiceError(
                PLAN_FAILED,
                "smolagents' InferenceClientModel is unavailable; install the "
                "package's pinned dependencies",
            ) from exc
        # huggingface-hub 2.2 refuses a client with both ``model`` and
        # ``base_url``. Pass the OpenAI-compatible proxy as the endpoint and the
        # provider model id as the request payload field, which leaves the
        # client URL at <proxy>/v1/chat/completions and sends
        # {"model": glm-5.3-flash} in the body.
        return InferenceClientModel(
            model_id=self.config.model_base_url,
            model=self.config.model_id,
            token=key,
            timeout=self.config.model_timeout_seconds,
            retry=False,
            max_tokens=self.config.model_max_tokens,
            temperature=self.config.model_temperature,
        )

    # -- admission and reads --------------------------------------------

    def start_generation(self, prompt: str, specs: dict) -> JobView:
        """Plan one brief and enqueue the resulting paid generation.

        ``specs`` is validated before the planner runs, so a malformed request
        never spends a model call. A known ``specs.id`` is answered from the
        store without planning: the id is the idempotency key and a planner run
        is not reproducible, so a replay could only invent a second job. The
        planner's template is pinned by the category, so the client-facing
        category keeps meaning the same thing even when the package changes.
        """
        with self._admission_lock:
            self._require_running()
        parsed = _parse_specs(specs)
        request = self._validated_request(prompt)
        # A replay is answered from the store before the live manifest or a
        # planner is consulted, so a package edit cannot turn a retry into an
        # error.
        with self._admission_lock:
            self._require_running()
            existing = self._store.get_by_request_id(parsed.id)
        if existing is not None:
            return self._replay_request(existing, request, parsed)
        manifest = self._snapshot.package.manifest
        if parsed.template not in (manifest.get("templates") or {}):
            raise ServiceError(
                RUNTIME_ERROR,
                f"category {parsed.category!r} maps to a template the shipped "
                "package does not declare",
            )
        self._validate_overrides(manifest, parsed)
        if not self._generation_available():
            raise self._generation_disabled()
        planned = self.plan(request, template=parsed.template)
        plan_data = {**planned.plan, "template": parsed.template, **parsed.overrides()}
        return self._admit_plan(
            plan_data,
            package_digest=planned.package_digest,
            request_text=request,
            specs=parsed,
            diagnostics=planned.diagnostics,
        )

    def _admit_plan(
        self,
        plan_data,
        *,
        package_digest: str,
        request_text: str,
        specs: _Specs,
        diagnostics: dict,
    ) -> JobView:
        """Validate one planned request against the manifest and enqueue it."""
        try:
            plan = parse_plan(self._snapshot.package.manifest, plan_data)
        except PlanError as exc:
            raise ServiceError(INVALID_INPUT, self._message(exc)) from exc
        canonical = canonical_plan(plan)
        if package_digest != self._snapshot.digest:
            # Both values come from this process's own pinned snapshot, so this
            # can only be an internal invariant break, never a client error.
            raise ServiceError(
                RUNTIME_ERROR,
                "the plan was produced against a package snapshot this runtime is "
                "not serving",
            )
        self._preflight(self._snapshot, canonical)
        with self._admission_lock:
            self._require_running()
            view, created = self._store.enqueue(
                request_id=specs.id,
                plan=canonical,
                package_digest=package_digest,
                max_waiting_jobs=self.config.max_waiting_jobs,
                request_text=request_text,
                specs=specs.canonical(),
                diagnostics=diagnostics,
            )
        if created:
            self._wake.set()
        return view

    def _validate_overrides(self, manifest: dict, specs: _Specs) -> None:
        """Range-check the pixel and seed overrides with the real plan parser.

        The override is parsed on its own against the pinned template, so an
        out-of-range value is rejected before the planner runs and the merged
        plan cannot disagree with what was accepted here.
        """
        if not specs.overrides():
            return
        try:
            parse_plan(manifest, {"template": specs.template, **specs.overrides()})
        except PlanError as exc:
            raise ServiceError(INVALID_INPUT, self._message(exc)) from exc

    def _replay_request(
        self, existing: JobView, request_text: str, specs: _Specs
    ) -> JobView:
        """Return the existing job only for the exact request that created it.

        The package digest is deliberately not part of the comparison: the job
        keeps its own pinned snapshot, and a live package edit must not turn a
        retry of the same request into a conflict.
        """
        if (
            existing.request_text == request_text
            and existing.specs == specs.canonical()
        ):
            return existing
        raise ServiceError(
            REQUEST_CONFLICT,
            "the supplied request id already names a different prompt or specs; "
            "use a new request id for a new job",
        )

    def get_generation(self, job_id: str) -> JobView:
        with self._admission_lock:
            self._require_open()
            view = self._store.get_job(job_id) if isinstance(job_id, str) else None
        if view is None:
            # Never echo the caller's value: an oversized id can smuggle a
            # credential past the bounded-message contract.
            raise ServiceError(UNKNOWN_JOB, "no job with the supplied id")
        return view

    def read_artifact(
        self, job_id: str, artifact_id: str, *, max_bytes: int = None
    ) -> ArtifactContent:
        with self._admission_lock:
            self._require_open()
            if not isinstance(job_id, str) or not SIMPLE_COMPONENT.fullmatch(job_id):
                raise ServiceError(UNKNOWN_ARTIFACT, "unknown artifact")
            record = (
                self._store.get_artifact(job_id, artifact_id)
                if isinstance(artifact_id, str)
                else None
            )
        if record is None:
            raise ServiceError(UNKNOWN_ARTIFACT, "unknown artifact")
        limit = self.config.max_inline_artifact_bytes if max_bytes is None else max_bytes
        if record.size_bytes > limit:
            raise ServiceError(
                ARTIFACT_TOO_LARGE,
                f"artifact is {record.size_bytes} bytes; this read is capped at "
                f"{limit}. Use the HTTP artifact route for larger files",
            )
        try:
            fd = _open_confined(self.config.output_dir / job_id, record.relative_path)
        except (OSError, _ArtifactProblem) as exc:
            raise ServiceError(
                ARTIFACT_CORRUPT,
                "the artifact is missing or is no longer a regular file inside the "
                "job output",
            ) from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size != record.size_bytes:
                raise ServiceError(
                    ARTIFACT_CORRUPT, "the artifact changed after it was recorded"
                )
            data = _read_exact(fd, record.size_bytes)
        except OSError as exc:
            raise ServiceError(
                ARTIFACT_CORRUPT, "the artifact could not be read"
            ) from exc
        except _ArtifactProblem as exc:
            raise ServiceError(ARTIFACT_CORRUPT, self._message(exc)) from exc
        finally:
            os.close(fd)
        if hashlib.sha256(data).hexdigest() != record.sha256:
            raise ServiceError(
                ARTIFACT_CORRUPT, "the artifact bytes do not match the recorded digest"
            )
        return ArtifactContent(
            metadata=ArtifactMetadata(
                artifact_id=record.artifact_id,
                job_id=job_id,
                mime_type=record.mime_type,
                width=record.width,
                height=record.height,
                size_bytes=record.size_bytes,
                sha256=record.sha256,
            ),
            data=data,
        )

    # -- worker ---------------------------------------------------------

    def _worker_loop(self) -> None:
        while not self._stopping.is_set():
            if not self._generation_available():
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            try:
                job = self._store.claim_next()
            except ServiceError:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            if job is None:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            try:
                self._process(job)
            except Exception as exc:  # keep the single worker alive
                self._finish_needs_attention(
                    job.job_id,
                    code=RUNTIME_ERROR,
                    message=f"the worker could not finish the job: {self._message(exc)}",
                )

    def _process(self, job: JobView) -> None:
        try:
            snapshot = self._load_snapshot(job.package_digest)
        except ServiceError as exc:
            self._finish_needs_attention(
                job.job_id, code=exc.code, message=self._message(exc)
            )
            return
        try:
            parse_plan(snapshot.package.manifest, job.plan)
        except PlanError as exc:
            self._finish_failure(
                job.job_id, code=PLAN_FAILED, message=self._message(exc)
            )
            return
        job_dir = self.config.output_dir / job.job_id
        if job_dir.is_symlink():
            self._finish_failure(
                job.job_id,
                code=INVALID_OUTPUT,
                message="the job output path is a symlink; refusing to write through it",
            )
            return
        try:
            job_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._finish_failure(
                job.job_id,
                code=STORAGE_ERROR,
                message=f"could not create the job output directory: {self._message(exc)}",
            )
            return
        try:
            result = self._submission(
                snapshot.package,
                job.plan,
                output_dir=job_dir,
                api_key_env=self.config.api_key_env,
                env=self._env,
                runner_prefix=self._runner_prefix,
                wait_timeout_s=self.config.bridge_poll_timeout_seconds,
                timeout=self.config.bridge_timeout_seconds,
            )
        except BridgeError as exc:
            # BridgeError text can quote the runner or a raw vendor response.
            # Persist only an allowlisted summary; the structured vendor id and
            # uncertainty flag are the operational contract for resubmission.
            if exc.may_have_submitted:
                self._finish_needs_attention(
                    job.job_id,
                    vendor_job_id=exc.job_id,
                    code=SUBMISSION_UNCERTAIN,
                    message=_uncertain_submission_message(exc.job_id),
                    may_have_submitted=True,
                )
            else:
                self._finish_failure(
                    job.job_id,
                    vendor_job_id=exc.job_id,
                    code=SUBMISSION_FAILED,
                    message=(
                        "the runner rejected the submission before a vendor job was "
                        "accepted; check the service environment and the runner logs"
                    ),
                )
            return
        except Exception:
            self._finish_needs_attention(
                job.job_id,
                code=SUBMISSION_UNCERTAIN,
                message=(
                    "the submission raised an unexpected error and may have been "
                    "accepted; check the endpoint's jobs before resubmitting"
                ),
                may_have_submitted=True,
            )
            return
        vendor_job_id = getattr(result, "job_id", None)
        output_dir = getattr(result, "output_dir", None)
        if not vendor_job_id:
            self._finish_needs_attention(
                job.job_id,
                code=SUBMISSION_UNCERTAIN,
                message=(
                    "the runner reported success without a vendor job id, so the "
                    "billable job cannot be identified"
                ),
                may_have_submitted=True,
            )
            return
        try:
            artifacts = self._ingest(job_dir, output_dir)
        except _OutcomeUncertain as exc:
            self._finish_needs_attention(
                job.job_id,
                vendor_job_id=vendor_job_id,
                code=SUBMISSION_UNCERTAIN,
                message=self._message(exc),
                may_have_submitted=True,
            )
            return
        except _ArtifactProblem as exc:
            self._finish_failure(
                job.job_id,
                vendor_job_id=vendor_job_id,
                code=INVALID_OUTPUT,
                message=self._message(exc),
            )
            return
        except OSError as exc:
            code = STORAGE_ERROR if _is_disk_full(exc) else INVALID_OUTPUT
            self._finish_failure(
                job.job_id,
                vendor_job_id=vendor_job_id,
                code=code,
                message=(
                    "could not validate the job output"
                    + ("; the filesystem is full" if code == STORAGE_ERROR else "")
                ),
            )
            return
        try:
            self._store.finish_success(job.job_id, vendor_job_id, artifacts)
        except ServiceError as exc:
            self._finish_needs_attention(
                job.job_id,
                vendor_job_id=vendor_job_id,
                code=exc.code,
                message=self._message(exc),
            )

    def _ingest(self, job_dir: Path, output_dir):
        if output_dir is None:
            raise _ArtifactProblem("the runner did not report an output directory")
        root = Path(job_dir).resolve()
        published = Path(output_dir).resolve()
        if published != root and root not in published.parents:
            raise _OutcomeUncertain(
                "the runner reported an output directory outside this job's own output"
            )
        if not published.is_dir():
            raise _ArtifactProblem("the reported output directory is missing")
        artifacts = []
        found_image = False
        for directory, dirnames, filenames in os.walk(published, followlinks=False):
            dirnames[:] = sorted(name for name in dirnames if name != "__pycache__")
            for name in sorted(filenames):
                path = Path(directory) / name
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                found_image = True
                relative = path.relative_to(root).as_posix()
                try:
                    info = os.lstat(path)
                except OSError as exc:
                    raise _ArtifactProblem(
                        f"{relative}: the published image cannot be inspected"
                    ) from exc
                if stat.S_ISLNK(info.st_mode):
                    raise _ArtifactProblem(f"{relative}: the published image is a symlink")
                if not stat.S_ISREG(info.st_mode):
                    raise _ArtifactProblem(
                        f"{relative}: the published image is not a regular file"
                    )
                if info.st_size <= 0:
                    raise _ArtifactProblem(f"{relative}: the published image is empty")
                if info.st_size > MAX_ARTIFACT_BYTES:
                    raise _ArtifactProblem(
                        f"{relative}: the image is {info.st_size} bytes; the cap is "
                        f"{MAX_ARTIFACT_BYTES}"
                    )
                data = _read_artifact_bytes(path, MAX_ARTIFACT_BYTES)
                try:
                    image_format, width, height = sniff_image(data)
                except ArtifactError as exc:
                    raise _ArtifactProblem(f"{relative}: {exc}") from exc
                artifacts.append(
                    StoredArtifact(
                        artifact_id=secrets.token_hex(16),
                        relative_path=relative,
                        mime_type=f"image/{image_format}",
                        width=width,
                        height=height,
                        size_bytes=len(data),
                        sha256=hashlib.sha256(data).hexdigest(),
                    )
                )
        if not found_image:
            raise _ArtifactProblem(
                "the job completed without publishing a PNG or JPEG image"
            )
        return tuple(artifacts)

    def _finish_failure(self, job_id, *, code, message, vendor_job_id=None, **kwargs):
        try:
            self._store.finish_failure(
                job_id,
                vendor_job_id=vendor_job_id,
                code=code,
                message=self._message(message),
                **kwargs,
            )
        except ServiceError:
            # The job stays running and startup recovery will mark it
            # needs_attention. A full or unwritable database cannot record more.
            pass

    def _finish_needs_attention(
        self, job_id, *, code, message, vendor_job_id=None, may_have_submitted=False
    ):
        self._finish_failure(
            job_id,
            vendor_job_id=vendor_job_id,
            code=code,
            message=message,
            may_have_submitted=may_have_submitted,
            needs_attention=True,
        )

    # -- snapshots ------------------------------------------------------

    def _pin_snapshot(self, source: Path) -> PackageSnapshot:
        try:
            Package.load(source)
        except PackageError as exc:
            raise ServiceError(
                RUNTIME_ERROR, f"package_root is not a usable package: {self._message(exc)}"
            ) from exc
        members = _snapshot_members(source)
        digest = _digest_members(source, members)
        destination = self.config.snapshot_dir / digest
        if destination.is_symlink():
            raise ServiceError(
                STORAGE_ERROR,
                "the package snapshot path is a symlink; remove it and restart",
            )
        if destination.exists():
            return _verified_snapshot(destination, digest)
        self.config.snapshot_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=".snapshot-", dir=self.config.snapshot_dir)
        )
        try:
            for relative in members:
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, target)
                os.chmod(target, 0o444)
            os.chmod(temporary, 0o755)
            try:
                os.rename(temporary, destination)
            except OSError:
                if not destination.exists():
                    raise
                return _verified_snapshot(destination, digest)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return _verified_snapshot(destination, digest)

    def _load_snapshot(self, digest: str) -> PackageSnapshot:
        if not isinstance(digest, str) or not DIGEST_PATTERN.fullmatch(digest):
            raise ServiceError(
                SNAPSHOT_UNAVAILABLE,
                "the job's package digest is not a valid content address",
            )
        root = self.config.snapshot_dir / digest
        if not root.is_dir():
            raise ServiceError(
                SNAPSHOT_UNAVAILABLE,
                "the package snapshot this job was planned against is no longer on "
                "disk; it was not resubmitted",
            )
        try:
            return _verified_snapshot(root, digest)
        except ServiceError as exc:
            if exc.code == SNAPSHOT_UNAVAILABLE:
                raise
            raise ServiceError(
                SNAPSHOT_UNAVAILABLE,
                "the package snapshot this job was planned against failed "
                f"verification: {exc.message}",
            ) from exc

    # -- preflight ------------------------------------------------------

    def _preflight(self, snapshot: PackageSnapshot, plan: dict) -> None:
        template = snapshot.package.manifest["templates"][plan["template"]]
        if (template.get("image") or {}).get("node"):
            raise ServiceError(
                INVALID_INPUT,
                "this endpoint submits text-only plans; image input is not accepted",
            )
        argv = [
            sys.executable,
            str(snapshot.package.runner_path),
            *parse_plan(snapshot.package.manifest, plan).to_argv(self.config.output_dir),
            "--dry-run",
        ]
        environment = {"PATH": "/usr/bin:/bin", "NO_COLOR": "1"}
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.config.preflight_timeout_seconds,
                env=environment,
                cwd=str(self.config.output_dir),
            )
        except subprocess.TimeoutExpired as exc:
            raise ServiceError(
                PLAN_FAILED,
                "the trusted runner dry-run did not finish within "
                f"{self.config.preflight_timeout_seconds:.0f}s",
            ) from exc
        except OSError as exc:
            raise ServiceError(
                PLAN_FAILED, f"the trusted runner could not be started: {self._message(exc)}"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise ServiceError(
                PLAN_FAILED,
                f"the trusted runner dry-run rejected the plan: {self._message(detail)}",
            )
        try:
            envelope = json.loads(completed.stdout)
            workflow = envelope["input"]["workflow"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ServiceError(
                PLAN_FAILED, "the trusted runner dry-run did not return a valid envelope"
            ) from exc
        if not isinstance(workflow, dict):
            raise ServiceError(
                PLAN_FAILED, "the trusted runner dry-run returned no workflow object"
            )
        width, height = _effective_dimensions(template, workflow)
        if width > MAX_ARTIFACT_DIMENSION or height > MAX_ARTIFACT_DIMENSION:
            raise ServiceError(
                INVALID_INPUT,
                f"effective output {width}x{height} exceeds the artifact decoder cap of "
                f"{MAX_ARTIFACT_DIMENSION} pixels per side",
            )
        if width * height > MAX_ARTIFACT_PIXELS:
            raise ServiceError(
                INVALID_INPUT,
                f"effective output {width}x{height} is {width * height} pixels; the "
                f"artifact decoder cap is {MAX_ARTIFACT_PIXELS} pixels",
            )

    # -- guards and messages -------------------------------------------

    def _generation_available(self) -> bool:
        return bool(
            self.config.generation_enabled
            and self._env.get(self.config.api_key_env)
        )

    def _generation_disabled(self) -> ServiceError:
        if not self.config.generation_enabled:
            return ServiceError(
                GENERATION_DISABLED,
                "generation is disabled; the operator must start the service with "
                "the explicit generation flag",
            )
        return ServiceError(
            GENERATION_DISABLED,
            f"{self.config.api_key_env} is not set; generation stays disabled until "
            "the RunPod key is present in the service environment",
        )

    def _require_running(self) -> None:
        if self._state != "running":
            raise ServiceError(
                SHUTTING_DOWN if self._state != "new" else RUNTIME_ERROR,
                f"the runtime is not accepting work (state {self._state!r})",
            )

    def _require_open(self) -> None:
        if self._state not in ("running", "stopping") or self._store is None:
            raise ServiceError(
                SHUTTING_DOWN, f"the runtime is not serving requests (state {self._state!r})"
            )

    def _message(self, value) -> str:
        text = redact(str(value) if value is not None else "")
        for name, secret in self._env.items():
            if (
                secret
                and len(secret) >= 8
                and any(marker in name.upper() for marker in _SENSITIVE_ENV_MARKERS)
            ):
                text = text.replace(secret, "[redacted]")
        text = _ABSOLUTE_PATH.sub("[path]", text)
        text = " ".join(text.split())
        if not text:
            return "no detail was reported"
        if len(text) <= MAX_ERROR_MESSAGE_CHARS:
            return text
        return text[: MAX_ERROR_MESSAGE_CHARS - 3] + "..."


def _uncertain_submission_message(vendor_job_id) -> str:
    if vendor_job_id:
        return (
            "the runner did not confirm the outcome of a billable submission; job "
            f"{vendor_job_id} may exist — read it on the endpoint before resubmitting"
        )
    return (
        "the runner did not confirm the outcome of a billable submission; check the "
        "endpoint's jobs before resubmitting"
    )


def _parse_specs(specs) -> _Specs:
    """Validate the client specs strictly: no coercion, no unknown fields.

    Values are only type-checked here. The pixel and seed ranges belong to the
    plan parser, which the runtime runs against the pinned template before the
    planner is started.
    """
    if not isinstance(specs, dict):
        raise ServiceError(INVALID_INPUT, "specs must be a JSON object")
    unknown = sorted(set(specs) - set(SPEC_FIELDS))
    if unknown:
        # Deliberately no echo: an unknown key can be arbitrarily long.
        raise ServiceError(
            INVALID_INPUT,
            "specs has an unsupported field; accepted fields are: "
            + ", ".join(SPEC_FIELDS),
        )
    request_id = specs.get("id")
    if not isinstance(request_id, str) or not REQUEST_ID_PATTERN.fullmatch(request_id):
        raise ServiceError(INVALID_INPUT, "specs.id must match [A-Za-z0-9._-]{1,128}")
    category = specs.get("category", DEFAULT_CATEGORY)
    if not isinstance(category, str) or category not in CATEGORY_TEMPLATES:
        raise ServiceError(
            INVALID_INPUT,
            "specs.category must be one of: " + ", ".join(sorted(CATEGORY_TEMPLATES)),
        )
    overrides = {}
    for name in OVERRIDE_FIELDS:
        value = specs.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ServiceError(
                INVALID_INPUT, f"specs.{name} must be an integer or omitted"
            )
        overrides[name] = value
    return _Specs(
        id=request_id,
        category=category,
        template=CATEGORY_TEMPLATES[category],
        **overrides,
    )


def _bounded_diagnostics(result) -> dict:
    diagnostics = getattr(result, "diagnostics", None) or {}
    return {
        "duration_seconds": round(float(diagnostics.get("duration_seconds") or 0.0), 3),
        "model_requests": int(diagnostics.get("model_requests") or 0),
        "agent_steps": int(diagnostics.get("agent_steps") or 0),
        "validated_previews": int(diagnostics.get("validated_previews") or 0),
        "state": str(result.state),
    }


def _effective_dimensions(template: dict, workflow: dict):
    spec = template.get("size") or {}
    node_id = str(spec.get("node") or "")
    node = workflow.get(node_id)
    inputs = node.get("inputs") if isinstance(node, dict) else None
    if not isinstance(inputs, dict):
        raise ServiceError(
            PLAN_FAILED,
            "the manifest's size node is absent from the dry-run workflow, so output "
            "dimensions cannot be checked",
        )
    width = inputs.get(spec.get("width_field") or "width")
    height = inputs.get(spec.get("height_field") or "height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ServiceError(
            PLAN_FAILED,
            f"the dry-run workflow has invalid output dimensions {width!r}x{height!r}",
        )
    return width, height


def _snapshot_members(root: Path):
    root = Path(root)
    members = []
    for relative in SNAPSHOT_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ServiceError(
                RUNTIME_ERROR, f"the package is missing a regular {relative}"
            )
        members.append(relative)
    workflow_dir = root / "workflows"
    workflows = sorted(
        path for path in workflow_dir.glob("*.json") if path.is_file()
    ) if workflow_dir.is_dir() else []
    if not workflows:
        raise ServiceError(RUNTIME_ERROR, "the package ships no workflow JSON files")
    for path in workflows:
        if path.is_symlink():
            raise ServiceError(RUNTIME_ERROR, "a package workflow is a symlink")
        members.append(path.relative_to(root).as_posix())
    skills_dir = root / "skills"
    if not skills_dir.is_dir():
        raise ServiceError(RUNTIME_ERROR, "the package ships no skills directory")
    for path in sorted(skills_dir.rglob("*")):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.is_symlink() or not path.is_file():
            raise ServiceError(RUNTIME_ERROR, "a package skill file is not a regular file")
        members.append(path.relative_to(root).as_posix())
    if len(set(members)) != len(members):
        raise ServiceError(RUNTIME_ERROR, "the package lists a file twice")
    return tuple(sorted(members))


def _digest_members(root: Path, members) -> str:
    hasher = hashlib.sha256()
    for relative in members:
        data = (Path(root) / relative).read_bytes()
        encoded = relative.encode("utf-8")
        hasher.update(encoded)
        hasher.update(b"\0")
        hasher.update(len(data).to_bytes(8, "big"))
        hasher.update(data)
    return hasher.hexdigest()


def _verified_snapshot(root: Path, digest: str) -> PackageSnapshot:
    try:
        package = Package.load(root)
        members = _snapshot_members(root)
        actual = _digest_members(root, members)
    except PackageError as exc:
        raise ServiceError(
            SNAPSHOT_UNAVAILABLE,
            f"the pinned package snapshot is incomplete: {exc}",
        ) from exc
    except OSError as exc:
        raise ServiceError(
            SNAPSHOT_UNAVAILABLE, "the pinned package snapshot cannot be read"
        ) from exc
    if actual != digest:
        raise ServiceError(
            SNAPSHOT_UNAVAILABLE,
            "the pinned package snapshot does not match its content address",
        )
    return PackageSnapshot(digest=digest, package=package)


def _open_confined(root: Path, relative: str) -> int:
    """Open one artifact without following a link at any path component."""
    parts = _relative_parts(relative)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    current = root_fd
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current
            )
            if current != root_fd:
                os.close(current)
            current = next_fd
        return os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=current,
        )
    finally:
        if current != root_fd:
            os.close(current)
        os.close(root_fd)


def _relative_parts(relative: str):
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise _ArtifactProblem("the artifact has no usable relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts:
        raise _ArtifactProblem("the artifact path is not relative")
    if any(part in ("", ".", "..") for part in pure.parts):
        raise _ArtifactProblem("the artifact path escapes its job directory")
    return pure.parts


def _read_artifact_bytes(path: Path, max_bytes: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise _ArtifactProblem("the published image is not a regular file")
        if info.st_size <= 0:
            raise _ArtifactProblem("the published image is empty")
        if info.st_size > max_bytes:
            raise _ArtifactProblem(f"the image exceeds the {max_bytes}-byte cap")
        return _read_exact(fd, info.st_size)
    finally:
        os.close(fd)


def _read_exact(fd: int, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = os.read(fd, min(remaining, 1024 * 1024))
        if not chunk:
            raise _ArtifactProblem("the file ended before its recorded size")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(fd, 1):
        raise _ArtifactProblem("the file grew after it was recorded")
    return b"".join(chunks)


def _is_disk_full(exc: OSError) -> bool:
    return exc.errno in (errno.ENOSPC, errno.EDQUOT)

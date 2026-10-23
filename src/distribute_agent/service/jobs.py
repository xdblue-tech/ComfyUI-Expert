"""Transactional SQLite job store with durable idempotency.

One database per runtime, schema ``user_version`` 1. Every mutation opens an
immediate transaction, so enqueue, claim and finish are serialised by SQLite
itself; the in-process lock only keeps one connection from interleaving
statements. ``request_id`` is unique: the same validated plan and package digest
returns the existing job, changed content is a conflict, and no caller can
create a second submission for one request id. The database is the queue, so a
restart recovers persisted jobs without any in-memory state.
"""

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .types import (
    ArtifactMetadata,
    CAPACITY,
    INTERRUPTED,
    JOB_STATUSES,
    JobView,
    REQUEST_CONFLICT,
    ServiceError,
    STORAGE_ERROR,
    error_record,
)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'succeeded', 'failed', 'needs_attention')
    ),
    plan_json TEXT NOT NULL,
    package_digest TEXT NOT NULL,
    vendor_job_id TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX jobs_status_created ON jobs(status, created_at);
CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(job_id, relative_path)
);
CREATE INDEX artifacts_job ON artifacts(job_id);
"""


@dataclass(frozen=True)
class StoredArtifact:
    """An artifact row before it becomes opaque wire metadata."""

    artifact_id: str
    relative_path: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    sha256: str


class JobStore:
    def __init__(self, path):
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ServiceError(
                STORAGE_ERROR,
                "cannot create the state directory; check permissions and free space",
            ) from exc
        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(
                str(self.path),
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
        except sqlite3.Error as exc:
            raise ServiceError(
                STORAGE_ERROR,
                "cannot open the job database; check the state directory's "
                "permissions and free space",
            ) from exc
        self._connection.row_factory = sqlite3.Row
        try:
            self._configure()
        except ServiceError:
            self._connection.close()
            raise
        except sqlite3.Error as exc:
            self._connection.close()
            raise ServiceError(
                STORAGE_ERROR,
                "the job database is not usable; move the state directory aside "
                "after saving any jobs that must be kept",
            ) from exc

    def _configure(self) -> None:
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=30000")
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            self._connection.executescript(_SCHEMA)
            self._connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        elif version != SCHEMA_VERSION:
            raise ServiceError(
                STORAGE_ERROR,
                f"job database schema version {version} is incompatible; this "
                f"runtime expects {SCHEMA_VERSION}",
            )
        tables = {
            row[0]
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        missing = {"jobs", "artifacts"} - tables
        if missing:
            raise ServiceError(
                STORAGE_ERROR,
                "the job database schema is incompatible; missing table(s): "
                + ", ".join(sorted(missing)),
            )

    # -- reads ---------------------------------------------------------

    def get_job(self, job_id) -> JobView:
        if not isinstance(job_id, str) or not job_id:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            return self._view(row)

    def get_by_request_id(self, request_id) -> JobView:
        if not isinstance(request_id, str) or not request_id:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE request_id = ?", (request_id,)
            ).fetchone()
            if row is None:
                return None
            return self._view(row)

    def count_queued(self) -> int:
        with self._lock:
            return self._connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'queued'"
            ).fetchone()[0]

    def get_artifact(self, job_id, artifact_id):
        if not isinstance(job_id, str) or not isinstance(artifact_id, str):
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM artifacts WHERE job_id = ? AND artifact_id = ?",
                (job_id, artifact_id),
            ).fetchone()
            return _artifact_record(row) if row is not None else None

    # -- mutations -----------------------------------------------------

    def enqueue(
        self,
        *,
        request_id: str,
        plan: dict,
        package_digest: str,
        max_waiting_jobs: int,
    ):
        """Insert one queued job, or return the identical existing one.

        Returns ``(view, created)``. Capacity is counted inside the same
        transaction, so concurrent admissions cannot exceed the waiting bound.
        """
        plan_json = _dump(plan, "plan")
        with self._lock:
            self._begin()
            try:
                row = self._connection.execute(
                    "SELECT * FROM jobs WHERE request_id = ?", (request_id,)
                ).fetchone()
                if row is not None:
                    self._connection.rollback()
                    if row["plan_json"] == plan_json and row["package_digest"] == package_digest:
                        return self.get_job(row["job_id"]), False
                    raise ServiceError(
                        REQUEST_CONFLICT,
                        f"request id {request_id!r} already names a different plan or "
                        "package digest; use a new request id for a new job",
                    )
                waiting = self._connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status = 'queued'"
                ).fetchone()[0]
                if waiting >= max_waiting_jobs:
                    self._connection.rollback()
                    raise ServiceError(
                        CAPACITY,
                        f"{waiting} jobs are already waiting and the limit is "
                        f"{max_waiting_jobs}; wait for the worker to drain one before "
                        "submitting another",
                    )
                job_id = str(uuid.uuid4())
                now = _now()
                self._connection.execute(
                    "INSERT INTO jobs (job_id, request_id, status, plan_json, "
                    "package_digest, created_at, updated_at) "
                    "VALUES (?, ?, 'queued', ?, ?, ?, ?)",
                    (job_id, request_id, plan_json, package_digest, now, now),
                )
                self._connection.commit()
            except ServiceError:
                self._rollback()
                raise
            except sqlite3.Error as exc:
                self._rollback()
                raise ServiceError(
                    STORAGE_ERROR,
                    "could not persist the job; the state filesystem may be full "
                    "or unwritable",
                ) from exc
            return self.get_job(job_id), True

    def claim_next(self) -> JobView:
        """Atomically move the oldest queued job to running, or return ``None``."""
        with self._lock:
            self._begin()
            try:
                row = self._connection.execute(
                    "SELECT * FROM jobs WHERE status = 'queued' "
                    "ORDER BY created_at, rowid LIMIT 1"
                ).fetchone()
                if row is None:
                    self._connection.commit()
                    return None
                now = _now()
                updated = self._connection.execute(
                    "UPDATE jobs SET status = 'running', started_at = ?, updated_at = ? "
                    "WHERE job_id = ? AND status = 'queued'",
                    (now, now, row["job_id"]),
                )
                if updated.rowcount != 1:
                    self._connection.rollback()
                    return None
                self._connection.commit()
            except sqlite3.Error as exc:
                self._rollback()
                raise ServiceError(
                    STORAGE_ERROR,
                    "could not claim a queued job; the state filesystem may be full "
                    "or unwritable",
                ) from exc
            except ServiceError:
                self._rollback()
                raise
            return self.get_job(row["job_id"])

    def finish_success(self, job_id, vendor_job_id, artifacts) -> None:
        with self._lock:
            self._begin()
            try:
                row = self._connection.execute(
                    "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                if row is None or row["status"] != "running":
                    self._connection.rollback()
                    raise ServiceError(
                        STORAGE_ERROR,
                        f"job {job_id!r} is not running; refusing to record a result",
                    )
                now = _now()
                self._connection.execute(
                    "UPDATE jobs SET status = 'succeeded', vendor_job_id = ?, "
                    "error_json = NULL, finished_at = ?, updated_at = ? "
                    "WHERE job_id = ?",
                    (vendor_job_id, now, now, job_id),
                )
                for artifact in artifacts:
                    self._connection.execute(
                        "INSERT INTO artifacts (artifact_id, job_id, relative_path, "
                        "mime_type, width, height, size_bytes, sha256, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            artifact.artifact_id,
                            job_id,
                            artifact.relative_path,
                            artifact.mime_type,
                            artifact.width,
                            artifact.height,
                            artifact.size_bytes,
                            artifact.sha256,
                            now,
                        ),
                    )
                self._connection.commit()
            except ServiceError:
                self._rollback()
                raise
            except sqlite3.Error as exc:
                self._rollback()
                raise ServiceError(
                    STORAGE_ERROR,
                    "could not record the completed job; the state filesystem may be "
                    "full or unwritable",
                ) from exc

    def finish_failure(
        self,
        job_id,
        *,
        vendor_job_id=None,
        code: str,
        message: str,
        may_have_submitted: bool = False,
        needs_attention: bool = False,
    ) -> None:
        status = "needs_attention" if needs_attention else "failed"
        with self._lock:
            self._begin()
            try:
                row = self._connection.execute(
                    "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                if row is None or row["status"] != "running":
                    self._connection.rollback()
                    raise ServiceError(
                        STORAGE_ERROR,
                        f"job {job_id!r} is not running; refusing to record a failure",
                    )
                now = _now()
                self._connection.execute(
                    "UPDATE jobs SET status = ?, vendor_job_id = COALESCE(?, "
                    "vendor_job_id), error_json = ?, finished_at = ?, updated_at = ? "
                    "WHERE job_id = ?",
                    (
                        status,
                        vendor_job_id,
                        _dump(
                            error_record(
                                code, message, may_have_submitted=may_have_submitted
                            ),
                            "error",
                        ),
                        now,
                        now,
                        job_id,
                    ),
                )
                self._connection.commit()
            except ServiceError:
                self._rollback()
                raise
            except sqlite3.Error as exc:
                self._rollback()
                raise ServiceError(
                    STORAGE_ERROR,
                    "could not record the failed job; the state filesystem may be "
                    "full or unwritable",
                ) from exc

    def interrupt_running(self) -> int:
        """Mark jobs left ``running`` by a previous process as needs_attention.

        They are never queued again: a bridge call may have been accepted before
        the process died, so resubmission could bill twice.
        """
        now = _now()
        message = (
            "the service stopped while this job was running; it was not queued "
            "again or resubmitted because the vendor may already have accepted it"
        )
        with self._lock:
            self._begin()
            try:
                updated = self._connection.execute(
                    "UPDATE jobs SET status = 'needs_attention', error_json = ?, "
                    "finished_at = ?, updated_at = ? WHERE status = 'running'",
                    (
                        _dump(
                            error_record(
                                INTERRUPTED, message, may_have_submitted=True
                            ),
                            "error",
                        ),
                        now,
                        now,
                    ),
                )
                self._connection.commit()
            except sqlite3.Error as exc:
                self._rollback()
                raise ServiceError(
                    STORAGE_ERROR,
                    "could not recover interrupted jobs; the state filesystem may be "
                    "full or unwritable",
                ) from exc
            return updated.rowcount

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    # -- helpers -------------------------------------------------------

    def _begin(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")

    def _rollback(self) -> None:
        try:
            self._connection.rollback()
        except sqlite3.Error:
            pass

    def _view(self, row) -> JobView:
        artifacts = self._connection.execute(
            "SELECT * FROM artifacts WHERE job_id = ? ORDER BY created_at, artifact_id",
            (row["job_id"],),
        ).fetchall()
        try:
            plan = json.loads(row["plan_json"])
            error = json.loads(row["error_json"]) if row["error_json"] else None
        except ValueError as exc:
            raise ServiceError(
                STORAGE_ERROR, "the job database contains an unreadable record"
            ) from exc
        if row["status"] not in JOB_STATUSES:
            raise ServiceError(
                STORAGE_ERROR,
                f"the job database contains unknown status {row['status']!r}",
            )
        return JobView(
            job_id=row["job_id"],
            request_id=row["request_id"],
            status=row["status"],
            plan=plan,
            package_digest=row["package_digest"],
            vendor_job_id=row["vendor_job_id"],
            artifacts=tuple(_artifact_metadata(artifact) for artifact in artifacts),
            error=error,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )


def _artifact_metadata(row) -> ArtifactMetadata:
    return ArtifactMetadata(
        artifact_id=row["artifact_id"],
        job_id=row["job_id"],
        mime_type=row["mime_type"],
        width=row["width"],
        height=row["height"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
    )


def _artifact_record(row) -> StoredArtifact:
    return StoredArtifact(
        artifact_id=row["artifact_id"],
        relative_path=row["relative_path"],
        mime_type=row["mime_type"],
        width=row["width"],
        height=row["height"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
    )


def _dump(value, label: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ServiceError(STORAGE_ERROR, f"could not serialise the {label}") from exc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")

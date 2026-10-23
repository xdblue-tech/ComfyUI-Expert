"""Settings for one private media runtime.

Defaults are the tested values from the approved plan: one bind address and
port, the pinned model and proxy, the six-step / seven-request / 180-second planning
budget, the 30-second sandbox action, and the 900-second bridge with a
780-second poll. Generation stays disabled unless the operator enables it with
the CLI flag or MEDIAGEN_ENABLE_GENERATION, and the RunPod key is present in the
environment.

The dataclass is plain data. CLI flags, environment parsing and secret handling
belong to the transport/CLI stage, which constructs this object.
"""

from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_API_KEY_ENV = "RUNPOD_API_KEY"
DEFAULT_MODEL_ID = "glm-5.3-flash"
DEFAULT_MODEL_BASE_URL = "http://hades.home:8317/v1"
DEFAULT_MODEL_API_KEY_ENV = "MEDIAGEN_PROXY_API_KEY"
DEFAULT_MODEL_MAX_TOKENS = 4096
DEFAULT_MODEL_TEMPERATURE = 0.2
DEFAULT_MODEL_TIMEOUT_SECONDS = 60
DEFAULT_PLANNER_MAX_STEPS = 6
DEFAULT_PLANNER_MAX_MODEL_REQUESTS = 7
DEFAULT_PLANNER_DEADLINE_SECONDS = 180.0
DEFAULT_SANDBOX_ACTION_TIMEOUT_SECONDS = 30.0
DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 60.0
DEFAULT_BRIDGE_TIMEOUT_SECONDS = 900.0
DEFAULT_BRIDGE_POLL_TIMEOUT_SECONDS = 780
DEFAULT_MAX_WAITING_JOBS = 20
DEFAULT_MAX_REQUEST_CHARS = 8192
DEFAULT_MAX_INLINE_ARTIFACT_BYTES = 8 * 1024 * 1024
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 5.0

DEFAULT_AUTH_SECRET_ENV = "MEDIAGEN_JWT_SECRET"
DEFAULT_AUTH_ISSUER = "mediagen"
DEFAULT_AUTH_AUDIENCE = "mediagen-mcp"
DEFAULT_AUTH_SCOPE = "media:access"
DEFAULT_AUTH_TOKEN_TTL_SECONDS = 7 * 24 * 60 * 60


class ConfigError(ValueError):
    """The service settings cannot describe a safe runtime."""


@dataclass(frozen=True)
class ServiceConfig:
    package_root: Path
    state_dir: Path
    output_dir: Path
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    generation_enabled: bool = False
    api_key_env: str = DEFAULT_API_KEY_ENV
    model_id: str = DEFAULT_MODEL_ID
    model_base_url: str = DEFAULT_MODEL_BASE_URL
    model_api_key_env: str = DEFAULT_MODEL_API_KEY_ENV
    model_max_tokens: int = DEFAULT_MODEL_MAX_TOKENS
    model_temperature: float = DEFAULT_MODEL_TEMPERATURE
    model_timeout_seconds: int = DEFAULT_MODEL_TIMEOUT_SECONDS
    planner_max_steps: int = DEFAULT_PLANNER_MAX_STEPS
    planner_max_model_requests: int = DEFAULT_PLANNER_MAX_MODEL_REQUESTS
    planner_deadline_seconds: float = DEFAULT_PLANNER_DEADLINE_SECONDS
    sandbox_action_timeout_seconds: float = DEFAULT_SANDBOX_ACTION_TIMEOUT_SECONDS
    preflight_timeout_seconds: float = DEFAULT_PREFLIGHT_TIMEOUT_SECONDS
    bridge_timeout_seconds: float = DEFAULT_BRIDGE_TIMEOUT_SECONDS
    bridge_poll_timeout_seconds: int = DEFAULT_BRIDGE_POLL_TIMEOUT_SECONDS
    max_waiting_jobs: int = DEFAULT_MAX_WAITING_JOBS
    max_request_chars: int = DEFAULT_MAX_REQUEST_CHARS
    max_inline_artifact_bytes: int = DEFAULT_MAX_INLINE_ARTIFACT_BYTES
    shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
    auth_secret_env: str = DEFAULT_AUTH_SECRET_ENV
    auth_issuer: str = DEFAULT_AUTH_ISSUER
    auth_audience: str = DEFAULT_AUTH_AUDIENCE
    auth_scope: str = DEFAULT_AUTH_SCOPE
    auth_token_ttl_seconds: int = DEFAULT_AUTH_TOKEN_TTL_SECONDS
    allowed_hosts: tuple = ()
    allowed_origins: tuple = ()

    def __post_init__(self):
        package_root = Path(self.package_root).resolve()
        state_dir = Path(self.state_dir).resolve()
        output_dir = Path(self.output_dir).resolve()
        object.__setattr__(self, "package_root", package_root)
        object.__setattr__(self, "state_dir", state_dir)
        object.__setattr__(self, "output_dir", output_dir)
        if not self.host or not isinstance(self.host, str):
            raise ConfigError("host must be a non-empty string")
        if not _is_bind_address(self.host):
            raise ConfigError(
                f"host {self.host!r} is not a bind address; use an IP literal such "
                "as 127.0.0.1, 0.0.0.0 or ::1, or 'localhost'"
            )
        if not isinstance(self.port, int) or isinstance(self.port, bool):
            raise ConfigError("port must be an integer")
        if not 1 <= self.port <= 65535:
            raise ConfigError(f"port {self.port} is outside 1..65535")
        for label, value in (
            ("planner_max_steps", self.planner_max_steps),
            ("planner_max_model_requests", self.planner_max_model_requests),
            ("model_max_tokens", self.model_max_tokens),
            ("model_timeout_seconds", self.model_timeout_seconds),
            ("max_inline_artifact_bytes", self.max_inline_artifact_bytes),
        ):
            _require_positive_int(label, value)
        for label, value in (
            ("planner_deadline_seconds", self.planner_deadline_seconds),
            ("sandbox_action_timeout_seconds", self.sandbox_action_timeout_seconds),
            ("preflight_timeout_seconds", self.preflight_timeout_seconds),
            ("bridge_timeout_seconds", self.bridge_timeout_seconds),
            ("shutdown_timeout_seconds", self.shutdown_timeout_seconds),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ConfigError(f"{label} must be a positive number")
        if not isinstance(self.bridge_poll_timeout_seconds, int) or isinstance(
            self.bridge_poll_timeout_seconds, bool
        ):
            raise ConfigError("bridge_poll_timeout_seconds must be an integer")
        if not 0 < self.bridge_poll_timeout_seconds < self.bridge_timeout_seconds:
            raise ConfigError(
                "bridge_poll_timeout_seconds must be positive and below "
                "bridge_timeout_seconds"
            )
        if not isinstance(self.max_waiting_jobs, int) or isinstance(
            self.max_waiting_jobs, bool
        ):
            raise ConfigError("max_waiting_jobs must be an integer")
        if self.max_waiting_jobs < 0:
            raise ConfigError("max_waiting_jobs must not be negative")
        _require_positive_int("max_request_chars", self.max_request_chars)
        _require_positive_int("auth_token_ttl_seconds", self.auth_token_ttl_seconds)
        for label, value in (
            ("auth_secret_env", self.auth_secret_env),
            ("auth_issuer", self.auth_issuer),
            ("auth_audience", self.auth_audience),
            ("auth_scope", self.auth_scope),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{label} must be a non-empty string")
        for label, value in (
            ("allowed_hosts", self.allowed_hosts),
            ("allowed_origins", self.allowed_origins),
        ):
            if isinstance(value, str) or not isinstance(value, (tuple, list)):
                raise ConfigError(f"{label} must be a sequence of strings")
            entries = tuple(value)
            if any(not isinstance(entry, str) or not entry for entry in entries):
                raise ConfigError(f"{label} entries must be non-empty strings")
            object.__setattr__(self, label, entries)
        if not 0 <= self.model_temperature <= 2:
            raise ConfigError("model_temperature must be between 0 and 2")
        _require_outside("output_dir", output_dir, package_root)
        _require_outside("state_dir", state_dir, package_root)
        _require_outside("output_dir", output_dir, state_dir / "packages")
        if output_dir == state_dir:
            raise ConfigError("output_dir and state_dir must be different directories")

    @property
    def database_path(self) -> Path:
        return self.state_dir / "jobs.sqlite3"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / "runtime.lock"

    @property
    def snapshot_dir(self) -> Path:
        return self.state_dir / "packages"

    @property
    def workspace_dir(self) -> Path:
        return self.state_dir / "workspaces"


def _is_bind_address(host: str) -> bool:
    """An IP literal or the localhost name; uvicorn can bind either one."""
    if host == "localhost":
        return True
    try:
        ip_address(host)
    except ValueError:
        return False
    return True


def _require_positive_int(label: str, value) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{label} must be a positive integer")


def _require_outside(label: str, path: Path, forbidden: Path) -> None:
    if path == forbidden or forbidden in path.parents:
        raise ConfigError(f"{label} {path} must stay outside {forbidden}")

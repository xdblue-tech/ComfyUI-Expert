"""Opt-in CLIs. Both refuse to run without their explicit enable flag."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .service.config import (
    ConfigError,
    DEFAULT_AUTH_AUDIENCE,
    DEFAULT_AUTH_ISSUER,
    DEFAULT_AUTH_SCOPE,
    DEFAULT_AUTH_SECRET_ENV,
    DEFAULT_AUTH_TOKEN_TTL_SECONDS,
    DEFAULT_HOST,
    DEFAULT_MODEL_API_KEY_ENV,
    DEFAULT_MODEL_BASE_URL,
    DEFAULT_MODEL_ID,
    DEFAULT_PORT,
)

BIND_HOST_ENV = "MEDIAGEN_HOST"
BIND_PORT_ENV = "MEDIAGEN_PORT"
MODEL_ID_ENV = "MEDIAGEN_MODEL_ID"
MODEL_BASE_URL_ENV = "MEDIAGEN_MODEL_BASE_URL"
MODEL_API_KEY_ENV_ENV = "MEDIAGEN_MODEL_API_KEY_ENV"
ENABLE_GENERATION_ENV = "MEDIAGEN_ENABLE_GENERATION"

# Snapshot before the harness imports below: importing the agent stack runs
# smolagents' load_dotenv(), which copies a nearby .env into os.environ. Reading
# the environment later would treat unexported serve settings as exported ones
# and silently move the listen address, swap the planning model or enable paid
# generation. Only these six values are captured.
_CONFIG_ENVIRONMENT = {
    name: os.environ[name]
    for name in (
        BIND_HOST_ENV,
        BIND_PORT_ENV,
        MODEL_ID_ENV,
        MODEL_BASE_URL_ENV,
        MODEL_API_KEY_ENV_ENV,
        ENABLE_GENERATION_ENV,
    )
    if name in os.environ
}

from .live_eval import DEFAULT_DEADLINE_SECONDS, DEFAULT_MAX_STEPS, DEFAULT_REPETITIONS
from .live_eval import run_live_eval
from .vendor_smoke import DEFAULT_PLAN, run_vendor_smoke, write_smoke_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Opt-in live harnesses for the distribute_agent package."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    live = subparsers.add_parser(
        "live-eval", help="5 briefs x 3 repetitions against a real model"
    )
    live.add_argument("--enable-live-model", action="store_true",
                      help="required: allow calls to a billed model provider")
    live.add_argument("--model-backend", choices=("litellm", "inference-client"),
                      default="litellm")
    live.add_argument("--model-id", help="provider model id, for example openai/gpt-4o-mini")
    live.add_argument("--api-key-env", default="HF_TOKEN",
                      help="environment variable holding the provider key")
    live.add_argument("--package", required=True, help="path to the shipped package")
    live.add_argument("--workspace", required=True, help="directory for per-run sandboxes")
    live.add_argument("--output", help="directory for live-eval.json")
    live.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    live.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    live.add_argument("--deadline-seconds", type=float, default=DEFAULT_DEADLINE_SECONDS)
    live.add_argument("--max-requests", type=int,
                      help="hard cap on provider requests for this invocation; "
                           "enforced at the model call, including final answers")

    vendor = subparsers.add_parser(
        "vendor-smoke", help="one real RunPod job, artifact checked by header"
    )
    vendor.add_argument("--enable-live-vendor", action="store_true",
                        help="required: allow one billed RunPod job")
    vendor.add_argument("--package", required=True, help="path to the shipped package")
    vendor.add_argument("--output", required=True, help="directory for job artifacts")
    vendor.add_argument("--budget-seconds", type=float, default=900.0)
    vendor.add_argument("--seed", type=int, default=DEFAULT_PLAN["seed"])
    vendor.add_argument("--width", type=int, default=DEFAULT_PLAN["width"])
    vendor.add_argument("--height", type=int, default=DEFAULT_PLAN["height"])

    serve = subparsers.add_parser(
        "serve", help="run the private authenticated MCP endpoint"
    )
    serve.add_argument("--package", required=True, help="path to the shipped package")
    serve.add_argument("--state-dir", default="state/mediagen",
                       help="runtime state and SQLite database")
    serve.add_argument("--output-dir", default="output/mediagen",
                       help="validated job artifacts")
    serve.add_argument(
        "--host",
        default=None,
        help=f"bind address; defaults to {BIND_HOST_ENV} or {DEFAULT_HOST}",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"TCP port; defaults to {BIND_PORT_ENV} or {DEFAULT_PORT}",
    )
    generation = serve.add_mutually_exclusive_group()
    generation.add_argument(
        "--enable-generation",
        action="store_true",
        dest="enable_generation",
        default=None,
        help=f"allow paid jobs; overrides {ENABLE_GENERATION_ENV}; "
             "also requires RUNPOD_API_KEY",
    )
    generation.add_argument(
        "--disable-generation",
        action="store_false",
        dest="enable_generation",
        help=f"force generation off, even when {ENABLE_GENERATION_ENV} is true",
    )
    serve.add_argument("--model-id", default=None, help="planning model id")
    serve.add_argument("--model-base-url", default=None,
                       help="OpenAI-compatible planning proxy")
    serve.add_argument("--model-api-key-env", default=None,
                       help="environment variable holding the planning proxy key")
    serve.add_argument("--allow-host", action="append", default=[],
                       help="extra Host header value accepted by the guard")
    serve.add_argument("--allow-origin", action="append", default=[],
                       help="extra browser Origin accepted by the guard")
    _add_auth_arguments(serve)

    token = subparsers.add_parser(
        "token", help="issue one HS256 bearer token for the endpoint"
    )
    token.add_argument("--subject", default="owner")
    token.add_argument("--ttl-seconds", type=int, default=DEFAULT_AUTH_TOKEN_TTL_SECONDS)
    _add_auth_arguments(token)
    return parser


def _add_auth_arguments(parser) -> None:
    parser.add_argument("--auth-secret-env", default=DEFAULT_AUTH_SECRET_ENV,
                        help="environment variable holding the shared JWT secret")
    parser.add_argument("--issuer", default=DEFAULT_AUTH_ISSUER)
    parser.add_argument("--audience", default=DEFAULT_AUTH_AUDIENCE)
    parser.add_argument("--scope", default=DEFAULT_AUTH_SCOPE)


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "serve":
        return _serve(arguments)
    if arguments.command == "token":
        return _token(arguments)
    if arguments.command == "live-eval":
        report = run_live_eval(
            enable_live_model=arguments.enable_live_model,
            model_factory=(
                _model_factory(arguments) if arguments.enable_live_model else None
            ),
            package_root=arguments.package,
            workspace_root=arguments.workspace,
            output_dir=arguments.output,
            repetitions=arguments.repetitions,
            max_steps=arguments.max_steps,
            deadline_seconds=arguments.deadline_seconds,
            max_requests=arguments.max_requests,
            model_metadata={
                "backend": arguments.model_backend,
                "model_id": arguments.model_id or "(not set)",
                "api_key_env": arguments.api_key_env,
            },
        )
        print(json.dumps(report, indent=2))
        if report["status"] == "not-run":
            return 2
        return 0 if report["status"] == "completed" else 1

    report = run_vendor_smoke(
        enable_live_vendor=arguments.enable_live_vendor,
        package_root=arguments.package,
        output_dir=arguments.output,
        plan_data={
            **DEFAULT_PLAN,
            "seed": arguments.seed,
            "width": arguments.width,
            "height": arguments.height,
        },
        budget_seconds=arguments.budget_seconds,
    )
    if report["status"] != "not-run":
        write_smoke_report(report, arguments.output)
    print(json.dumps(report, indent=2))
    if report["status"] == "not-run":
        return 2
    return 0 if report["status"] == "passed" else 1


def _bind_address(cli_host, cli_port, environment):
    """Resolve the listen address: flags, then exported settings, then defaults.

    The caller passes the environment snapshot taken at the CLI boundary before
    the harness imports; those pull in smolagents, whose module-level
    load_dotenv() copies a nearby .env into os.environ. Reading later would treat
    unexported bind settings as exported ones and silently move the listen
    address.
    """
    host = cli_host
    if host is None:
        host = environment.get(BIND_HOST_ENV, "").strip() or DEFAULT_HOST
    port = cli_port
    if port is None:
        exported = environment.get(BIND_PORT_ENV, "").strip()
        if exported:
            try:
                port = int(exported)
            except ValueError as exc:
                raise ConfigError(
                    f"{BIND_PORT_ENV}={exported!r} is not an integer port"
                ) from exc
        else:
            port = DEFAULT_PORT
    return host, port


def _model_settings(arguments, environment):
    """Resolve the planning model: flags, then exported settings, then defaults."""
    settings = {}
    for flag, variable, fallback in (
        ("model_id", MODEL_ID_ENV, DEFAULT_MODEL_ID),
        ("model_base_url", MODEL_BASE_URL_ENV, DEFAULT_MODEL_BASE_URL),
        ("model_api_key_env", MODEL_API_KEY_ENV_ENV, DEFAULT_MODEL_API_KEY_ENV),
    ):
        value = getattr(arguments, flag) or environment.get(variable, "").strip()
        settings[flag] = value or fallback
    return settings


def _generation_enabled(cli_value, environment):
    """Resolve paid generation: an explicit CLI choice, then a strict export."""
    if cli_value is not None:
        return cli_value
    exported = environment.get(ENABLE_GENERATION_ENV, "")
    normalized = exported.strip().lower()
    if not normalized:
        return False
    if normalized in ("true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    raise ConfigError(
        f"{ENABLE_GENERATION_ENV}={exported!r} is not a boolean; use "
        "true/false, 1/0, yes/no or on/off"
    )


def _serve(arguments) -> int:
    # Resolve from the boundary snapshot, before the service stack imports the
    # agent and smolagents' load_dotenv() can inject a nearby .env.
    try:
        host, port = _bind_address(arguments.host, arguments.port, _CONFIG_ENVIRONMENT)
        model_settings = _model_settings(arguments, _CONFIG_ENVIRONMENT)
        generation_enabled = _generation_enabled(
            arguments.enable_generation, _CONFIG_ENVIRONMENT
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        from .service.auth import AuthConfigError, build_verifier, load_secret
        from .service.app import build_server, run_server
        from .service.config import ServiceConfig
        from .service.runtime import MediaRuntime
    except ModuleNotFoundError as exc:
        print(
            "error: the server extra is not installed "
            f"({exc}); install the package with its 'server' extra",
            file=sys.stderr,
        )
        return 2

    try:
        config = ServiceConfig(
            package_root=Path(arguments.package),
            state_dir=Path(arguments.state_dir),
            output_dir=Path(arguments.output_dir),
            host=host,
            port=port,
            generation_enabled=generation_enabled,
            auth_secret_env=arguments.auth_secret_env,
            auth_issuer=arguments.issuer,
            auth_audience=arguments.audience,
            auth_scope=arguments.scope,
            allowed_hosts=tuple(arguments.allow_host),
            allowed_origins=tuple(arguments.allow_origin),
            **model_settings,
        )
        secret = load_secret(secret_env=config.auth_secret_env)
        verifier = build_verifier(
            secret,
            issuer=config.auth_issuer,
            audience=config.auth_audience,
            scope=config.auth_scope,
        )
    except (ConfigError, AuthConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if config.generation_enabled and not os.environ.get(config.api_key_env):
        print(
            f"warning: generation is enabled but {config.api_key_env} is absent; "
            "generation stays disabled",
            file=sys.stderr,
        )
    runtime = MediaRuntime(config)
    server = build_server(runtime, verifier=verifier)
    try:
        asyncio.run(
            run_server(
                server,
                host=config.host,
                port=config.port,
                allowed_hosts=config.allowed_hosts,
                allowed_origins=config.allowed_origins,
            )
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _token(arguments) -> int:
    try:
        from .service.auth import AuthConfigError, issue_token, load_secret
    except ModuleNotFoundError as exc:
        print(
            "error: the server extra is not installed "
            f"({exc}); install the package with its 'server' extra",
            file=sys.stderr,
        )
        return 2
    try:
        secret = load_secret(secret_env=arguments.auth_secret_env)
        token = issue_token(
            secret,
            subject=arguments.subject,
            issuer=arguments.issuer,
            audience=arguments.audience,
            scope=arguments.scope,
            ttl_seconds=arguments.ttl_seconds,
        )
    except AuthConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(token)
    return 0


def _model_factory(arguments):
    model_id = arguments.model_id
    if not model_id:
        raise SystemExit("--model-id is required with --enable-live-model")
    key = os.environ.get(arguments.api_key_env)
    if not key:
        raise SystemExit(
            f"{arguments.api_key_env} is not set — export the provider key in the "
            "environment; it is never written to artifacts"
        )
    if arguments.model_backend == "litellm":
        from smolagents import LiteLLMModel

        model_class, model_kwargs = LiteLLMModel, {"api_key": key}
    else:
        # huggingface-hub, the only dependency of this backend, ships with base smolagents.
        from smolagents import InferenceClientModel

        model_class, model_kwargs = InferenceClientModel, {"token": key}

    def factory():
        return model_class(model_id=model_id, **model_kwargs)

    # Preflight once, before the matrix starts: constructing the model triggers the
    # backend's own dependency import (litellm for LiteLLMModel) but no network call.
    # A missing backend must not be discovered after the first bills are spent.
    try:
        factory()
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"the {arguments.model_backend} backend cannot be constructed: {exc} — "
            "install it (this project's pyproject extra is 'litellm') and rerun. "
            "No provider call was made"
        ) from exc
    return factory


if __name__ == "__main__":
    sys.exit(main())

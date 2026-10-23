"""Shared fixtures for the transport suite.

This directory has no ``__init__.py``; ``discover -s tests/service`` puts it on
``sys.path`` so this module imports like the other service test helpers.
"""

from distribute_agent.service.app import build_http_app, build_server
from distribute_agent.service.auth import build_verifier, issue_token
from distribute_agent.service.runtime import MediaRuntime

from service_test_support import TEST_KEY, base_config

TEST_SECRET = "test-only-jwt-secret-0123456789abcdef"
TEST_TOKEN = issue_token(TEST_SECRET)
TEST_ENV = {"RUNPOD_API_KEY": TEST_KEY, "MEDIAGEN_PROXY_API_KEY": "proxy-test-key"}


def test_verifier():
    return build_verifier(TEST_SECRET)


def auth_headers(token=None):
    return {"Authorization": f"Bearer {token or TEST_TOKEN}"}


def make_runtime(
    tmp,
    *,
    generation_enabled=True,
    model_factory=None,
    submission=None,
    env=None,
    **config_overrides,
):
    config = base_config(tmp, generation_enabled=generation_enabled, **config_overrides)
    runtime = MediaRuntime(
        config,
        model_factory=model_factory,
        submission=submission,
        env=TEST_ENV if env is None else env,
    )
    return runtime, config


def make_server(runtime):
    return build_server(runtime, verifier=test_verifier())


def make_http_app(server, **kwargs):
    return build_http_app(server, **kwargs)


def free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


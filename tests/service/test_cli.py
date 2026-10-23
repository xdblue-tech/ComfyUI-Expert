"""The real CLI: token issuance and an actual TCP server on a real socket."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from distribute_agent.service.auth import build_verifier

from service_test_support import PROJECT_ROOT
from transport_support import TEST_SECRET, free_port

PACKAGE = PROJECT_ROOT / "distribute"

# Loaded as sitecustomize in one subprocess. It reproduces smolagents'
# import-time load_dotenv(): MEDIAGEN_HOST appears at the first harness import,
# before any service import, which must still be too late for an absent bind
# setting.
LATE_BIND_PROBE = '''\
import os
import sys


def _hook(event, args):
    if event != "import" or not args:
        return
    name = str(args[0])
    if not (name.startswith("smolagents") or name in (
        "distribute_agent.live_eval",
        "distribute_agent.vendor_smoke",
    )):
        return
    os.environ.setdefault("MEDIAGEN_HOST", "0.0.0.0")
    with open(os.environ["BIND_PROBE_MARKER"], "w", encoding="utf-8") as handle:
        handle.write("fired")

sys.addaudithook(_hook)
'''

# Same hook for the four planning/generation settings: they are injected when
# smolagents is first imported, which is after the CLI's boundary snapshot.
LATE_MODEL_PROBE = '''\
import os
import sys


def _hook(event, args):
    if event != "import" or not args:
        return
    name = str(args[0])
    if not (name.startswith("smolagents") or name in (
        "distribute_agent.live_eval",
        "distribute_agent.vendor_smoke",
    )):
        return
    os.environ.setdefault("MEDIAGEN_MODEL_ID", "late/model")
    os.environ.setdefault("MEDIAGEN_MODEL_BASE_URL", "http://late.invalid/v1")
    os.environ.setdefault("MEDIAGEN_MODEL_API_KEY_ENV", "LATE_PROXY_KEY")
    os.environ.setdefault("MEDIAGEN_ENABLE_GENERATION", "true")
    with open(os.environ["MODEL_PROBE_MARKER"], "w", encoding="utf-8") as handle:
        handle.write("fired")

sys.addaudithook(_hook)
'''

# Imports the CLI first (so its boundary snapshot runs before smolagents'
# import-time load_dotenv), then swaps the runtime and server for fakes that
# record the resolved configuration. No model, vendor or socket call is made.
RUNTIME_CAPTURE = '''\
import json
import sys
from pathlib import Path

import distribute_agent.__main__ as cli
import distribute_agent.service.app as app_module
import distribute_agent.service.runtime as runtime_module

capture_path = Path(sys.argv[1])


class RecordingRuntime:
    def __init__(self, config):
        self.config = config
        capture_path.write_text(
            json.dumps(
                {
                    "host": config.host,
                    "port": config.port,
                    "model_id": config.model_id,
                    "model_base_url": config.model_base_url,
                    "model_api_key_env": config.model_api_key_env,
                    "generation_enabled": config.generation_enabled,
                }
            ),
            encoding="utf-8",
        )


async def fake_run_server(
    server, *, host, port, allowed_hosts=(), allowed_origins=()
):
    return None


app_module.run_server = fake_run_server
runtime_module.MediaRuntime = RecordingRuntime
raise SystemExit(cli.main(sys.argv[2:]))
'''


def cli_env(**extra):
    # Built from scratch instead of inherited: the operator's .env must never
    # reach these subprocesses. Importing smolagents calls load_dotenv(), which
    # walks up from this checkout's in-repo virtualenv and would otherwise
    # inject that file, so every operator-owned setting is pinned to an empty
    # export (falsy, so the CLI's own defaults still apply).
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": tempfile.gettempdir(),
        "NO_COLOR": "1",
        "LANG": "C.UTF-8",
        "MEDIAGEN_HOST": "",
        "MEDIAGEN_PORT": "",
        "MEDIAGEN_MODEL_ID": "",
        "MEDIAGEN_MODEL_BASE_URL": "",
        "MEDIAGEN_MODEL_API_KEY_ENV": "",
        "MEDIAGEN_ENABLE_GENERATION": "",
        "MEDIAGEN_JWT_SECRET": "",
        "MEDIAGEN_PROXY_API_KEY": "",
        "RUNPOD_API_KEY": "",
    }
    environment.update(extra)
    return environment


def run_cli(*arguments, env=None, cwd=None, timeout=30):
    return subprocess.run(
        [sys.executable, "-m", "distribute_agent", *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
        env=env if env is not None else cli_env(),
    )


def wait_for_health(port, process, timeout=20.0):
    target = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"serve exited {process.returncode} before health\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        try:
            with urllib.request.urlopen(target, timeout=1) as response:
                if response.status == 200:
                    return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError):
            time.sleep(0.05)
    process.terminate()
    stdout, stderr = process.communicate(timeout=10)
    raise AssertionError(f"serve never answered /health\nstdout:\n{stdout}\nstderr:\n{stderr}")


async def list_tools_over_tcp(port, token):
    transport = StreamableHttpTransport(
        f"http://127.0.0.1:{port}/mcp",
        headers={"Authorization": f"Bearer {token}"},
    )
    async with Client(transport) as client:
        return [tool.name for tool in await client.list_tools()]


def assert_mcp_requires_bearer(testcase, port):
    """An unauthenticated MCP initialize must be rejected on the live socket."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/mcp",
        data=json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "cli-test", "version": "1"},
                },
            }
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    with testcase.assertRaises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=5)
    testcase.assertEqual(caught.exception.code, 401)
    testcase.assertIn("Bearer", caught.exception.headers.get("www-authenticate", ""))


class TokenCliTests(unittest.TestCase):
    def test_token_prints_a_verifiable_token_and_no_secret(self):
        completed = run_cli(
            "token", "--subject", "owner", "--ttl-seconds", "600",
            env=cli_env(MEDIAGEN_JWT_SECRET=TEST_SECRET),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        token = completed.stdout.strip()
        self.assertEqual(len(token.splitlines()), 1)
        self.assertNotIn(TEST_SECRET, completed.stdout + completed.stderr)
        access = asyncio.run(build_verifier(TEST_SECRET).verify_token(token))
        self.assertIsNotNone(access)
        self.assertEqual(access.subject, "owner")
        self.assertEqual(access.scopes, ["media:access"])
        self.assertEqual(access.claims["exp"] - access.claims["iat"], 600)

    def test_token_refuses_missing_or_short_secrets(self):
        for label, env in (
            ("missing", cli_env()),
            ("short", cli_env(MEDIAGEN_JWT_SECRET="short")),
        ):
            with self.subTest(secret=label):
                completed = run_cli("token", env=env)
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(completed.stdout, "")
                self.assertIn("MEDIAGEN_JWT_SECRET", completed.stderr)

    def test_token_refuses_invalid_ttl(self):
        completed = run_cli(
            "token", "--ttl-seconds", "0", env=cli_env(MEDIAGEN_JWT_SECRET=TEST_SECRET)
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("ttl_seconds", completed.stderr)


class ServeCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def serve_arguments(self, port=None, state="state", output="output"):
        arguments = [
            "serve",
            "--package", str(PACKAGE),
            "--state-dir", str(self.root / state),
            "--output-dir", str(self.root / output),
        ]
        if port is not None:
            arguments += ["--port", str(port)]
        return arguments

    def capture_config(self, *flags, env=None, name="runtime.json"):
        capture_path = self.root / name
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                RUNTIME_CAPTURE,
                str(capture_path),
                *self.serve_arguments(),
                *flags,
            ],
            cwd=self.root,
            env=env if env is not None else cli_env(MEDIAGEN_JWT_SECRET=TEST_SECRET),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn(TEST_SECRET, completed.stdout + completed.stderr)
        return json.loads(capture_path.read_text(encoding="utf-8"))

    def test_exported_model_settings_reach_the_runtime_configuration(self):
        config = self.capture_config(
            env=cli_env(
                MEDIAGEN_JWT_SECRET=TEST_SECRET,
                MEDIAGEN_MODEL_ID="exported/model",
                MEDIAGEN_MODEL_BASE_URL="http://exported.invalid/v1",
                MEDIAGEN_MODEL_API_KEY_ENV="EXPORTED_PROXY_KEY",
            )
        )
        self.assertEqual(config["model_id"], "exported/model")
        self.assertEqual(config["model_base_url"], "http://exported.invalid/v1")
        self.assertEqual(config["model_api_key_env"], "EXPORTED_PROXY_KEY")

    def test_model_flags_override_exported_settings(self):
        config = self.capture_config(
            "--model-id", "flag/model",
            "--model-base-url", "http://flag.invalid/v1",
            "--model-api-key-env", "FLAG_PROXY_KEY",
            env=cli_env(
                MEDIAGEN_JWT_SECRET=TEST_SECRET,
                MEDIAGEN_MODEL_ID="exported/model",
                MEDIAGEN_MODEL_BASE_URL="http://exported.invalid/v1",
                MEDIAGEN_MODEL_API_KEY_ENV="EXPORTED_PROXY_KEY",
            ),
        )
        self.assertEqual(config["model_id"], "flag/model")
        self.assertEqual(config["model_base_url"], "http://flag.invalid/v1")
        self.assertEqual(config["model_api_key_env"], "FLAG_PROXY_KEY")

    def test_absent_or_blank_model_exports_keep_the_pinned_defaults(self):
        blank = cli_env(MEDIAGEN_JWT_SECRET=TEST_SECRET)
        absent = cli_env(MEDIAGEN_JWT_SECRET=TEST_SECRET)
        for name in (
            "MEDIAGEN_MODEL_ID",
            "MEDIAGEN_MODEL_BASE_URL",
            "MEDIAGEN_MODEL_API_KEY_ENV",
        ):
            del absent[name]
        for label, environment in (("blank", blank), ("absent", absent)):
            with self.subTest(exports=label):
                config = self.capture_config(env=environment, name=f"{label}.json")
                self.assertEqual(config["model_id"], "glm-5.3-flash")
                self.assertEqual(
                    config["model_base_url"], "http://hades.home:8317/v1"
                )
                self.assertEqual(
                    config["model_api_key_env"], "MEDIAGEN_PROXY_API_KEY"
                )

    def test_exported_generation_boolean_is_strict_and_case_insensitive(self):
        for exported, expected in (
            ("true", True),
            (" TRUE ", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("", False),
        ):
            with self.subTest(export=exported):
                config = self.capture_config(
                    env=cli_env(
                        MEDIAGEN_JWT_SECRET=TEST_SECRET,
                        MEDIAGEN_ENABLE_GENERATION=exported,
                    ),
                    name=f"generation-{exported.strip() or 'empty'}.json",
                )
                self.assertEqual(config["generation_enabled"], expected)

    def test_serve_refuses_an_invalid_generation_export(self):
        completed = run_cli(
            *self.serve_arguments(),
            env=cli_env(
                MEDIAGEN_JWT_SECRET=TEST_SECRET,
                MEDIAGEN_ENABLE_GENERATION="maybe",
            ),
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("MEDIAGEN_ENABLE_GENERATION", completed.stderr)
        self.assertFalse((self.root / "state").exists())

    def test_explicit_generation_flags_override_exports(self):
        for flag, exported, expected in (
            ("--disable-generation", "true", False),
            ("--enable-generation", "false", True),
            # An explicit flag must win without parsing the malformed export.
            ("--disable-generation", "maybe", False),
        ):
            with self.subTest(flag=flag, export=exported):
                config = self.capture_config(
                    flag,
                    env=cli_env(
                        MEDIAGEN_JWT_SECRET=TEST_SECRET,
                        MEDIAGEN_ENABLE_GENERATION=exported,
                    ),
                    name=f"{flag.strip('-')}-{exported}.json",
                )
                self.assertEqual(config["generation_enabled"], expected)

    def test_absent_model_and_generation_exports_ignore_a_late_dotenv_side_effect(self):
        probe_dir = self.root / "late-model-env"
        probe_dir.mkdir()
        (probe_dir / "sitecustomize.py").write_text(
            LATE_MODEL_PROBE, encoding="utf-8"
        )
        marker = self.root / "late-model-env.fired"
        environment = cli_env(
            MEDIAGEN_JWT_SECRET=TEST_SECRET,
            PYTHONPATH=str(probe_dir),
            MODEL_PROBE_MARKER=str(marker),
        )
        for name in (
            "MEDIAGEN_MODEL_ID",
            "MEDIAGEN_MODEL_BASE_URL",
            "MEDIAGEN_MODEL_API_KEY_ENV",
            "MEDIAGEN_ENABLE_GENERATION",
        ):
            del environment[name]
        config = self.capture_config(env=environment)
        self.assertTrue(marker.exists(), "the late model-env side effect never fired")
        self.assertEqual(config["model_id"], "glm-5.3-flash")
        self.assertEqual(config["model_base_url"], "http://hades.home:8317/v1")
        self.assertEqual(config["model_api_key_env"], "MEDIAGEN_PROXY_API_KEY")
        self.assertFalse(config["generation_enabled"])

    def test_serve_refuses_to_start_without_the_jwt_secret(self):
        completed = run_cli(*self.serve_arguments(free_port()), env=cli_env())
        self.assertEqual(completed.returncode, 2)
        self.assertIn("MEDIAGEN_JWT_SECRET", completed.stderr)
        self.assertFalse((self.root / "state").exists())

    def test_serve_refuses_an_invalid_bind_host(self):
        completed = run_cli(
            *self.serve_arguments(free_port()),
            "--host", "media.internal",
            env=cli_env(MEDIAGEN_JWT_SECRET=TEST_SECRET),
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("bind address", completed.stderr)
        self.assertFalse((self.root / "state").exists())

    def test_serve_refuses_invalid_exported_bind_settings(self):
        for label, bind_env, marker in (
            (
                "host",
                {"MEDIAGEN_HOST": "media.internal", "MEDIAGEN_PORT": str(free_port())},
                "bind address",
            ),
            (
                "port",
                {"MEDIAGEN_HOST": "127.0.0.1", "MEDIAGEN_PORT": "http"},
                "MEDIAGEN_PORT",
            ),
        ):
            with self.subTest(setting=label):
                completed = run_cli(
                    *self.serve_arguments(),
                    env=cli_env(MEDIAGEN_JWT_SECRET=TEST_SECRET, **bind_env),
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(marker, completed.stderr)
                self.assertFalse((self.root / "state").exists())

    def test_serve_honors_exported_bind_settings(self):
        port = free_port()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "distribute_agent",
                *self.serve_arguments(),
                # A listed host keeps FastMCP's guard active on a wildcard bind.
                "--allow-host",
                "127.0.0.1",
            ],
            cwd=self.root,
            env=cli_env(
                MEDIAGEN_JWT_SECRET=TEST_SECRET,
                MEDIAGEN_HOST="0.0.0.0",
                MEDIAGEN_PORT=str(port),
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # 0.0.0.0 is reached through loopback, which is where the client talks.
            self.assertEqual(wait_for_health(port, process), {"status": "ok"})
            foreign_host = urllib.request.Request(
                f"http://127.0.0.1:{port}/health",
                headers={"Host": "evil.example"},
            )
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(foreign_host, timeout=5)
            self.assertEqual(caught.exception.code, 421)
            assert_mcp_requires_bearer(self, port)
            token = run_cli(
                "token", env=cli_env(MEDIAGEN_JWT_SECRET=TEST_SECRET)
            ).stdout.strip()
            self.assertEqual(
                asyncio.run(list_tools_over_tcp(port, token)),
                ["start_generation", "get_generation"],
            )
        finally:
            process.terminate()
            stdout, stderr = process.communicate(timeout=20)
        self.assertIn("Application shutdown complete", stderr)
        self.assertNotIn(TEST_SECRET, stdout + stderr)

    def test_absent_bind_settings_ignore_a_late_dotenv_side_effect(self):
        probe_dir = self.root / "late-env"
        probe_dir.mkdir()
        (probe_dir / "sitecustomize.py").write_text(LATE_BIND_PROBE, encoding="utf-8")
        marker = self.root / "late-env.fired"
        port = free_port()
        environment = cli_env(
            MEDIAGEN_JWT_SECRET=TEST_SECRET,
            PYTHONPATH=str(probe_dir),
            BIND_PROBE_MARKER=str(marker),
        )
        del environment["MEDIAGEN_HOST"]
        del environment["MEDIAGEN_PORT"]
        process = subprocess.Popen(
            [sys.executable, "-m", "distribute_agent", *self.serve_arguments(port)],
            cwd=self.root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(wait_for_health(port, process), {"status": "ok"})
        finally:
            process.terminate()
            stdout, stderr = process.communicate(timeout=20)
        self.assertTrue(marker.exists(), "the late bind-env side effect never fired")
        self.assertIn(f"http://127.0.0.1:{port}", stderr)
        self.assertNotIn("http://0.0.0.0", stderr)
        self.assertIn("Application shutdown complete", stderr)

    def test_serve_flags_override_exported_bind_settings(self):
        port = free_port()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "distribute_agent",
                *self.serve_arguments(port),
                "--host",
                "0.0.0.0",
            ],
            cwd=self.root,
            env=cli_env(
                MEDIAGEN_JWT_SECRET=TEST_SECRET,
                # An unroutable bind and an unparsable port: the flags must win
                # over both, and a broken export must not block a good --port.
                MEDIAGEN_HOST="192.0.2.10",
                MEDIAGEN_PORT="not-a-port",
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(wait_for_health(port, process), {"status": "ok"})
        finally:
            process.terminate()
            stdout, stderr = process.communicate(timeout=20)
        self.assertIn("Application shutdown complete", stderr)
        self.assertNotIn(TEST_SECRET, stdout + stderr)

    def test_serve_runs_on_a_real_socket_and_protects_the_artifact_route(self):
        port = free_port()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "distribute_agent",
                *self.serve_arguments(port),
                "--enable-generation",
            ],
            cwd=self.root,
            env=cli_env(MEDIAGEN_JWT_SECRET=TEST_SECRET),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(wait_for_health(port, process), {"status": "ok"})
            foreign_host = urllib.request.Request(
                f"http://127.0.0.1:{port}/health",
                headers={"Host": "evil.example"},
            )
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(foreign_host, timeout=5)
            self.assertEqual(caught.exception.code, 421)
            foreign_origin = urllib.request.Request(
                f"http://127.0.0.1:{port}/health",
                headers={"Origin": "http://evil.example"},
            )
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(foreign_origin, timeout=5)
            self.assertEqual(caught.exception.code, 403)
            assert_mcp_requires_bearer(self, port)

            token = run_cli(
                "token", env=cli_env(MEDIAGEN_JWT_SECRET=TEST_SECRET)
            ).stdout.strip()
            self.assertEqual(
                asyncio.run(list_tools_over_tcp(port, token)),
                ["start_generation", "get_generation"],
            )
            artifact = urllib.request.Request(
                f"http://127.0.0.1:{port}/artifacts/missing/missing",
                headers={"Authorization": f"Bearer {token}"},
            )
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(artifact, timeout=5)
            self.assertEqual(caught.exception.code, 404)
            self.assertIn(b"unknown_artifact", caught.exception.read())
        finally:
            process.terminate()
            stdout, stderr = process.communicate(timeout=20)
        # uvicorn re-raises the captured SIGTERM after a graceful shutdown, so the
        # process reports the signal; the shutdown-complete line proves it drained.
        self.assertIn(
            process.returncode,
            (0, -signal.SIGTERM),
            f"stdout:\n{stdout}\nstderr:\n{stderr}",
        )
        self.assertIn("Application shutdown complete", stderr)
        self.assertIn("generation stays disabled", stderr)
        self.assertNotIn(TEST_SECRET, stdout + stderr)

    def test_serve_refuses_a_second_instance_on_the_same_state(self):
        first_port = free_port()
        first = subprocess.Popen(
            [sys.executable, "-m", "distribute_agent", *self.serve_arguments(first_port)],
            cwd=self.root,
            env=cli_env(MEDIAGEN_JWT_SECRET=TEST_SECRET),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            wait_for_health(first_port, first)
            second = run_cli(
                *self.serve_arguments(free_port()),
                env=cli_env(MEDIAGEN_JWT_SECRET=TEST_SECRET),
                timeout=30,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already holds this state directory", second.stderr)
        finally:
            first.terminate()
            first.communicate(timeout=20)


if __name__ == "__main__":
    unittest.main()

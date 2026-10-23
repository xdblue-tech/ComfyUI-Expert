"""The real CLI: token issuance and an actual TCP server on loopback."""

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


def cli_env(**extra):
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": tempfile.gettempdir(),
        "NO_COLOR": "1",
        "LANG": "C.UTF-8",
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

    def serve_arguments(self, port, state="state", output="output"):
        return [
            "serve",
            "--package", str(PACKAGE),
            "--state-dir", str(self.root / state),
            "--output-dir", str(self.root / output),
            "--port", str(port),
        ]

    def test_serve_refuses_to_start_without_the_jwt_secret(self):
        completed = run_cli(*self.serve_arguments(free_port()), env=cli_env())
        self.assertEqual(completed.returncode, 2)
        self.assertIn("MEDIAGEN_JWT_SECRET", completed.stderr)
        self.assertFalse((self.root / "state").exists())

    def test_serve_refuses_a_non_loopback_bind(self):
        completed = run_cli(
            *self.serve_arguments(free_port()),
            "--host", "0.0.0.0",
            env=cli_env(MEDIAGEN_JWT_SECRET=TEST_SECRET),
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("loopback", completed.stderr)
        self.assertFalse((self.root / "state").exists())

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
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(caught.exception.code, 401)
            self.assertIn("Bearer", caught.exception.headers.get("www-authenticate", ""))

            token = run_cli(
                "token", env=cli_env(MEDIAGEN_JWT_SECRET=TEST_SECRET)
            ).stdout.strip()
            self.assertEqual(
                asyncio.run(list_tools_over_tcp(port, token)),
                ["plan_media", "start_generation", "get_generation"],
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

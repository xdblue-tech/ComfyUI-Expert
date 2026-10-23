"""Shared fixtures for the service suite.

This directory has no ``__init__.py``, so the root stdlib discovery never
imports it; ``discover -s tests/service`` puts this directory on ``sys.path``.
"""

import json
import shutil
import struct
import threading
import time
import zlib
from dataclasses import replace
from pathlib import Path

from distribute_agent.bridge import BridgeError, BridgeResult
from distribute_agent.scripted_model import ScriptedModel, code_reply
from distribute_agent.service.config import ServiceConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SOURCE = PROJECT_ROOT / "distribute"

VALID_PLAN = {
    "template": "unholy-sdxl",
    "prompt": "masterpiece, a brass clock on a stormy cliff, no text",
    "seed": 42,
    "width": 832,
    "height": 1216,
}
TEST_KEY = "rpa_test_secret_value"


def copy_package(destination):
    destination = Path(destination)
    shutil.copytree(
        PACKAGE_SOURCE,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "output", "projects"),
    )
    return destination


def base_config(tmp, **overrides):
    base = ServiceConfig(
        package_root=copy_package(Path(tmp) / "package"),
        state_dir=Path(tmp) / "state",
        output_dir=Path(tmp) / "output",
        generation_enabled=True,
    )
    return replace(base, **overrides)


def plan_reply(plan):
    return code_reply(f"final_answer({plan!r})")


def scripted_factory(replies):
    """A model factory that builds a fresh ScriptedModel for every call."""
    created = []

    def factory():
        model = ScriptedModel(list(replies))
        created.append(model)
        return model

    return factory, created


class BlockingScriptedModel(ScriptedModel):
    """Block on the first provider call until the test releases it."""

    def __init__(self, replies):
        super().__init__(replies)
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, messages, **kwargs):
        self.started.set()
        if not self.release.wait(10):
            raise RuntimeError("blocking scripted model was never released")
        return super().generate(messages, **kwargs)


def png_chunk(kind, payload):
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def build_png(width=4, height=6, color=(220, 40, 40)):
    raw = b"".join(b"\x00" + bytes(color) * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(raw))
        + png_chunk(b"IEND", b"")
    )


PNG_BYTES = build_png()
JPEG_HEADER_ONLY = bytes.fromhex(
    "ffd8ffc0001108000100010301110002110003110000ffd9"
)


class FakeSubmission:
    """A ``submit_plan``-shaped callable with switchable outcomes.

    It deliberately has the same keyword contract as
    :func:`distribute_agent.bridge.submit_plan`, so the runtime's injection point
    is the real seam and not a test-only protocol.
    """

    def __init__(self, behavior="valid", vendor_id="vendor-1"):
        self.behavior = behavior
        self.vendor_id = vendor_id
        self.calls = []
        self.lock = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.counter = 0

    def block(self):
        self.entered.clear()
        self.release.clear()

    def unblock(self):
        self.release.set()

    def __call__(
        self,
        package,
        plan,
        *,
        output_dir,
        api_key_env,
        env,
        runner_prefix,
        wait_timeout_s,
        timeout,
    ):
        with self.lock:
            self.counter += 1
            call = {
                "package": package,
                "plan": dict(plan),
                "output_dir": Path(output_dir),
                "api_key_env": api_key_env,
                "env": dict(env),
                "runner_prefix": runner_prefix,
                "wait_timeout_s": wait_timeout_s,
                "timeout": timeout,
                "number": self.counter,
            }
            self.calls.append(call)
        self.entered.set()
        if not self.release.wait(10):
            raise AssertionError("the fake submission was never released")
        behavior = self.behavior
        if callable(behavior):
            return behavior(call)
        if behavior == "raise_uncertain":
            raise BridgeError(
                "runner exited 1: network error after POST at /home/user/private; "
                f"key={TEST_KEY}; vendor_private_body=DO_NOT_PUBLISH_123 — "
                "submission outcome is UNKNOWN",
                may_have_submitted=True,
                job_id="vendor-77",
            )
        if behavior == "raise_definite":
            raise BridgeError(
                "runner exited 1: authentication failed (HTTP 401)",
                may_have_submitted=False,
            )
        if behavior == "raise_unexpected":
            raise RuntimeError("unexpected bridge crash in /home/user/private")
        destination = Path(output_dir) / self.vendor_id
        destination.mkdir(parents=True, exist_ok=True)
        if behavior == "valid":
            (destination / "image_000.png").write_bytes(PNG_BYTES)
        elif behavior == "two_images":
            (destination / "a.png").write_bytes(build_png(3, 5))
            (destination / "b.png").write_bytes(build_png(7, 2, color=(0, 100, 200)))
            (destination / "response.json").write_text("{}", encoding="utf-8")
        elif behavior == "corrupt":
            (destination / "bad.png").write_bytes(b"<html>not an image</html>")
        elif behavior == "partial":
            (destination / "partial.jpg").write_bytes(JPEG_HEADER_ONLY)
        elif behavior == "empty_file":
            (destination / "empty.png").write_bytes(b"")
        elif behavior == "oversized":
            target = destination / "big.png"
            with target.open("wb") as handle:
                handle.truncate(64 * 1024 * 1024 + 1)
        elif behavior == "symlink":
            outside = Path(output_dir).parent / "outside.png"
            outside.write_bytes(PNG_BYTES)
            (destination / "linked.png").symlink_to(outside)
        elif behavior == "one_valid_one_corrupt":
            (destination / "a.png").write_bytes(PNG_BYTES)
            (destination / "b.png").write_bytes(b"broken")
        elif behavior == "no_output":
            return BridgeResult(
                returncode=0,
                job_id=self.vendor_id,
                output_dir=None,
                stdout="",
                stderr="",
            )
        elif behavior == "no_vendor_id":
            return BridgeResult(
                returncode=0,
                job_id=None,
                output_dir=destination,
                stdout="",
                stderr="",
            )
        elif behavior == "outside_output":
            return BridgeResult(
                returncode=0,
                job_id=self.vendor_id,
                output_dir=Path(output_dir).parent,
                stdout="",
                stderr="",
            )
        return BridgeResult(
            returncode=0,
            job_id=self.vendor_id,
            output_dir=destination,
            stdout="",
            stderr="",
        )


def wait_until(predicate, timeout=10.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition was not met before the timeout")


def wait_for_job(runtime, job_id, statuses, timeout=10.0):
    statuses = set(statuses)

    def matches():
        try:
            view = runtime.get_generation(job_id)
        except Exception:
            return None
        return view if view.status in statuses else None

    return wait_until(matches, timeout=timeout)


def read_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_manifest(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

"""Behaviour tests for the shipped standalone runner ``distribute/bin/rp_oneshot.py``.

The runner is exercised directly from a temporary copy of the package, never
from ``distribute/`` itself. Network traffic goes through one narrow seam,
``rp_oneshot._http_open``; the fake transport records the real requests the
runner builds (method, URL, JSON body, headers) so submission and polling logic
is tested without the network.
"""

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_SOURCE = PROJECT_ROOT / "distribute"

TEMPLATE = "unholy-sdxl"
TEMPLATE_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, jpeg "
    "artifacts, signature, watermark, username, blurry, deformed, ugly, blurry "
    "eyes, blurry face, out of focus eyes, bad eyes, closed eyes, crossed eyes, "
    "old, mature, elderly, middle aged, wrinkles, aged skin, saggy, big cock, "
    "large penis, huge dick, massive penis, bbw, hung, illustration, drawing, "
    "cartoon, painting, sketch, art style"
)


def copy_package(destination):
    shutil.copytree(
        PACKAGE_SOURCE,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "output", "projects"),
    )
    return Path(destination)


def load_runner(package_root):
    path = Path(package_root) / "bin" / "rp_oneshot.py"
    spec = importlib.util.spec_from_file_location(f"rp_oneshot_under_test_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PACKAGE_ROOT == Path(package_root).resolve()
    return module


def tree_digest(root):
    digest = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


class FakeResponse:
    def __init__(self, payload):
        if isinstance(payload, bytes):
            self._body = payload
        elif isinstance(payload, str):
            self._body = payload.encode("utf-8")
        else:
            self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class FakeTransport:
    """Records the runner's requests and replays scripted responses."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def __call__(self, target, timeout=None):
        if isinstance(target, str):
            method, url, body, headers = "GET", target, None, {}
        else:
            method = target.get_method()
            url = target.full_url
            body = json.loads(target.data.decode("utf-8")) if target.data else None
            headers = {name.lower(): value for name, value in target.header_items()}
        call = {"method": method, "url": url, "body": body, "headers": headers, "timeout": timeout}
        self.calls.append(call)
        result = self.handler(method, url, body)
        if isinstance(result, BaseException):
            raise result
        return result if isinstance(result, FakeResponse) else FakeResponse(result)

    def methods(self):
        return [call["method"] for call in self.calls]

    def url_calls(self, needle):
        return [call for call in self.calls if needle in call["url"]]


def encode(data):
    return base64.b64encode(data).decode("ascii")


class RunnerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.package = copy_package(self.root / "package")
        self.runner = load_runner(self.package)
        self.runner.POLL_SECONDS = 0
        self.output = self.root / "output"

    def enable_image_input(self):
        manifest_path = self.package / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["templates"][TEMPLATE]["image"] = {"node": "8"}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def run_cli(self, argv, transport=None, env=None):
        module = self.runner
        transport = transport or FakeTransport(lambda *_args: self.fail("unexpected HTTP request"))
        module._http_open = transport
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, env or {}, clear=True):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    code = module.main(argv)
                except module.OneshotError as exc:
                    stderr.write(f"[FAIL] {exc}\n")
                    code = 1
                except SystemExit as exc:
                    code = exc.code if isinstance(exc.code, int) else 1
        return code, stdout.getvalue(), stderr.getvalue()

    def dry_run(self, transport, extra=()):
        return self.run_cli(
            [
                "--template", TEMPLATE,
                "--output", str(self.output),
                "--dry-run",
                *extra,
            ],
            transport=transport,
        )

    def completed_job(self, job_id="job-1", output=None):
        return {
            "id": job_id,
            "status": "COMPLETED",
            "output": output if output is not None else {"images": []},
        }

    def submit_and_complete(self, output, job_id="job-1"):
        """A transport that accepts one job and returns it completed."""

        def handler(method, url, body):
            if method == "POST":
                return {"id": job_id, "status": "IN_QUEUE"}
            if "/status/" in url:
                return self.completed_job(job_id, output)
            self.fail(f"unexpected request {method} {url}")

        return FakeTransport(handler)


class DryRunTests(RunnerTestCase):
    """E2/E3/P1/P3/S4: the offline preview path."""

    def test_dry_run_needs_no_key_makes_no_http_calls_and_writes_nothing(self):
        transport = FakeTransport(lambda *_args: self.fail("dry-run performed an HTTP request"))
        code, stdout, stderr = self.dry_run(
            transport,
            ["--prompt", "a red apple", "--seed", "42", "--width", "832", "--height", "1216"],
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(transport.calls, [])
        envelope = json.loads(stdout)
        self.assertIn("workflow", envelope["input"])
        self.assertFalse(self.output.exists(), "dry-run must not create the output directory")

    def test_prompt_negative_seed_and_size_are_preserved_as_data(self):
        prompt = (
            'portrait "of" $(touch /tmp/should-not-exist); `id` & \\ backslash\n'
            "second line — unicode ✓ 日本語, exclusions: no text"
        )
        code, stdout, stderr = self.dry_run(
            FakeTransport(lambda *_args: self.fail("dry-run performed an HTTP request")),
            ["--prompt", prompt, "--seed", "42", "--width", "832", "--height", "1216"],
        )
        self.assertEqual(code, 0, stderr)
        workflow = json.loads(stdout)["input"]["workflow"]
        self.assertEqual(workflow["2"]["inputs"]["text"], prompt)
        self.assertEqual(workflow["3"]["inputs"]["text"], TEMPLATE_NEGATIVE)
        self.assertEqual(workflow["5"]["inputs"]["seed"], 42)
        self.assertIsInstance(workflow["5"]["inputs"]["seed"], int)
        self.assertEqual(workflow["4"]["inputs"]["width"], 832)
        self.assertEqual(workflow["4"]["inputs"]["height"], 1216)

    def test_explicit_negative_override_replaces_template_negative(self):
        code, stdout, stderr = self.dry_run(
            FakeTransport(lambda *_args: self.fail("dry-run performed an HTTP request")),
            ["--prompt", "x", "--negative", "only the explicit negative"],
        )
        self.assertEqual(code, 0, stderr)
        workflow = json.loads(stdout)["input"]["workflow"]
        self.assertEqual(workflow["3"]["inputs"]["text"], "only the explicit negative")

    def test_runner_works_from_a_package_path_with_spaces(self):
        package = copy_package(self.root / "package dir with spaces")
        runner = load_runner(package)
        runner._http_open = FakeTransport(lambda *_args: self.fail("dry-run performed an HTTP request"))
        stdout, stderr = io.StringIO(), io.StringIO()
        output = self.root / "output dir with spaces"
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = runner.main(
                ["--template", TEMPLATE, "--prompt", "spaces", "--output", str(output),
                 "--dry-run"]
            )
        self.assertEqual(code, 0, stderr.getvalue())
        workflow = json.loads(stdout.getvalue())["input"]["workflow"]
        self.assertEqual(workflow["2"]["inputs"]["text"], "spaces")
        self.assertFalse(output.exists())

    def test_package_bytes_are_unchanged_by_a_dry_run(self):
        before = tree_digest(self.package)
        code, _stdout, stderr = self.dry_run(
            FakeTransport(lambda *_args: self.fail("dry-run performed an HTTP request")),
            ["--prompt", "x", "--seed", "1"],
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(tree_digest(self.package), before)


class ValidationTests(RunnerTestCase):
    """S3/P4: nothing invalid reaches the vendor."""

    def assert_rejected(self, argv, needle, env=None):
        transport = FakeTransport(lambda *_args: self.fail("validation failure reached the network"))
        code, stdout, stderr = self.run_cli(argv, transport=transport, env=env)
        self.assertEqual(code, 1, stderr)
        self.assertIn(needle, stderr)
        self.assertEqual(transport.calls, [])
        return stderr

    def test_unknown_template_is_actionable(self):
        stderr = self.assert_rejected(
            ["--template", "nope", "--output", str(self.output)],
            "unknown template",
        )
        self.assertIn(TEMPLATE, stderr)

    def test_missing_endpoint_mapping_is_actionable(self):
        (self.package / "endpoints.json").write_text("{}", encoding="utf-8")
        self.assert_rejected(
            ["--template", TEMPLATE, "--output", str(self.output)],
            "no endpoint recorded",
        )

    def test_missing_and_malformed_package_files_are_actionable(self):
        cases = [
            ("manifest.json", None, "missing manifest"),
            ("manifest.json", "{not json", "not valid JSON"),
            ("endpoints.json", None, "missing endpoint snapshot"),
            ("workflows/unholy-sdxl.json", None, "missing workflow"),
        ]
        for relative, content, needle in cases:
            with self.subTest(file=relative, content=content):
                package = copy_package(self.root / f"package-{uuid.uuid4().hex}")
                runner = load_runner(package)
                runner.POLL_SECONDS = 0
                target = package / relative
                if content is None:
                    target.unlink()
                else:
                    target.write_text(content, encoding="utf-8")
                transport = FakeTransport(lambda *_args: self.fail("broken package reached the network"))
                runner._http_open = transport
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(runner.OneshotError) as caught:
                        runner.main(["--template", TEMPLATE, "--output", str(self.output)])
                self.assertIn(needle, str(caught.exception))
                self.assertEqual(transport.calls, [])

    def test_invalid_seed_is_rejected(self):
        for seed in ("abc", "1.5", "-1"):
            with self.subTest(seed=seed):
                stderr = self.assert_rejected(
                    ["--template", TEMPLATE, "--output", str(self.output), "--seed", seed],
                    "--seed",
                )
                self.assertNotIn("Traceback", stderr)

    def test_invalid_dimensions_are_rejected(self):
        for flag, value in (("--width", "0"), ("--width", "-8"), ("--width", "7"),
                            ("--height", "20000")):
            with self.subTest(flag=flag, value=value):
                self.assert_rejected(
                    ["--template", TEMPLATE, "--output", str(self.output), flag, value],
                    flag,
                )

    def test_non_integer_dimension_is_rejected_by_the_cli(self):
        transport = FakeTransport(lambda *_args: self.fail("validation failure reached the network"))
        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output), "--height", "1216.5"],
            transport=transport,
        )
        self.assertEqual(code, 2)
        self.assertIn("--height", stderr)
        self.assertEqual(transport.calls, [])

    def test_image_on_a_text_only_template_is_rejected(self):
        image = self.root / "input.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        stderr = self.assert_rejected(
            ["--template", TEMPLATE, "--output", str(self.output), "--image", str(image)],
            "text-only",
        )
        self.assertIn(TEMPLATE, stderr)

    def test_missing_image_file_is_actionable(self):
        self.enable_image_input()
        self.assert_rejected(
            ["--template", TEMPLATE, "--output", str(self.output),
             "--image", str(self.root / "absent.png")],
            "input image not found",
        )

    def test_settings_a_template_does_not_declare_are_rejected(self):
        manifest_path = self.package / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["templates"][TEMPLATE]["size"]
        del manifest["templates"][TEMPLATE]["seed"]
        del manifest["templates"][TEMPLATE]["default_seed_policy"]
        del manifest["templates"][TEMPLATE]["positive"]
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.assert_rejected(
            ["--template", TEMPLATE, "--output", str(self.output), "--width", "832"],
            "declares no size node",
        )
        self.assert_rejected(
            ["--template", TEMPLATE, "--output", str(self.output), "--seed", "42"],
            "declares no seed node",
        )
        self.assert_rejected(
            ["--template", TEMPLATE, "--output", str(self.output), "--prompt", "x"],
            "declares no positive node",
        )


class SubmissionTests(RunnerTestCase):
    """V1/V2: one POST, the same job id, distinct failures, no retries."""

    def test_one_post_then_polls_the_same_job_and_saves_the_output(self):
        statuses = ["IN_QUEUE", "IN_PROGRESS"]
        png = b"\x89PNG\r\n\x1a\n" + b"payload"

        def handler(method, url, body):
            if method == "POST":
                return {"id": "job-1", "status": "IN_QUEUE"}
            if "/status/job-1" in url:
                if statuses:
                    return {"id": "job-1", "status": statuses.pop(0)}
                return {"id": "job-1", "status": "COMPLETED",
                        "output": {"images": [{"filename": "image.png", "data": encode(png)}]}}
            self.fail(f"unexpected request {method} {url}")

        transport = FakeTransport(handler)
        code, stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output), "--wait",
             "--prompt", "a red apple", "--seed", "42"],
            transport=transport,
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(transport.methods().count("POST"), 1)
        status_calls = transport.url_calls("/status/")
        self.assertEqual(len(status_calls), 3)
        self.assertTrue(all(call["url"].endswith("/status/job-1") for call in status_calls))
        self.assertEqual((self.output / "job-1" / "image.png").read_bytes(), png)
        response = json.loads((self.output / "job-1" / "response.json").read_text(encoding="utf-8"))
        self.assertEqual(response["status"], "COMPLETED")
        self.assertIn(str((self.output / "job-1").resolve()), stdout)

    def test_terminal_states_are_reported_distinctly(self):
        messages = {}
        for state, needle in (("FAILED", "failed"), ("CANCELLED", "cancelled"),
                              ("TIMED_OUT", "timed out on the worker")):
            with self.subTest(state=state):
                def handler(method, url, body, state=state):
                    if method == "POST":
                        return {"id": "job-9", "status": "IN_QUEUE"}
                    return {"id": "job-9", "status": state, "error": "boom"}

                code, _stdout, stderr = self.run_cli(
                    ["--template", TEMPLATE, "--output", str(self.output), "--wait"],
                    transport=FakeTransport(handler),
                    env={"RUNPOD_API_KEY": "rpa_test_key"},
                )
                self.assertEqual(code, 1)
                self.assertIn(needle, stderr)
                self.assertIn("job-9", stderr)
                messages[state] = stderr
                self.assertFalse((self.output / "job-9").exists())
        self.assertEqual(len(set(messages.values())), 3)

    def test_a_submitted_job_is_announced_before_polling_starts(self):
        def handler(method, url, body):
            if method == "POST":
                return {"id": "job-8", "status": "IN_QUEUE"}
            return URLError("connection reset")

        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output), "--wait"],
            transport=FakeTransport(handler),
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 1)
        self.assertIn("[info] job job-8 accepted", stderr)
        self.assertIn("network error", stderr)

    def test_an_unusable_post_response_reports_an_unknown_outcome(self):
        cases = {
            "malformed": FakeResponse(b"<html>bad gateway</html>"),
            "not-an-object": [],
            "no-job-id": {"status": "IN_QUEUE"},
        }
        for label, response in cases.items():
            with self.subTest(case=label):
                transport = FakeTransport(lambda *_args, response=response: response)
                code, _stdout, stderr = self.run_cli(
                    ["--template", TEMPLATE, "--output", str(self.output)],
                    transport=transport,
                    env={"RUNPOD_API_KEY": "rpa_test_key"},
                )
                self.assertEqual(code, 1)
                self.assertIn("submission outcome is UNKNOWN", stderr)
                self.assertEqual(transport.methods().count("POST"), 1)

    def test_a_server_error_on_post_reports_an_unknown_outcome(self):
        transport = FakeTransport(
            lambda method, url, body: HTTPError(url, 500, "Server Error", None, None)
        )
        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output)],
            transport=transport,
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 1)
        self.assertIn("HTTP 500", stderr)
        self.assertIn("submission outcome is UNKNOWN", stderr)
        self.assertEqual(transport.methods(), ["POST"])

    def test_wait_deadline_is_distinct_from_worker_timeout(self):
        def handler(method, url, body):
            if method == "POST":
                return {"id": "job-3", "status": "IN_QUEUE"}
            return {"id": "job-3", "status": "IN_PROGRESS"}

        transport = FakeTransport(handler)
        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output), "--wait", "--timeout-s", "1"],
            transport=transport,
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 1)
        self.assertIn("did not finish within 1 s", stderr)
        self.assertEqual(transport.methods().count("POST"), 1)

    def test_auth_failure_is_actionable_and_hides_the_key(self):
        secret = "sk-test-secret-1234567890"

        def handler(method, url, body):
            return HTTPError(url, 401, "Unauthorized", None, None)

        transport = FakeTransport(handler)
        code, stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output)],
            transport=transport,
            env={"RUNPOD_API_KEY": secret},
        )
        self.assertEqual(code, 1)
        self.assertIn("authentication failed", stderr)
        self.assertIn("RUNPOD_API_KEY", stderr)
        self.assertNotIn(secret, stderr + stdout)
        self.assertEqual(transport.methods(), ["POST"])

    def test_malformed_json_response_is_reported(self):
        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output)],
            transport=FakeTransport(lambda method, url, body: FakeResponse(b"<html>bad gateway</html>")),
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 1)
        self.assertIn("malformed JSON", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_uncertain_post_timeout_is_not_resubmitted(self):
        transport = FakeTransport(lambda method, url, body: TimeoutError("timed out"))
        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output)],
            transport=transport,
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 1)
        self.assertEqual(transport.methods().count("POST"), 1)
        self.assertIn("MAY have been accepted", stderr)

    def test_non_object_json_responses_are_reported(self):
        transport = FakeTransport(lambda method, url, body: [])
        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output)],
            transport=transport,
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 1)
        self.assertIn("must be a JSON object", stderr)
        self.assertEqual(transport.methods(), ["POST"])

        def handler(method, url, body):
            if method == "POST":
                return {"id": "job-5", "status": "IN_QUEUE"}
            return []

        transport = FakeTransport(handler)
        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output), "--wait"],
            transport=transport,
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 1)
        self.assertIn("status response for job job-5", stderr)
        self.assertEqual(transport.methods().count("POST"), 1)

    def test_poll_network_errors_retry_but_stay_bounded(self):
        def handler(method, url, body):
            if method == "POST":
                return {"id": "job-4", "status": "IN_QUEUE"}
            return URLError("connection reset")

        transport = FakeTransport(handler)
        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output), "--wait"],
            transport=transport,
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 1)
        self.assertIn("network error", stderr)
        self.assertEqual(transport.methods().count("POST"), 1)
        self.assertEqual(len(transport.url_calls("/status/")), 3)


class RequestLimitTests(RunnerTestCase):
    """V3: whole-envelope caps are enforced before the vendor sees a request."""

    def test_run_cap_rejects_oversized_body_before_http(self):
        prompt = "x" * (11 * 1024 * 1024)
        transport = FakeTransport(lambda *_args: self.fail("oversized body reached the network"))
        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output), "--prompt", prompt],
            transport=transport,
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 1)
        self.assertIn("cap", stderr)
        self.assertIn("10", stderr)
        self.assertEqual(transport.calls, [])

    def test_runsync_uses_the_larger_cap_and_succeeds(self):
        png = b"\x89PNG\r\n\x1a\n" + b"sync-payload"
        transport = FakeTransport(
            lambda method, url, body: self.completed_job(
                "job-sync",
                {"images": [{"filename": "sync.png", "data": encode(png)}]},
            )
        )
        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output), "--sync",
             "--prompt", "x" * (11 * 1024 * 1024)],
            transport=transport,
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(transport.methods(), ["POST"])
        self.assertEqual((self.output / "job-sync" / "sync.png").read_bytes(), png)

    def test_single_image_larger_than_the_cap_is_rejected_before_http(self):
        self.enable_image_input()
        image = self.root / "big.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * (21 * 1024 * 1024))
        transport = FakeTransport(lambda *_args: self.fail("oversized image reached the network"))
        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output), "--sync", "--image", str(image)],
            transport=transport,
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 1)
        self.assertIn("whole request body is capped", stderr)
        self.assertEqual(transport.calls, [])


class OutputTests(RunnerTestCase):
    """V4/A1: artifacts land under the right job; partial failures are visible."""

    def test_data_and_url_outputs_are_saved_under_the_job(self):
        png = b"\x89PNG\r\n\x1a\n" + b"image-bytes"
        video = b"video-bytes"

        def handler(method, url, body):
            if method == "POST":
                return {"id": "job-a", "status": "IN_QUEUE"}
            if "/status/" in url:
                return self.completed_job(
                    "job-a",
                    {
                        "images": [{"filename": "still.png", "data": encode(png)}],
                        "videos": [{"filename": "clip.mp4", "url": "https://cdn.example/clip.mp4?token=x"}],
                    },
                )
            if url.startswith("https://cdn.example/"):
                return FakeResponse(video)
            self.fail(f"unexpected request {method} {url}")

        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output), "--wait"],
            transport=FakeTransport(handler),
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 0, stderr)
        job_dir = self.output / "job-a"
        self.assertEqual((job_dir / "still.png").read_bytes(), png)
        self.assertEqual((job_dir / "clip.mp4").read_bytes(), video)
        self.assertTrue((job_dir / "response.json").is_file())

    def test_completed_without_outputs_is_a_failure(self):
        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output), "--wait"],
            transport=self.submit_and_complete({}),
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 1)
        self.assertIn("no outputs", stderr)
        self.assertTrue((self.output / "job-1" / "response.json").is_file())

    def test_partial_download_failure_is_not_success(self):
        png = b"\x89PNG\r\n\x1a\n" + b"one"

        def handler(method, url, body):
            if method == "POST":
                return {"id": "job-2", "status": "IN_QUEUE"}
            if "/status/" in url:
                return self.completed_job(
                    "job-2",
                    {
                        "images": [
                            {"filename": "ok.png", "data": encode(png)},
                            {"filename": "broken.png", "url": "https://cdn.example/broken.png"},
                        ]
                    },
                )
            return URLError("connection reset")

        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output), "--wait"],
            transport=FakeTransport(handler),
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 1)
        self.assertIn("saved 1 of 2", stderr)
        self.assertIn("broken.png", stderr)
        self.assertTrue((self.output / "job-2" / "ok.png").is_file())
        self.assertFalse((self.output / "job-2" / "broken.png").exists())

    def test_artifact_write_failure_is_reported_as_a_partial_failure(self):
        transport = self.submit_and_complete(
            {"images": [{"filename": "image.png", "data": encode(b"payload")}]}
        )
        original = Path.write_bytes

        def failing_write(path, data):
            if path.name == "image.png":
                raise OSError(28, "No space left on device")
            return original(path, data)

        with mock.patch.object(Path, "write_bytes", failing_write):
            code, _stdout, stderr = self.run_cli(
                ["--template", TEMPLATE, "--output", str(self.output), "--wait"],
                transport=transport,
                env={"RUNPOD_API_KEY": "rpa_test_key"},
            )
        self.assertEqual(code, 1)
        self.assertIn("No space left on device", stderr)
        self.assertIn("saved 0 of 1", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_wrapped_base64_is_accepted(self):
        png = b"\x89PNG\r\n\x1a\n" + b"wrapped-payload" * 8
        wrapped = "\n".join(
            encode(png)[index:index + 4] for index in range(0, len(encode(png)), 4)
        )
        transport = self.submit_and_complete(
            {"images": [{"filename": "wrapped.png", "data": wrapped}]},
            job_id="job-wrap",
        )
        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output), "--wait"],
            transport=transport,
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual((self.output / "job-wrap" / "wrapped.png").read_bytes(), png)

    def test_unrecognized_empty_and_corrupt_items_are_reported(self):
        transport = self.submit_and_complete(
            {
                "images": [
                    {"filename": "unknown.png"},
                    {"filename": "empty.png", "data": ""},
                    {"filename": "blank.png", "data": "data:image/png;base64,"},
                    {"filename": "corrupt.png", "data": "!!!not-base64!!!"},
                ]
            },
            job_id="job-3",
        )
        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output), "--wait"],
            transport=transport,
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 1)
        self.assertIn("saved 0 of 4", stderr)
        for name in ("unknown.png", "empty.png", "blank.png", "corrupt.png"):
            self.assertIn(name, stderr)
        self.assertTrue((self.output / "job-3" / "response.json").is_file())

    def test_two_runs_keep_workflow_state_and_artifacts_separate(self):
        png = b"\x89PNG\r\n\x1a\n" + b"x"
        posted_prompts = []

        def handler(method, url, body):
            if method == "POST":
                posted_prompts.append(body["input"]["workflow"]["2"]["inputs"]["text"])
                job_id = "job-first" if len(posted_prompts) == 1 else "job-second"
                return {"id": job_id, "status": "COMPLETED",
                        "output": {"images": [{"filename": "out.png", "data": encode(png)}]}}
            self.fail(f"unexpected request {method} {url}")

        transport = FakeTransport(handler)
        for prompt in ("first prompt", "second prompt"):
            code, _stdout, stderr = self.run_cli(
                ["--template", TEMPLATE, "--output", str(self.output), "--sync", "--prompt", prompt],
                transport=transport,
                env={"RUNPOD_API_KEY": "rpa_test_key"},
            )
            self.assertEqual(code, 0, stderr)
        self.assertEqual(posted_prompts, ["first prompt", "second prompt"])
        self.assertTrue((self.output / "job-first" / "out.png").is_file())
        self.assertTrue((self.output / "job-second" / "out.png").is_file())

    def test_submission_does_not_modify_the_package(self):
        before = tree_digest(self.package)
        code, _stdout, stderr = self.run_cli(
            ["--template", TEMPLATE, "--output", str(self.output), "--wait"],
            transport=self.submit_and_complete(
                {"images": [{"filename": "a.png", "data": encode(b"\x89PNG\r\n\x1a\n")}]}
            ),
            env={"RUNPOD_API_KEY": "rpa_test_key"},
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(tree_digest(self.package), before)


if __name__ == "__main__":
    unittest.main()

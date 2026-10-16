import base64
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import rp_common  # noqa: E402  (path set up above)
import rp_endpoints  # noqa: E402
import rp_images  # noqa: E402
import rp_submit  # noqa: E402


def fake_docker(tmpdir, script):
    """Put an executable `docker` on PATH so run_docker() can be tested offline."""
    path = Path(tmpdir) / "docker"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


class RunDockerTests(unittest.TestCase):
    def test_uncaptured_success_returns_empty_string(self):
        # build/push call run_docker(capture=False), where subprocess leaves stdout=None;
        # stripping it raised AttributeError after an otherwise successful docker run.
        with tempfile.TemporaryDirectory() as tmp:
            fake_docker(tmp, "#!/bin/sh\nexit 0\n")
            with mock.patch.dict(os.environ, {"PATH": tmp}):
                self.assertEqual(
                    rp_images.run_docker(["build", "."], capture=False), ""
                )

    def test_captured_success_returns_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_docker(tmp, "#!/bin/sh\necho sha256:abc\n")
            with mock.patch.dict(os.environ, {"PATH": tmp}):
                self.assertEqual(
                    rp_images.run_docker(["image", "inspect", "x"]), "sha256:abc"
                )

    def test_failure_raises_config_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_docker(tmp, "#!/bin/sh\necho boom >&2\nexit 1\n")
            with mock.patch.dict(os.environ, {"PATH": tmp}):
                with self.assertRaises(rp_common.ConfigError) as ctx:
                    rp_images.run_docker(["push", "x"], capture=False)
            # Output is inherited, so it stays live in the terminal; the message names the command.
            self.assertIn("docker push x failed", str(ctx.exception))

    def test_captured_failure_includes_the_output_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_docker(tmp, "#!/bin/sh\necho boom >&2\nexit 1\n")
            with mock.patch.dict(os.environ, {"PATH": tmp}):
                with self.assertRaises(rp_common.ConfigError) as ctx:
                    rp_images.run_docker(["push", "x"])
            self.assertIn("boom", str(ctx.exception))


class DryRunResolutionTests(unittest.TestCase):
    def test_unregistered_target_never_calls_the_api(self):
        def no_network(*args, **kwargs):
            raise AssertionError(f"network call attempted: {args[:2]}")

        state = {"endpoints": {"ep1": {"name": "comfyui-video-wan"}}}
        with mock.patch.object(rp_submit, "api_request", no_network), mock.patch.object(
            rp_submit, "api_key", lambda config, required=True: "test-key"
        ):
            self.assertEqual(
                rp_submit.resolve_endpoint("ep1", state, {}, allow_network=False), "ep1"
            )
            self.assertEqual(
                rp_submit.resolve_endpoint(
                    "comfyui-video-wan", state, {}, allow_network=False
                ),
                "ep1",
            )
            self.assertEqual(
                rp_submit.resolve_endpoint(
                    "ep-unknown", state, {}, allow_network=False
                ),
                "ep-unknown",
            )

    def test_live_fallback_still_resolves_when_allowed(self):
        calls = []

        def live(method, url, payload=None, key=None, timeout=120):
            calls.append((method, url))
            return [{"id": "ep9", "name": "comfyui-sdxl"}]

        with mock.patch.object(rp_submit, "api_request", live), mock.patch.object(
            rp_submit, "api_key", lambda config, required=True: "test-key"
        ):
            self.assertEqual(rp_submit.resolve_endpoint("comfyui-sdxl", {}, {}), "ep9")
        self.assertEqual([call[0] for call in calls], ["GET"])


class RequestSizeTests(unittest.TestCase):
    def test_run_limit_rejects_an_oversize_envelope(self):
        envelope = {
            "input": {"workflow": {"pad": "x" * rp_submit.MAX_RUN_REQUEST_BYTES}}
        }
        with self.assertRaises(rp_common.ConfigError) as ctx:
            rp_submit.check_request_size(envelope, sync=False)
        self.assertIn("/run", str(ctx.exception))

    def test_runsync_limit_allows_what_run_rejects(self):
        envelope = {"input": {"workflow": {"pad": "x" * (11 * 1024 * 1024)}}}
        size = rp_submit.check_request_size(envelope, sync=True)
        self.assertGreater(size, rp_submit.MAX_RUN_REQUEST_BYTES)
        with self.assertRaises(rp_common.ConfigError):
            rp_submit.check_request_size(envelope, sync=False)

    def test_many_small_images_overflow_the_shared_limit(self):
        # Each entry is ~4 MB, under the 10 MB limit on its own; three together are not.
        image = "data:image/png;base64," + "A" * (4 * 1024 * 1024)
        envelope = rp_submit.build_envelope(
            {"1": {"class_type": "X", "inputs": {}}},
            [{"name": f"{index}.png", "image": image} for index in range(3)],
        )
        with self.assertRaises(rp_common.ConfigError) as ctx:
            rp_submit.check_request_size(envelope, sync=False)
        self.assertIn("/run", str(ctx.exception))

    def test_preliminary_file_guard_rejects_a_file_that_cannot_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.png"
            path.write_bytes(b"\x89PNG" + b"0" * (8 * 1024 * 1024))
            with self.assertRaises(rp_common.ConfigError) as ctx:
                rp_submit.encode_image(path, rp_submit.MAX_RUN_REQUEST_BYTES, "/run")
            self.assertIn("/run", str(ctx.exception))
            encoded = rp_submit.encode_image(
                path, rp_submit.MAX_RUNSYNC_REQUEST_BYTES, "/runsync"
            )
            self.assertEqual(encoded["name"], "big.png")
            self.assertTrue(encoded["image"].startswith("data:image/png;base64,"))


class OutputShapeTests(unittest.TestCase):
    def test_s3_url_response_is_a_url_not_base64(self):
        item = {
            "filename": "out.png",
            "type": "s3_url",
            "data": "https://example.invalid/out.png",
        }
        self.assertEqual(
            rp_submit.describe_output(item),
            {"filename": "out.png", "url": "https://example.invalid/out.png"},
        )

    def test_base64_response_is_decoded(self):
        item = {
            "filename": "out.png",
            "type": "base64",
            "data": base64.b64encode(b"png-bytes").decode("ascii"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            written, failed = rp_submit.save_outputs({"images": [item]}, Path(tmp))
            self.assertEqual(failed, [])
            self.assertEqual(Path(written[0]).read_bytes(), b"png-bytes")

    def test_s3_url_response_is_downloaded(self):
        # A file:// URL keeps the test offline while exercising the documented s3_url path.
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "remote.bin"
            source.write_bytes(b"remote-bytes")
            output = {
                "images": [
                    {"filename": "out.png", "type": "s3_url", "data": source.as_uri()}
                ]
            }
            written, failed = rp_submit.save_outputs(output, Path(tmp) / "out")
            self.assertEqual(failed, [])
            self.assertEqual(Path(written[0]).read_bytes(), b"remote-bytes")

    def test_undecodable_base64_is_reported_as_failed(self):
        output = {
            "images": [{"filename": "out.png", "type": "base64", "data": "not*base64"}]
        }
        with tempfile.TemporaryDirectory() as tmp:
            written, failed = rp_submit.save_outputs(output, Path(tmp))
            self.assertEqual(written, [])
            self.assertEqual([path.name for path in failed], ["out.png"])

    def test_failed_download_is_reported_as_failed(self):
        output = {
            "images": [
                {
                    "filename": "out.png",
                    "type": "s3_url",
                    "data": "https://127.0.0.1:9/x",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            written, failed = rp_submit.save_outputs(output, Path(tmp))
            self.assertEqual(written, [])
            self.assertEqual([path.name for path in failed], ["out.png"])

    def test_unrecognized_output_yields_no_files_and_no_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            written, failed = rp_submit.save_outputs({"message": "done"}, Path(tmp))
        self.assertEqual((written, failed), ([], []))


class RegistryTests(unittest.TestCase):
    def test_save_state_replaces_atomically_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "images.json"
            registry.write_text('{"images": {"old": {}}}', encoding="utf-8")
            rp_common.save_state(registry, {"images": {"new": {}}})
            self.assertEqual(
                sorted(json.loads(registry.read_text())["images"]), ["new"]
            )
            self.assertEqual(
                [path.name for path in Path(tmp).iterdir()], ["images.json"]
            )

    def test_transaction_merges_concurrent_writers(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "endpoints.json"

            def writer(name):
                with rp_common.state_transaction(registry) as state:
                    time.sleep(0.05)  # the read-modify-write window
                    state.setdefault("endpoints", {})[name] = {"name": name}

            threads = [
                threading.Thread(target=writer, args=(name,)) for name in ("a", "b")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            saved = json.loads(registry.read_text())["endpoints"]
        self.assertEqual(sorted(saved), ["a", "b"])

    def test_transaction_discards_changes_when_the_block_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "endpoints.json"
            with self.assertRaises(RuntimeError):
                with rp_common.state_transaction(registry) as state:
                    state["endpoints"] = {"ep1": {"status": "active"}}
                    raise RuntimeError("repair failed")
            self.assertFalse(registry.exists())


class ImageTagTests(unittest.TestCase):
    def test_default_endpoint_name_reuses_the_repo_name(self):
        self.assertEqual(
            rp_endpoints.default_endpoint_name("ns/comfyui-video-wan"),
            "comfyui-video-wan",
        )
        self.assertEqual(
            rp_endpoints.default_endpoint_name("ns/worker"), "comfyui-worker"
        )

    def test_create_rejects_a_non_semantic_tag_before_any_api_call(self):
        args = SimpleNamespace(image_tag="ns/comfyui-video-wan:latest", name=None)
        with mock.patch.object(
            rp_endpoints, "api_key", lambda config, required=True: "test-key"
        ):
            with self.assertRaises(rp_common.ConfigError):
                rp_endpoints.create(args, {}, {})

    def test_create_rejects_an_untagged_image(self):
        args = SimpleNamespace(image_tag="ns/comfyui-video-wan", name=None)
        with self.assertRaises(rp_common.ConfigError):
            rp_endpoints.create(args, {}, {})

    def test_push_rejects_a_non_semantic_version_before_docker(self):
        state = {
            "images": {"images-sdxl": {"repo": "ns/comfyui-sdxl", "version": "v1.0.0"}}
        }
        args = SimpleNamespace(spec="images-sdxl", version="latest")
        with mock.patch.object(
            rp_images, "run_docker", lambda *a, **k: self.fail("docker should not run")
        ):
            with self.assertRaises(rp_common.ConfigError):
                rp_images.push(args, {}, state)

    def test_push_rejects_a_non_semantic_recorded_version(self):
        state = {
            "images": {"images-sdxl": {"repo": "ns/comfyui-sdxl", "version": "latest"}}
        }
        args = SimpleNamespace(spec="images-sdxl", version=None)
        with self.assertRaises(rp_common.ConfigError):
            rp_images.push(args, {}, state)


class SyncTests(unittest.TestCase):
    def sync(self, rows, respond):
        """Run rp_endpoints.sync() against stubbed responses; return (exit, rows, calls)."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "endpoints.json"
            registry.write_text(json.dumps({"endpoints": rows}), encoding="utf-8")
            calls = []

            def fake_request(method, url, payload=None, key=None, timeout=120):
                calls.append((method, url, payload))
                result = respond(method, url, payload)
                if isinstance(result, Exception):
                    raise result
                return result

            with mock.patch.object(
                rp_endpoints, "ENDPOINTS_STATE", registry
            ), mock.patch.object(
                rp_endpoints, "api_request", fake_request
            ), mock.patch.object(
                rp_endpoints, "api_key", lambda config, required=True: "k"
            ):
                code = rp_endpoints.sync(None, {}, {})
            return code, json.loads(registry.read_text())["endpoints"], calls

    ROWS = {"ep1": {"name": "comfyui-video-wan", "workers_max": 1, "status": "active"}}

    def test_only_a_matching_read_is_active(self):
        code, rows, calls = self.sync(
            self.ROWS, lambda method, url, payload: [{"id": "ep1", "workersMax": 1}]
        )
        self.assertEqual(code, 0)
        self.assertEqual(rows["ep1"]["status"], "active")
        self.assertEqual([call[0] for call in calls], ["GET"])

    def test_repair_is_confirmed_by_a_read_back(self):
        def respond(method, url, payload):
            if method == "GET" and url.endswith("/endpoints"):
                return [{"id": "ep1", "workersMax": 0}]
            if method == "PATCH":
                return {}
            return {"id": "ep1", "workersMax": 1}

        code, rows, calls = self.sync(self.ROWS, respond)
        self.assertEqual(code, 0)
        self.assertEqual(rows["ep1"]["status"], "active")
        self.assertEqual([call[0] for call in calls], ["GET", "PATCH", "GET"])
        self.assertIn("/v2/serverless/ep1", calls[1][1])
        self.assertEqual(calls[1][2], {"workers": {"max": 1}})

    def test_unconfirmed_repair_stays_drifted_and_exits_nonzero(self):
        def respond(method, url, payload):
            if method == "PATCH":
                return {}
            if url.endswith("/endpoints"):
                return [{"id": "ep1", "workersMax": 0}]
            return {"id": "ep1", "workersMax": 0}

        code, rows, _calls = self.sync(self.ROWS, respond)
        self.assertEqual(code, 1)
        self.assertEqual(rows["ep1"]["status"], "drifted")
        self.assertIn("read-back", rows["ep1"]["drift_note"])

    def test_refused_repair_stays_drifted_and_exits_nonzero(self):
        def respond(method, url, payload):
            if method == "PATCH":
                return rp_common.ConfigError("PATCH ... -> HTTP 405")
            return [{"id": "ep1", "workersMax": 0}]

        code, rows, calls = self.sync(self.ROWS, respond)
        self.assertEqual(code, 1)
        self.assertEqual(rows["ep1"]["status"], "drifted")
        self.assertEqual([call[0] for call in calls], ["GET", "PATCH"])

    def test_unreadable_workers_max_is_unknown_not_active(self):
        responses = iter([[{"id": "ep1"}], {"id": "ep1"}])
        code, rows, calls = self.sync(
            self.ROWS, lambda method, url, payload: next(responses)
        )
        self.assertEqual(code, 1)
        self.assertEqual(rows["ep1"]["status"], "unknown")
        self.assertEqual([call[0] for call in calls], ["GET", "GET"])


if __name__ == "__main__":
    unittest.main()

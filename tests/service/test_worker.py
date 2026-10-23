"""Worker results, validated opaque artifact access and failure retention."""

import errno
import hashlib
import os
import sqlite3
import subprocess
import tempfile
import time
import unittest
import uuid
from unittest import mock

from distribute_agent.bridge import submit_plan
from distribute_agent.service.runtime import MediaRuntime
from distribute_agent.service.types import (
    ARTIFACT_CORRUPT,
    ARTIFACT_TOO_LARGE,
    INVALID_OUTPUT,
    STORAGE_ERROR,
    SUBMISSION_FAILED,
    SUBMISSION_UNCERTAIN,
    UNKNOWN_ARTIFACT,
    UNKNOWN_JOB,
    ServiceError,
)

from service_test_support import (
    FakeSubmission,
    PNG_BYTES,
    TEST_KEY,
    VALID_PLAN,
    base_config,
    build_png,
    plan_reply,
    scripted_factory,
    specs,
    wait_for_job,
)


TERMINAL = {"succeeded", "failed", "needs_attention"}


class WorkerTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = base_config(self.temporary.name)
        self.env = {"RUNPOD_API_KEY": TEST_KEY}

    def start(self, *, fake=None, config=None):
        fake = fake or FakeSubmission()
        factory, _ = scripted_factory([plan_reply(VALID_PLAN)])
        runtime = MediaRuntime(
            config or self.config,
            submission=fake,
            env=self.env,
            model_factory=factory,
        )
        runtime.start()
        self.addCleanup(self._stop, runtime, fake)
        return runtime, fake

    def _stop(self, runtime, fake):
        fake.release.set()
        runtime.stop(timeout=10)

    def submit(self, runtime, request_id="req-1", **overrides):
        return runtime.start_generation("plan one", specs(request_id, **overrides))


class OutcomeTests(WorkerTestCase):
    def run_outcome(self, behavior):
        """Run one job in its own state directory and return plain data."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        config = base_config(temporary.name)
        fake = FakeSubmission(behavior=behavior)
        factory, _ = scripted_factory([plan_reply(VALID_PLAN)])
        runtime = MediaRuntime(
            config, submission=fake, env=self.env, model_factory=factory
        )
        runtime.start()
        try:
            view = runtime.start_generation("plan one", specs())
            done = wait_for_job(runtime, view.job_id, TERMINAL, timeout=20)
            return done.as_dict(), list(fake.calls)
        finally:
            fake.release.set()
            runtime.stop(timeout=10)

    def test_success_stores_every_validated_image_and_ignores_other_files(self):
        done, calls = self.run_outcome("two_images")
        self.assertEqual(done["status"], "succeeded", done)
        self.assertEqual(done["vendor_job_id"], "vendor-1")
        self.assertEqual(len(done["artifacts"]), 2)
        self.assertEqual(len(calls), 1)
        dimensions = {(item["width"], item["height"]) for item in done["artifacts"]}
        self.assertEqual(dimensions, {(3, 5), (7, 2)})
        self.assertEqual(
            {item["mime_type"] for item in done["artifacts"]}, {"image/png"}
        )
        for item in done["artifacts"]:
            self.assertEqual(len(item["artifact_id"]), 32)
            self.assertEqual(len(item["sha256"]), 64)
            self.assertGreater(item["size_bytes"], 0)
            self.assertNotIn("path", item)
            self.assertNotIn("filename", item)

    def test_invalid_or_missing_output_cannot_succeed(self):
        failures = (
            "corrupt",
            "partial",
            "empty_file",
            "oversized",
            "symlink",
            "one_valid_one_corrupt",
            "no_output",
        )
        for behavior in failures:
            with self.subTest(behavior=behavior):
                done, calls = self.run_outcome(behavior)
                self.assertEqual(done["status"], "failed", done)
                self.assertEqual(done["error"]["code"], INVALID_OUTPUT)
                self.assertEqual(done["artifacts"], [])
                self.assertEqual(done["vendor_job_id"], "vendor-1")
                self.assertEqual(len(calls), 1)
                self.assertNotIn("/home", done["error"]["message"])

    def test_success_without_a_vendor_id_or_inside_the_job_is_uncertain(self):
        for behavior in ("no_vendor_id", "outside_output"):
            with self.subTest(behavior=behavior):
                done, calls = self.run_outcome(behavior)
                self.assertEqual(done["status"], "needs_attention", done)
                self.assertEqual(done["error"]["code"], SUBMISSION_UNCERTAIN)
                self.assertTrue(done["error"]["may_have_submitted"])
                self.assertEqual(done["artifacts"], [])
                self.assertEqual(len(calls), 1)

    def test_ambiguous_bridge_error_retains_vendor_id_and_is_not_retried(self):
        done, calls = self.run_outcome("raise_uncertain")
        self.assertEqual(done["status"], "needs_attention", done)
        self.assertEqual(done["error"]["code"], SUBMISSION_UNCERTAIN)
        self.assertTrue(done["error"]["may_have_submitted"])
        self.assertEqual(done["vendor_job_id"], "vendor-77")
        self.assertEqual(done["artifacts"], [])
        message = done["error"]["message"]
        self.assertNotIn(TEST_KEY, message)
        self.assertNotIn("/home/user/private", message)
        self.assertNotIn("vendor_private_body", message)
        self.assertNotIn("DO_NOT_PUBLISH_123", message)
        self.assertLessEqual(len(message), 400)
        self.assertEqual(len(calls), 1)
        time.sleep(0.4)
        self.assertEqual(len(calls), 1)

    def test_definite_bridge_error_fails_without_a_vendor_id(self):
        done, calls = self.run_outcome("raise_definite")
        self.assertEqual(done["status"], "failed", done)
        self.assertEqual(done["error"]["code"], SUBMISSION_FAILED)
        self.assertFalse(done["error"]["may_have_submitted"])
        self.assertIsNone(done["vendor_job_id"])
        self.assertEqual(len(calls), 1)

    def test_unexpected_submission_exception_is_uncertain_and_not_retried(self):
        done, calls = self.run_outcome("raise_unexpected")
        self.assertEqual(done["status"], "needs_attention", done)
        self.assertEqual(done["error"]["code"], SUBMISSION_UNCERTAIN)
        self.assertTrue(done["error"]["may_have_submitted"])
        self.assertNotIn("/home/user/private", done["error"]["message"])
        self.assertEqual(len(calls), 1)
        time.sleep(0.4)
        self.assertEqual(len(calls), 1)

    def test_a_timed_out_bridge_keeps_the_accepted_vendor_id(self):
        factory, _ = scripted_factory([plan_reply(VALID_PLAN)])
        expired = subprocess.TimeoutExpired(
            cmd="runner",
            timeout=5,
            stderr=b"[info] job vendor-timeout accepted\n",
        )

        def timeout_submission(package, plan, **kwargs):
            with mock.patch(
                "distribute_agent.bridge.subprocess.run", side_effect=expired
            ):
                return submit_plan(package, plan, **kwargs)

        runtime = MediaRuntime(
            self.config, env=self.env, model_factory=factory, submission=timeout_submission
        )
        runtime.start()
        self.addCleanup(runtime.stop, 10)
        view = self.submit(runtime, request_id="req-timeout")
        done = wait_for_job(runtime, view.job_id, {"needs_attention"}, timeout=20)
        self.assertEqual(done.error["code"], SUBMISSION_UNCERTAIN)
        self.assertTrue(done.error["may_have_submitted"])
        self.assertEqual(done.vendor_job_id, "vendor-timeout")
        self.assertEqual(done.artifacts, ())

    def test_disk_exhaustion_during_artifact_read_is_a_storage_failure(self):
        runtime, fake = self.start()
        with mock.patch(
            "distribute_agent.service.runtime._read_artifact_bytes",
            side_effect=OSError(errno.ENOSPC, "No space left on device"),
        ):
            view = self.submit(runtime)
            done = wait_for_job(runtime, view.job_id, TERMINAL, timeout=20)
        self.assertEqual(done.status, "failed")
        self.assertEqual(done.error["code"], STORAGE_ERROR)
        self.assertIn("filesystem is full", done.error["message"])
        self.assertEqual(done.vendor_job_id, "vendor-1")
        self.assertEqual(done.artifacts, ())

    def test_unknown_job_has_a_stable_bounded_error_without_echoing_the_id(self):
        runtime, _ = self.start()
        for job_id in ("missing", TEST_KEY + "X" * 1000):
            with self.subTest(length=len(job_id)):
                with self.assertRaises(ServiceError) as caught:
                    runtime.get_generation(job_id)
                self.assertEqual(caught.exception.code, UNKNOWN_JOB)
                self.assertNotIn(TEST_KEY, caught.exception.message)
                self.assertLessEqual(len(caught.exception.message), 400)

    def test_job_id_is_a_uuid_and_timestamps_are_recorded(self):
        runtime, _ = self.start()
        view = self.submit(runtime)
        done = wait_for_job(runtime, view.job_id, {"succeeded"})
        uuid.UUID(done.job_id)
        self.assertTrue(done.created_at)
        self.assertTrue(done.updated_at)
        self.assertTrue(done.started_at)
        self.assertTrue(done.finished_at)
        self.assertEqual(done.request_id, "req-1")
        self.assertEqual(done.plan, VALID_PLAN)


    def test_planning_diagnostics_are_persisted_and_read_back(self):
        runtime, _ = self.start()
        view = self.submit(runtime)
        expected = {
            "duration_seconds",
            "model_requests",
            "agent_steps",
            "validated_previews",
            "state",
        }
        self.assertEqual(set(view.diagnostics), expected)
        self.assertGreaterEqual(view.diagnostics["model_requests"], 1)
        done = wait_for_job(runtime, view.job_id, {"succeeded"})
        self.assertEqual(runtime.get_generation(done.job_id).diagnostics, done.diagnostics)
        self.assertEqual(done.as_dict()["diagnostics"], done.diagnostics)


class ArtifactAccessTests(WorkerTestCase):
    def setUp(self):
        super().setUp()
        self.runtime, self.fake = self.start(fake=FakeSubmission(behavior="two_images"))
        self.view = self.submit(self.runtime)
        self.view = wait_for_job(self.runtime, self.view.job_id, {"succeeded"})
        self.artifact = next(
            item
            for item in self.view.artifacts
            if (item.width, item.height) == (3, 5)
        )
        self.artifact_path = self.fake.calls[0]["output_dir"] / "vendor-1" / "a.png"

    def test_read_returns_the_recorded_bytes_and_metadata(self):
        content = self.runtime.read_artifact(self.view.job_id, self.artifact.artifact_id)
        self.assertEqual(content.data, build_png(3, 5))
        self.assertEqual(content.metadata.sha256, hashlib.sha256(content.data).hexdigest())
        self.assertEqual(content.metadata.size_bytes, len(content.data))
        self.assertEqual((content.metadata.width, content.metadata.height), (3, 5))

    def test_unknown_job_and_artifact_are_indistinguishable(self):
        for job_id, artifact_id in (
            ("missing", "missing"),
            (self.view.job_id, "missing"),
            ("../escape", self.artifact.artifact_id),
        ):
            with self.subTest(job_id=job_id, artifact_id=artifact_id):
                with self.assertRaises(ServiceError) as caught:
                    self.runtime.read_artifact(job_id, artifact_id)
                self.assertEqual(caught.exception.code, UNKNOWN_ARTIFACT)

    def test_inline_read_is_capped(self):
        with self.assertRaises(ServiceError) as caught:
            self.runtime.read_artifact(
                self.view.job_id, self.artifact.artifact_id, max_bytes=1
            )
        self.assertEqual(caught.exception.code, ARTIFACT_TOO_LARGE)

    def test_a_symlink_cannot_redirect_the_read(self):
        outside = self.config.output_dir / "outside.png"
        outside.write_bytes(PNG_BYTES)
        self.artifact_path.unlink()
        self.artifact_path.symlink_to(outside)
        with self.assertRaises(ServiceError) as caught:
            self.runtime.read_artifact(self.view.job_id, self.artifact.artifact_id)
        self.assertEqual(caught.exception.code, ARTIFACT_CORRUPT)

    def test_a_traversal_relative_path_is_refused(self):
        outside = self.config.output_dir / "outside.png"
        outside.write_bytes(PNG_BYTES)
        self._set_relative_path("../outside.png")
        with self.assertRaises(ServiceError) as caught:
            self.runtime.read_artifact(self.view.job_id, self.artifact.artifact_id)
        self.assertEqual(caught.exception.code, ARTIFACT_CORRUPT)

    def test_bytes_changed_after_recording_are_refused(self):
        original = self.artifact_path.read_bytes()
        mutated = bytearray(original)
        mutated[-1] ^= 0xFF
        self.artifact_path.write_bytes(bytes(mutated))
        with self.assertRaises(ServiceError) as caught:
            self.runtime.read_artifact(self.view.job_id, self.artifact.artifact_id)
        self.assertEqual(caught.exception.code, ARTIFACT_CORRUPT)

    def test_a_missing_artifact_is_refused(self):
        self.artifact_path.unlink()
        with self.assertRaises(ServiceError) as caught:
            self.runtime.read_artifact(self.view.job_id, self.artifact.artifact_id)
        self.assertEqual(caught.exception.code, ARTIFACT_CORRUPT)

    def test_a_fifo_at_the_artifact_path_cannot_block_the_read(self):
        self.artifact_path.unlink()
        os.mkfifo(self.artifact_path)
        with self.assertRaises(ServiceError) as caught:
            self.runtime.read_artifact(self.view.job_id, self.artifact.artifact_id)
        self.assertEqual(caught.exception.code, ARTIFACT_CORRUPT)

    def test_the_wire_view_has_no_paths_payloads_or_credentials(self):
        wire = self.view.as_dict()
        self.assertEqual(wire["error"], None)
        self.assertEqual(
            set(wire),
            {
                "job_id",
                "request_id",
                "status",
                "plan",
                "package_digest",
                "vendor_job_id",
                "artifacts",
                "error",
                "diagnostics",
                "created_at",
                "updated_at",
                "started_at",
                "finished_at",
            },
        )
        self.assertNotIn("output_dir", wire)
        self.assertNotIn("stdout", wire)
        self.assertNotIn("stderr", wire)
        self.assertNotIn(TEST_KEY, repr(wire))

    def _set_relative_path(self, relative):
        connection = sqlite3.connect(self.config.database_path)
        try:
            connection.execute(
                "UPDATE artifacts SET relative_path = ? WHERE artifact_id = ?",
                (relative, self.artifact.artifact_id),
            )
            connection.commit()
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()

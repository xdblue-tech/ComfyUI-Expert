"""Runtime lifecycle: planning, admission, snapshots, stop and the OS lock."""

import sqlite3
import subprocess
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest import mock

from distribute_agent.service.jobs import JobStore
from smolagents.models import ChatMessage, MessageRole

from distribute_agent.scripted_model import ScriptedModel
from distribute_agent.service.runtime import MediaRuntime
from distribute_agent.service.types import (
    BUSY,
    CAPACITY,
    GENERATION_DISABLED,
    INTERRUPTED,
    INVALID_INPUT,
    PLAN_FAILED,
    REQUEST_CONFLICT,
    SHUTTING_DOWN,
    SNAPSHOT_UNAVAILABLE,
    STALE_DIGEST,
    STATE_LOCKED,
    RUNTIME_ERROR,
    ServiceError,
)

from service_test_support import (
    BlockingScriptedModel,
    FakeSubmission,
    TEST_KEY,
    VALID_PLAN,
    base_config,
    plan_reply,
    read_manifest,
    scripted_factory,
    wait_for_job,
    wait_until,
    write_manifest,
)


class RuntimeTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = base_config(self.temporary.name)
        self.fake = FakeSubmission()
        self.env = {"RUNPOD_API_KEY": TEST_KEY}

    def start_runtime(self, *, config=None, fake=None, factory=None, env=None):
        runtime = MediaRuntime(
            config or self.config,
            submission=fake or self.fake,
            env=self.env if env is None else env,
            model_factory=factory,
        )
        runtime.start()
        self.addCleanup(self._stop, runtime, fake or self.fake)
        return runtime

    def _stop(self, runtime, fake):
        fake.release.set()
        runtime.stop(timeout=10)

    def plain_factory(self, plan=VALID_PLAN):
        return scripted_factory([plan_reply(plan)])[0]

    def job_count(self):
        connection = sqlite3.connect(self.config.database_path)
        try:
            return connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        finally:
            connection.close()


class PlanningTests(RuntimeTestCase):
    def test_plan_returns_a_strict_plan_and_bounded_diagnostics(self):
        factory, models = scripted_factory([plan_reply(VALID_PLAN)])
        runtime = self.start_runtime(factory=factory)
        result = runtime.plan("make one portrait")
        self.assertEqual(result.plan, VALID_PLAN)
        self.assertEqual(len(result.package_digest), 64)
        self.assertEqual(result.model_id, "scripted-deterministic")
        self.assertEqual(
            set(result.diagnostics),
            {
                "duration_seconds",
                "model_requests",
                "agent_steps",
                "validated_previews",
                "state",
            },
        )
        self.assertGreaterEqual(result.diagnostics["model_requests"], 1)
        self.assertNotIn("sandbox", result.diagnostics)
        self.assertNotIn("last_code", result.diagnostics)
        self.assertEqual(len(models), 1)
        self.assertEqual(list(self.config.workspace_dir.iterdir()), [])

    def test_default_model_factory_uses_the_tested_openai_compatible_client(self):
        runtime = MediaRuntime(
            self.config,
            submission=self.fake,
            env={"MEDIAGEN_PROXY_API_KEY": "hf_dummy_proxy_key"},
        )
        model = runtime._create_model()
        self.assertEqual(model.client_kwargs["model"], self.config.model_base_url)
        self.assertIsNone(model.client_kwargs["base_url"])
        self.assertEqual(model.client_kwargs["timeout"], self.config.model_timeout_seconds)
        self.assertEqual(model.kwargs["model"], self.config.model_id)
        self.assertEqual(model.kwargs["max_tokens"], self.config.model_max_tokens)
        self.assertEqual(model.kwargs["temperature"], self.config.model_temperature)
        self.assertEqual(model.retryer.max_attempts, 1)
        completion = model._prepare_completion_kwargs(
            messages=[ChatMessage(role=MessageRole.USER, content="hello")]
        )
        self.assertEqual(completion["model"], self.config.model_id)
        self.assertEqual(completion["max_tokens"], self.config.model_max_tokens)

    def test_plan_rejects_empty_and_oversized_requests_without_a_model_call(self):
        factory, models = scripted_factory([plan_reply(VALID_PLAN)])
        runtime = self.start_runtime(factory=factory)
        for request in ("", "   ", "x" * 8193):
            with self.subTest(length=len(request)):
                with self.assertRaises(ServiceError) as caught:
                    runtime.plan(request)
                self.assertEqual(caught.exception.code, INVALID_INPUT)
        self.assertEqual(models, [])

    def test_only_one_planner_runs_and_admission_is_released_after_completion(self):
        blocking = BlockingScriptedModel([plan_reply(VALID_PLAN)])
        models = [blocking, ScriptedModel([plan_reply(VALID_PLAN)])]

        def factory():
            return models.pop(0)

        runtime = self.start_runtime(factory=factory)
        outcome = {}

        def run_plan():
            try:
                outcome["result"] = runtime.plan("first")
            except Exception as exc:  # recorded for the assertion
                outcome["error"] = exc

        thread = threading.Thread(target=run_plan)
        thread.start()
        self.assertTrue(blocking.started.wait(10))
        with self.assertRaises(ServiceError) as caught:
            runtime.plan("second while busy")
        self.assertEqual(caught.exception.code, BUSY)
        blocking.release.set()
        thread.join(20)
        self.assertIn("result", outcome, outcome)
        self.assertEqual(outcome["result"].plan, VALID_PLAN)
        second = runtime.plan("third after release")
        self.assertEqual(second.plan, VALID_PLAN)

    def test_plan_failure_releases_admission_for_the_next_request(self):
        state = {"replies": [plan_reply({"template": "unholy-sdxl", "seed": "42"})]}

        def factory():

            return ScriptedModel(list(state["replies"]))

        runtime = self.start_runtime(factory=factory)
        with self.assertRaises(ServiceError) as caught:
            runtime.plan("an invalid plan")
        self.assertEqual(caught.exception.code, PLAN_FAILED)
        state["replies"] = [plan_reply(VALID_PLAN)]
        result = runtime.plan("a valid plan")
        self.assertEqual(result.plan, VALID_PLAN)


class AdmissionTests(RuntimeTestCase):
    def test_invalid_plan_values_are_not_coerced(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        result = runtime.plan("plan one")
        cases = {
            "not-an-object": ["unholy-sdxl"],
            "numeric-seed": {**VALID_PLAN, "seed": "42"},
            "boolean-seed": {**VALID_PLAN, "seed": True},
            "unknown-field": {**VALID_PLAN, "command": "rm -rf /"},
            "unknown-template": {**VALID_PLAN, "template": "missing"},
            "image-field": {**VALID_PLAN, "image": "/etc/passwd"},
        }
        for label, plan in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ServiceError) as caught:
                    runtime.start_generation(plan, result.package_digest, f"req-{label}")
                self.assertEqual(caught.exception.code, INVALID_INPUT)
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(self.job_count(), 0)

    def test_request_id_and_digest_are_validated(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        result = runtime.plan("plan one")
        for request_id in ("", "has space", "slash/id", "x" * 129):
            with self.subTest(request_id=request_id):
                with self.assertRaises(ServiceError) as caught:
                    runtime.start_generation(VALID_PLAN, result.package_digest, request_id)
                self.assertEqual(caught.exception.code, INVALID_INPUT)
        with self.assertRaises(ServiceError) as caught:
            runtime.start_generation(VALID_PLAN, "", "req-1")
        self.assertEqual(caught.exception.code, INVALID_INPUT)

    def test_a_stale_package_digest_is_rejected_before_enqueue(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        runtime.plan("plan one")
        with self.assertRaises(ServiceError) as caught:
            runtime.start_generation(VALID_PLAN, "b" * 64, "req-1")
        self.assertEqual(caught.exception.code, STALE_DIGEST)
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(self.job_count(), 0)

    def test_duplicate_request_returns_the_same_job_and_conflict_rejects_change(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        result = runtime.plan("plan one")
        self.fake.block()
        first = runtime.start_generation(VALID_PLAN, result.package_digest, "req-1")
        second = runtime.start_generation(VALID_PLAN, result.package_digest, "req-1")
        self.assertEqual(first.job_id, second.job_id)
        with self.assertRaises(ServiceError) as caught:
            runtime.start_generation(
                {**VALID_PLAN, "prompt": "a different image"},
                result.package_digest,
                "req-1",
            )
        self.assertEqual(caught.exception.code, REQUEST_CONFLICT)
        self.fake.unblock()
        done = wait_for_job(runtime, first.job_id, {"succeeded"})
        self.assertEqual(done.status, "succeeded")
        again = runtime.start_generation(VALID_PLAN, result.package_digest, "req-1")
        self.assertEqual(again.job_id, first.job_id)
        self.assertEqual(len(self.fake.calls), 1)

    def test_concurrent_duplicate_admissions_create_one_job_and_one_submission(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        result = runtime.plan("plan one")
        barrier = threading.Barrier(4)
        outcomes = []

        def submit():
            barrier.wait()
            outcomes.append(
                runtime.start_generation(VALID_PLAN, result.package_digest, "req-race")
            )

        threads = [threading.Thread(target=submit) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(20)
        self.assertEqual(len({view.job_id for view in outcomes}), 1)
        wait_for_job(runtime, outcomes[0].job_id, {"succeeded"})
        self.assertEqual(len(self.fake.calls), 1)

    def test_concurrent_conflicting_admissions_leave_one_winner(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        result = runtime.plan("plan one")
        barrier = threading.Barrier(2)
        outcomes = []

        def submit(prompt):
            barrier.wait()
            try:
                outcomes.append(
                    (
                        "ok",
                        runtime.start_generation(
                            {**VALID_PLAN, "prompt": prompt},
                            result.package_digest,
                            "req-conflict",
                        ),
                    )
                )
            except Exception as exc:
                outcomes.append(("error", exc))

        threads = [
            threading.Thread(target=submit, args=("left",)),
            threading.Thread(target=submit, args=("right",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(20)
        winners = [value for status, value in outcomes if status == "ok"]
        errors = [value for status, value in outcomes if status == "error"]
        self.assertEqual(len(winners), 1, outcomes)
        self.assertEqual([error.code for error in errors], [REQUEST_CONFLICT])
        wait_for_job(runtime, winners[0].job_id, {"succeeded"})
        self.assertEqual(len(self.fake.calls), 1)

    def test_generation_is_disabled_without_the_flag(self):
        config = replace(
            self.config,
            generation_enabled=False,
            state_dir=Path(self.temporary.name) / "disabled-state",
            output_dir=Path(self.temporary.name) / "disabled-output",
        )
        runtime = self.start_runtime(config=config, factory=self.plain_factory())
        result = runtime.plan("plan one")
        with self.assertRaises(ServiceError) as caught:
            runtime.start_generation(VALID_PLAN, result.package_digest, "req-1")
        self.assertEqual(caught.exception.code, GENERATION_DISABLED)
        self.assertEqual(self.fake.calls, [])

    def test_generation_stays_disabled_without_the_runpod_key(self):
        runtime = self.start_runtime(factory=self.plain_factory(), env={})
        result = runtime.plan("plan one")
        with self.assertRaises(ServiceError) as caught:
            runtime.start_generation(VALID_PLAN, result.package_digest, "req-1")
        self.assertEqual(caught.exception.code, GENERATION_DISABLED)
        self.assertIn("RUNPOD_API_KEY", caught.exception.message)
        self.assertEqual(self.fake.calls, [])

    def test_waiting_queue_rejects_admission_above_the_limit(self):
        self.config = replace(self.config, max_waiting_jobs=2)
        self.fake.block()
        runtime = self.start_runtime(config=self.config, factory=self.plain_factory())
        result = runtime.plan("plan one")
        first = runtime.start_generation(VALID_PLAN, result.package_digest, "req-1")
        self.assertTrue(self.fake.entered.wait(10))
        runtime.start_generation(VALID_PLAN, result.package_digest, "req-2")
        runtime.start_generation(VALID_PLAN, result.package_digest, "req-3")
        with self.assertRaises(ServiceError) as caught:
            runtime.start_generation(VALID_PLAN, result.package_digest, "req-4")
        self.assertEqual(caught.exception.code, CAPACITY)
        self.fake.unblock()
        wait_for_job(runtime, first.job_id, {"succeeded"})

    def test_preflight_accepts_template_default_dimensions(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        plan = {"template": "unholy-sdxl", "prompt": "defaults", "seed": 7}
        result = runtime.plan("plan defaults")
        view = runtime.start_generation(plan, result.package_digest, "req-defaults")
        done = wait_for_job(runtime, view.job_id, {"succeeded"})
        self.assertEqual(done.plan, plan)
        self.assertEqual(self.fake.calls[0]["plan"], plan)

    def test_preflight_rejects_effective_dimensions_over_decoder_caps(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        result = runtime.plan("plan one")
        cases = {
            "side": {"width": 16376, "height": 1216},
            "pixels": {"width": 8192, "height": 4096},
        }
        for label, changes in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ServiceError) as caught:
                    runtime.start_generation(
                        {**VALID_PLAN, **changes}, result.package_digest, f"req-{label}"
                    )
                self.assertEqual(caught.exception.code, INVALID_INPUT)
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(self.job_count(), 0)

    def test_preflight_accepts_the_exact_decoder_pixel_limit(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        result = runtime.plan("plan one")
        view = runtime.start_generation(
            {**VALID_PLAN, "width": 8192, "height": 2048},
            result.package_digest,
            "req-limit",
        )
        wait_for_job(runtime, view.job_id, {"succeeded"})
        self.assertEqual(len(self.fake.calls), 1)

    def test_preflight_runner_failure_prevents_enqueue(self):
        workflow = self.config.package_root / "workflows" / "unholy-sdxl.json"
        workflow.write_text("{not valid json", encoding="utf-8")
        runtime = self.start_runtime(factory=self.plain_factory())
        result = runtime.plan("plan one")
        with self.assertRaises(ServiceError) as caught:
            runtime.start_generation(VALID_PLAN, result.package_digest, "req-1")
        self.assertEqual(caught.exception.code, PLAN_FAILED)
        self.assertIn("[path]", caught.exception.message)
        self.assertNotIn(str(self.config.package_root), caught.exception.message)
        self.assertLessEqual(len(caught.exception.message), 400)
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(self.job_count(), 0)

    def test_preflight_detail_is_credential_and_path_scrubbed(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        planned = runtime.plan("plan one")
        completed = subprocess.CompletedProcess(
            args=["runner"],
            returncode=1,
            stdout="",
            stderr=f"failed with {TEST_KEY} at /home/user/private\n",
        )
        with mock.patch(
            "distribute_agent.service.runtime.subprocess.run", return_value=completed
        ):
            with self.assertRaises(ServiceError) as caught:
                runtime.start_generation(VALID_PLAN, planned.package_digest, "req-1")
        self.assertEqual(caught.exception.code, PLAN_FAILED)
        self.assertNotIn(TEST_KEY, caught.exception.message)
        self.assertIn("[redacted]", caught.exception.message)
        self.assertIn("[path]", caught.exception.message)
        self.assertLessEqual(len(caught.exception.message), 400)

    def test_preflight_rejects_a_template_that_accepts_images(self):
        manifest_path = self.config.package_root / "manifest.json"
        manifest = read_manifest(manifest_path)
        manifest["templates"]["unholy-sdxl"]["image"] = {"node": "7"}
        write_manifest(manifest_path, manifest)
        runtime = self.start_runtime(factory=self.plain_factory())
        result = runtime.plan("plan one")
        with self.assertRaises(ServiceError) as caught:
            runtime.start_generation(VALID_PLAN, result.package_digest, "req-1")
        self.assertEqual(caught.exception.code, INVALID_INPUT)
        self.assertEqual(self.job_count(), 0)


class SnapshotTests(RuntimeTestCase):
    def test_snapshot_is_content_addressed_and_contains_only_shipped_inputs(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        result = runtime.plan("plan one")
        snapshot = self.config.snapshot_dir / result.package_digest
        self.assertTrue(snapshot.is_dir())
        self.assertTrue((snapshot / "manifest.json").is_file())
        self.assertTrue((snapshot / "bin" / "rp_oneshot.py").is_file())
        self.assertTrue((snapshot / "workflows" / "unholy-sdxl.json").is_file())
        self.assertTrue((snapshot / "skills" / "runpod-oneshot" / "SKILL.md").is_file())
        self.assertFalse((snapshot / "README.md").exists())
        self.assertTrue((self.config.package_root / "README.md").is_file())

    def test_live_package_edits_do_not_change_the_serving_snapshot(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        first = runtime.plan("plan one")
        manifest_path = self.config.package_root / "manifest.json"
        original = read_manifest(manifest_path)
        changed = dict(original)
        changed["built_at"] = "2999-01-01T00:00:00+00:00"
        write_manifest(manifest_path, changed)
        second = runtime.plan("plan two")
        self.assertEqual(first.package_digest, second.package_digest)
        pinned = read_manifest(
            self.config.snapshot_dir / first.package_digest / "manifest.json"
        )
        self.assertEqual(pinned["built_at"], original["built_at"])
        self.assertTrue(runtime.stop())
        restarted = self.start_runtime(factory=self.plain_factory())
        third = restarted.plan("plan three")
        self.assertNotEqual(first.package_digest, third.package_digest)

    def test_queued_job_uses_its_original_snapshot_after_restart_and_live_change(self):
        factory = self.plain_factory()
        first_runtime = self.start_runtime(factory=factory)
        planned = first_runtime.plan("plan one")
        original_built_at = read_manifest(
            self.config.package_root / "manifest.json"
        )["built_at"]
        self.assertTrue(first_runtime.stop())
        manifest_path = self.config.package_root / "manifest.json"
        manifest = read_manifest(manifest_path)
        manifest["built_at"] = "2999-01-01T00:00:00+00:00"
        write_manifest(manifest_path, manifest)
        store = JobStore(self.config.database_path)
        try:
            queued, created = store.enqueue(
                request_id="queued-1",
                plan=VALID_PLAN,
                package_digest=planned.package_digest,
                max_waiting_jobs=20,
            )
        finally:
            store.close()
        self.assertTrue(created)
        second_runtime = self.start_runtime(factory=factory)
        done = wait_for_job(second_runtime, queued.job_id, {"succeeded"})
        self.assertEqual(done.package_digest, planned.package_digest)
        self.assertEqual(self.fake.calls[0]["package"].root.name, planned.package_digest)
        self.assertEqual(
            self.fake.calls[0]["package"].manifest["built_at"], original_built_at
        )
        with self.assertRaises(ServiceError) as caught:
            second_runtime.start_generation(
                VALID_PLAN, planned.package_digest, "new-request"
            )
        self.assertEqual(caught.exception.code, STALE_DIGEST)

    def test_an_exact_retry_survives_a_renamed_live_template(self):
        disabled = replace(self.config, generation_enabled=False)
        first = self.start_runtime(config=disabled, factory=self.plain_factory())
        planned = first.plan("plan one")
        store = JobStore(self.config.database_path)
        try:
            queued, created = store.enqueue(
                request_id="queued-rename",
                plan=VALID_PLAN,
                package_digest=planned.package_digest,
                max_waiting_jobs=20,
            )
        finally:
            store.close()
        self.assertTrue(created)
        self.assertTrue(first.stop())
        self._rename_live_template("unholy-sdxl", "replacement")
        second = self.start_runtime(config=disabled, factory=self.plain_factory())
        retried = second.start_generation(
            VALID_PLAN, planned.package_digest, "queued-rename"
        )
        self.assertEqual(retried.job_id, queued.job_id)
        self.assertEqual(retried.status, "queued")
        for label, plan in (
            ("changed prompt", {**VALID_PLAN, "prompt": "a different image"}),
            ("malformed seed", {**VALID_PLAN, "seed": "42"}),
        ):
            with self.subTest(body=label):
                with self.assertRaises(ServiceError) as caught:
                    second.start_generation(
                        plan, planned.package_digest, "queued-rename"
                    )
                self.assertEqual(caught.exception.code, REQUEST_CONFLICT)
        with self.assertRaises(ServiceError) as caught:
            second.start_generation(
                {**VALID_PLAN, "template": "replacement"},
                planned.package_digest,
                "fresh-old-digest",
            )
        self.assertEqual(caught.exception.code, STALE_DIGEST)
        self.assertEqual(self.fake.calls, [])

    def _rename_live_template(self, old, new):
        manifest_path = self.config.package_root / "manifest.json"
        manifest = read_manifest(manifest_path)
        manifest["templates"][new] = manifest["templates"].pop(old)
        write_manifest(manifest_path, manifest)
        endpoints_path = self.config.package_root / "endpoints.json"
        endpoints = read_manifest(endpoints_path)
        endpoints[new] = endpoints.pop(old)
        write_manifest(endpoints_path, endpoints)
        workflow = self.config.package_root / "workflows" / f"{old}.json"
        workflow.rename(workflow.with_name(f"{new}.json"))

    def test_queued_jobs_wait_while_generation_is_disabled_then_resume(self):
        disabled = replace(self.config, generation_enabled=False)
        first = self.start_runtime(config=disabled, factory=self.plain_factory())
        planned = first.plan("plan one")
        self.assertTrue(first.stop())
        store = JobStore(self.config.database_path)
        try:
            queued, _ = store.enqueue(
                request_id="waiting-1",
                plan=VALID_PLAN,
                package_digest=planned.package_digest,
                max_waiting_jobs=20,
            )
        finally:
            store.close()
        waiting = self.start_runtime(config=disabled, factory=self.plain_factory())
        time.sleep(0.7)
        self.assertEqual(waiting.get_generation(queued.job_id).status, "queued")
        self.assertEqual(self.fake.calls, [])
        self.assertTrue(waiting.stop())
        resumed = self.start_runtime(factory=self.plain_factory())
        done = wait_for_job(resumed, queued.job_id, {"succeeded"})
        self.assertEqual(done.status, "succeeded")
        self.assertEqual(len(self.fake.calls), 1)

    def test_a_succeeded_job_and_its_artifacts_survive_a_restart(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        planned = runtime.plan("plan one")
        view = runtime.start_generation(VALID_PLAN, planned.package_digest, "req-1")
        done = wait_for_job(runtime, view.job_id, {"succeeded"})
        artifact_id = done.artifacts[0].artifact_id
        expected = runtime.read_artifact(done.job_id, artifact_id).data
        self.assertTrue(runtime.stop())
        restarted = self.start_runtime(factory=self.plain_factory())
        again = restarted.get_generation(done.job_id)
        self.assertEqual(again.status, "succeeded")
        self.assertEqual(again.vendor_job_id, done.vendor_job_id)
        self.assertEqual(restarted.read_artifact(done.job_id, artifact_id).data, expected)
        self.assertEqual(len(self.fake.calls), 1)

    def test_a_queued_job_with_a_missing_snapshot_needs_attention_without_submission(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        store = JobStore(self.config.database_path)
        try:
            queued, _ = store.enqueue(
                request_id="missing-snapshot",
                plan=VALID_PLAN,
                package_digest="b" * 64,
                max_waiting_jobs=20,
            )
        finally:
            store.close()
        done = wait_for_job(runtime, queued.job_id, {"needs_attention"})
        self.assertEqual(done.error["code"], SNAPSHOT_UNAVAILABLE)
        self.assertEqual(self.fake.calls, [])

    def test_running_job_from_a_previous_process_is_not_resubmitted(self):
        store = JobStore(self.config.database_path)
        try:
            queued, _ = store.enqueue(
                request_id="interrupted",
                plan=VALID_PLAN,
                package_digest="b" * 64,
                max_waiting_jobs=20,
            )
            claimed = store.claim_next()
        finally:
            store.close()
        self.assertEqual(claimed.status, "running")
        runtime = self.start_runtime(factory=self.plain_factory())
        done = wait_for_job(runtime, queued.job_id, {"needs_attention"})
        self.assertEqual(done.error["code"], INTERRUPTED)
        self.assertTrue(done.error["may_have_submitted"])
        self.assertEqual(self.fake.calls, [])


class LifecycleTests(RuntimeTestCase):
    def test_second_instance_is_locked_out_then_released_on_stop(self):
        first = self.start_runtime(factory=self.plain_factory())
        second = MediaRuntime(
            self.config, submission=self.fake, env=self.env, model_factory=None
        )
        with self.assertRaises(ServiceError) as caught:
            second.start()
        self.assertEqual(caught.exception.code, STATE_LOCKED)
        second.stop()
        self.assertTrue(first.stop())
        third = self.start_runtime(factory=self.plain_factory())
        third.stop()

    def test_graceful_stop_drains_active_work_and_rejects_new_admission(self):
        self.fake.block()
        runtime = self.start_runtime(factory=self.plain_factory())
        planned = runtime.plan("plan one")
        view = runtime.start_generation(VALID_PLAN, planned.package_digest, "req-1")
        self.assertTrue(self.fake.entered.wait(10))
        stopped = {}

        def stop():
            stopped["result"] = runtime.stop(timeout=10)

        thread = threading.Thread(target=stop)
        thread.start()
        time.sleep(0.05)
        self.assertTrue(thread.is_alive())
        self.fake.unblock()
        thread.join(15)
        self.assertTrue(stopped["result"])
        store = JobStore(self.config.database_path)
        try:
            done = store.get_job(view.job_id)
        finally:
            store.close()
        self.assertEqual(done.status, "succeeded")
        self.assertEqual(len(self.fake.calls), 1)
        with self.assertRaises(ServiceError) as caught:
            runtime.start_generation(VALID_PLAN, planned.package_digest, "req-2")
        self.assertEqual(caught.exception.code, SHUTTING_DOWN)
        with self.assertRaises(ServiceError):
            runtime.plan("another")

    def test_stop_with_an_active_planner_abandons_without_releasing_the_lock(self):
        models = []

        def factory():
            model = BlockingScriptedModel([plan_reply(VALID_PLAN)])
            models.append(model)
            return model

        runtime = MediaRuntime(
            self.config, submission=self.fake, env=self.env, model_factory=factory
        )
        runtime.start()
        outcome = {}

        def run_plan():
            try:
                outcome["result"] = runtime.plan("slow plan")
            except Exception as exc:
                outcome["error"] = exc

        planner = threading.Thread(target=run_plan, daemon=True)
        planner.start()
        self.addCleanup(planner.join, 10)
        self.addCleanup(lambda: models and models[0].release.set())
        wait_until(lambda: models and models[0].started.is_set())
        self.assertFalse(runtime.stop(timeout=0.05))
        self.assertEqual(runtime._state, "abandoned")
        replacement = MediaRuntime(self.config, submission=self.fake, env=self.env)
        self.addCleanup(replacement.stop)
        with self.assertRaises(ServiceError) as caught:
            replacement.start()
        self.assertEqual(caught.exception.code, STATE_LOCKED)
        with self.assertRaises(ServiceError) as caught:
            runtime.plan("second planner")
        self.assertEqual(caught.exception.code, SHUTTING_DOWN)
        models[0].release.set()
        planner.join(20)
        self.assertFalse(planner.is_alive())
        self.assertEqual(outcome["result"].plan, VALID_PLAN)
        self.assertFalse(runtime.stop())

    def test_stop_waits_for_a_planner_that_finishes_within_the_bound(self):
        models = []

        def factory():
            model = BlockingScriptedModel([plan_reply(VALID_PLAN)])
            models.append(model)
            return model

        runtime = MediaRuntime(
            self.config, submission=self.fake, env=self.env, model_factory=factory
        )
        runtime.start()
        outcome = {}

        def run_plan():
            try:
                outcome["result"] = runtime.plan("slow plan")
            except Exception as exc:
                outcome["error"] = exc

        planner = threading.Thread(target=run_plan, daemon=True)
        planner.start()
        self.addCleanup(planner.join, 10)
        self.addCleanup(lambda: models and models[0].release.set())
        wait_until(lambda: models and models[0].started.is_set())
        stopped = {}

        def stop():
            stopped["result"] = runtime.stop(timeout=10)

        stopper = threading.Thread(target=stop)
        stopper.start()
        self.addCleanup(stopper.join, 10)
        wait_until(lambda: runtime._state == "stopping")
        self.assertTrue(stopper.is_alive())
        models[0].release.set()
        stopper.join(15)
        self.assertTrue(stopped["result"])
        self.assertFalse(planner.is_alive())
        self.assertEqual(outcome["result"].plan, VALID_PLAN)
        replacement = MediaRuntime(self.config, submission=self.fake, env=self.env)
        replacement.start()
        self.assertTrue(replacement.stop())

    def test_stop_waits_for_a_pooled_planner_operation_not_the_idle_pool_thread(self):
        models = []

        def factory():
            model = BlockingScriptedModel([plan_reply(VALID_PLAN)])
            models.append(model)
            return model

        runtime = self.start_runtime(factory=factory)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="review-pool")
        self.addCleanup(executor.shutdown, wait=False)
        future = executor.submit(runtime.plan, "slow plan")
        self.addCleanup(lambda: models and models[0].release.set())
        wait_until(lambda: models and models[0].started.is_set())
        stopped = {}

        def stop():
            stopped["result"] = runtime.stop(timeout=10)

        stopper = threading.Thread(target=stop)
        stopper.start()
        self.addCleanup(stopper.join, 10)
        wait_until(lambda: runtime._state == "stopping")
        self.assertTrue(stopper.is_alive())
        models[0].release.set()
        self.assertEqual(future.result(20).plan, VALID_PLAN)
        stopper.join(10)
        self.assertTrue(stopped["result"])
        self.assertEqual(runtime._state, "stopped")
        # The executor-owned thread deliberately stays alive and idle; shutdown
        # must have waited for the planning operation, not that thread's exit.
        self.assertEqual(executor.submit(lambda: "alive").result(5), "alive")
        replacement = MediaRuntime(self.config, submission=self.fake, env=self.env)
        replacement.start()
        self.assertTrue(replacement.stop())

    def test_a_cancelled_planner_keeps_admission_until_it_really_finishes(self):
        models = []

        def factory():
            model = BlockingScriptedModel([plan_reply(VALID_PLAN)])
            if models:
                model.release.set()
            models.append(model)
            return model

        runtime = self.start_runtime(factory=factory)
        executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(executor.shutdown, wait=False)
        future = executor.submit(runtime.plan, "slow plan")
        self.addCleanup(lambda: models and models[0].release.set())
        wait_until(lambda: models and models[0].started.is_set())
        self.assertFalse(future.cancel())
        with self.assertRaises(ServiceError) as caught:
            runtime.plan("second planner")
        self.assertEqual(caught.exception.code, BUSY)
        models[0].release.set()
        self.assertEqual(future.result(20).plan, VALID_PLAN)
        self.assertEqual(runtime.plan("third planner").plan, VALID_PLAN)

    def test_stop_during_a_slow_preflight_rejects_the_admission(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        planned = runtime.plan("plan one")
        entered = threading.Event()
        release = threading.Event()

        def slow_preflight(snapshot, plan):
            entered.set()
            release.wait(5)

        runtime._preflight = slow_preflight
        outcome = {}

        def admit():
            try:
                outcome["view"] = runtime.start_generation(
                    VALID_PLAN, planned.package_digest, "req-race"
                )
            except Exception as exc:
                outcome["error"] = exc

        admission = threading.Thread(target=admit)
        admission.start()
        self.assertTrue(entered.wait(10))
        stopped = {}

        def stop():
            stopped["result"] = runtime.stop(timeout=5)

        stopper = threading.Thread(target=stop)
        stopper.start()
        stopper.join(10)
        self.assertTrue(stopped["result"])
        release.set()
        admission.join(10)
        self.assertIn("error", outcome)
        self.assertEqual(outcome["error"].code, SHUTTING_DOWN)
        self.assertEqual(self.job_count(), 0)

    def test_start_is_single_use(self):
        runtime = self.start_runtime(factory=self.plain_factory())
        with self.assertRaises(ServiceError) as caught:
            runtime.start()
        self.assertEqual(caught.exception.code, RUNTIME_ERROR)
        self.assertIn("cannot start", caught.exception.message)


if __name__ == "__main__":
    unittest.main()

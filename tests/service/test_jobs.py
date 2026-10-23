"""Transactional store behaviour: schema, idempotency, capacity and recovery."""

import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from distribute_agent.service.jobs import JobStore, StoredArtifact
from distribute_agent.service.types import (
    CAPACITY,
    INTERRUPTED,
    REQUEST_CONFLICT,
    STORAGE_ERROR,
    ServiceError,
)

DIGEST = "a" * 64
PLAN = {
    "template": "unholy-sdxl",
    "prompt": "a lighthouse in fog",
    "seed": 42,
    "width": 832,
    "height": 1216,
}


class SchemaTests(unittest.TestCase):
    def test_new_database_is_version_one_with_jobs_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.sqlite3"
            store = JobStore(path)
            try:
                version = store._connection.execute("PRAGMA user_version").fetchone()[0]
                tables = {
                    row[0]
                    for row in store._connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            finally:
                store.close()
            self.assertEqual(version, 1)
            self.assertLessEqual({"jobs", "artifacts"}, tables)

    def test_incompatible_schema_version_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute("PRAGMA user_version=2")
            connection.commit()
            connection.close()
            with self.assertRaises(ServiceError) as caught:
                JobStore(path)
            self.assertEqual(caught.exception.code, STORAGE_ERROR)
            self.assertIn("incompatible", caught.exception.message)

    def test_version_one_without_expected_tables_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE unrelated (id INTEGER)")
            connection.execute("PRAGMA user_version=1")
            connection.commit()
            connection.close()
            with self.assertRaises(ServiceError) as caught:
                JobStore(path)
            self.assertEqual(caught.exception.code, STORAGE_ERROR)
            self.assertIn("missing table", caught.exception.message)


class EnqueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = JobStore(Path(self.temporary.name) / "jobs.sqlite3")
        self.addCleanup(self.store.close)

    def test_same_request_and_plan_returns_the_existing_job(self):
        first, created = self.store.enqueue(
            request_id="req-1",
            plan=PLAN,
            package_digest=DIGEST,
            max_waiting_jobs=20,
        )
        reordered = dict(reversed(list(PLAN.items())))
        second, created_again = self.store.enqueue(
            request_id="req-1",
            plan=reordered,
            package_digest=DIGEST,
            max_waiting_jobs=20,
        )
        self.assertTrue(created)
        self.assertFalse(created_again)
        self.assertEqual(first.job_id, second.job_id)
        self.assertEqual(self.store.count_queued(), 1)

    def test_changed_plan_or_digest_conflicts(self):
        self.store.enqueue(
            request_id="req-1",
            plan=PLAN,
            package_digest=DIGEST,
            max_waiting_jobs=20,
        )
        for label, plan, digest in (
            ("prompt", {**PLAN, "prompt": "different"}, DIGEST),
            ("digest", PLAN, "b" * 64),
        ):
            with self.subTest(case=label):
                with self.assertRaises(ServiceError) as caught:
                    self.store.enqueue(
                        request_id="req-1",
                        plan=plan,
                        package_digest=digest,
                        max_waiting_jobs=20,
                    )
                self.assertEqual(caught.exception.code, REQUEST_CONFLICT)

    def test_concurrent_duplicates_create_exactly_one_job(self):
        barrier = threading.Barrier(8)

        def submit(_):
            barrier.wait()
            return self.store.enqueue(
                request_id="req-race",
                plan=PLAN,
                package_digest=DIGEST,
                max_waiting_jobs=20,
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(submit, range(8)))
        self.assertEqual(len({view.job_id for view, _ in results}), 1)
        self.assertEqual(sum(1 for _, created in results if created), 1)
        self.assertEqual(self.store.count_queued(), 1)

    def test_concurrent_conflicting_bodies_leave_one_winner(self):
        barrier = threading.Barrier(2)
        outcomes = []

        def submit(prompt):
            barrier.wait()
            try:
                view, created = self.store.enqueue(
                    request_id="req-conflict",
                    plan={**PLAN, "prompt": prompt},
                    package_digest=DIGEST,
                    max_waiting_jobs=20,
                )
                return ("ok", view.job_id, created)
            except ServiceError as exc:
                return ("error", exc.code, None)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(submit, "left"), pool.submit(submit, "right")]
            outcomes = [future.result() for future in futures]
        ok = [outcome for outcome in outcomes if outcome[0] == "ok"]
        errors = [outcome for outcome in outcomes if outcome[0] == "error"]
        self.assertEqual(len(ok), 1, outcomes)
        self.assertEqual([error[1] for error in errors], [REQUEST_CONFLICT])
        self.assertEqual(self.store.count_queued(), 1)

    def test_capacity_is_counted_inside_the_enqueue_transaction(self):
        self.store.enqueue(
            request_id="req-1",
            plan=PLAN,
            package_digest=DIGEST,
            max_waiting_jobs=1,
        )
        with self.assertRaises(ServiceError) as caught:
            self.store.enqueue(
                request_id="req-2",
                plan=PLAN,
                package_digest=DIGEST,
                max_waiting_jobs=1,
            )
        self.assertEqual(caught.exception.code, CAPACITY)
        self.assertEqual(self.store.count_queued(), 1)


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = JobStore(Path(self.temporary.name) / "jobs.sqlite3")
        self.addCleanup(self.store.close)

    def enqueue(self, request_id="req-1"):
        view, _ = self.store.enqueue(
            request_id=request_id,
            plan=PLAN,
            package_digest=DIGEST,
            max_waiting_jobs=20,
        )
        return view

    def test_claim_orders_jobs_and_finish_success_stores_artifacts(self):
        first = self.enqueue("req-1")
        second = self.enqueue("req-2")
        claimed = self.store.claim_next()
        self.assertEqual(claimed.job_id, first.job_id)
        self.assertEqual(claimed.status, "running")
        self.assertIsNotNone(claimed.started_at)
        artifact = StoredArtifact(
            artifact_id="artifact-1",
            relative_path="vendor/image_000.png",
            mime_type="image/png",
            width=4,
            height=6,
            size_bytes=123,
            sha256="f" * 64,
        )
        self.store.finish_success(claimed.job_id, "vendor-9", (artifact,))
        done = self.store.get_job(claimed.job_id)
        self.assertEqual(done.status, "succeeded")
        self.assertEqual(done.vendor_job_id, "vendor-9")
        self.assertIsNotNone(done.finished_at)
        self.assertEqual(len(done.artifacts), 1)
        self.assertEqual(done.artifacts[0].artifact_id, "artifact-1")
        self.assertEqual(self.store.get_artifact(first.job_id, "artifact-1").sha256, "f" * 64)
        self.assertEqual(self.store.claim_next().job_id, second.job_id)

    def test_finishing_a_job_twice_is_refused(self):
        view = self.enqueue()
        self.store.claim_next()
        artifact = StoredArtifact(
            artifact_id="artifact-1",
            relative_path="image.png",
            mime_type="image/png",
            width=4,
            height=6,
            size_bytes=1,
            sha256="f" * 64,
        )
        self.store.finish_success(view.job_id, "vendor-1", (artifact,))
        with self.assertRaises(ServiceError) as caught:
            self.store.finish_success(view.job_id, "vendor-1", (artifact,))
        self.assertEqual(caught.exception.code, STORAGE_ERROR)

    def test_interrupted_running_job_keeps_its_vendor_id_and_is_not_requeued(self):
        view = self.enqueue()
        claimed = self.store.claim_next()
        self.assertEqual(claimed.status, "running")
        connection = sqlite3.connect(self.store.path)
        try:
            connection.execute(
                "UPDATE jobs SET vendor_job_id = 'vendor-9' WHERE job_id = ?",
                (view.job_id,),
            )
            connection.commit()
        finally:
            connection.close()
        changed = self.store.interrupt_running()
        self.assertEqual(changed, 1)
        recovered = self.store.get_job(view.job_id)
        self.assertEqual(recovered.status, "needs_attention")
        self.assertEqual(recovered.error["code"], INTERRUPTED)
        self.assertTrue(recovered.error["may_have_submitted"])
        self.assertEqual(recovered.vendor_job_id, "vendor-9")
        self.assertEqual(self.store.count_queued(), 0)
        self.assertIsNone(self.store.claim_next())

    def test_failure_can_preserve_a_vendor_id_and_needs_attention(self):
        view = self.enqueue()
        self.store.claim_next()
        self.store.finish_failure(
            view.job_id,
            vendor_job_id="vendor-42",
            code="submission_uncertain",
            message="the outcome is unknown",
            may_have_submitted=True,
            needs_attention=True,
        )
        failed = self.store.get_job(view.job_id)
        self.assertEqual(failed.status, "needs_attention")
        self.assertEqual(failed.vendor_job_id, "vendor-42")
        self.assertTrue(failed.error["may_have_submitted"])

    def test_unknown_ids_are_not_found(self):
        self.assertIsNone(self.store.get_job("missing"))
        self.assertIsNone(self.store.get_by_request_id("missing"))
        self.assertIsNone(self.store.get_artifact("missing", "missing"))
        self.assertIsNone(self.store.claim_next())


if __name__ == "__main__":
    unittest.main()

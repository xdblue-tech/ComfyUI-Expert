"""MCP tools, resources, custom routes, auth and lifecycle over the ASGI stack."""

import asyncio
import base64
import hashlib
import sqlite3
import tempfile
import time
import unittest

from fastmcp.exceptions import ToolError
from fastmcp.utilities.tests import asgi_server
from mcp.shared.exceptions import MCPError

from distribute_agent.service.auth import issue_token
from distribute_agent.service.app import MCP_PATH, build_http_app

from service_test_support import (
    PNG_BYTES,
    TEST_KEY,
    VALID_PLAN,
    BlockingScriptedModel,
    FakeSubmission,
    plan_reply,
    scripted_factory,
    specs,
)
from transport_support import (
    TEST_SECRET,
    TEST_TOKEN,
    auth_headers,
    make_runtime,
    make_server,
)

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "mediagen-tests", "version": "1.0"},
    },
}
TIMEOUT = 15.0


class TransportCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    async def wait_for_status(self, runtime, job_id, statuses=("succeeded",)):
        statuses = set(statuses)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + TIMEOUT
        while loop.time() < deadline:
            view = await asyncio.to_thread(runtime.get_generation, job_id)
            if view.status in statuses:
                return view
            await asyncio.sleep(0.02)
        raise AssertionError(f"job {job_id} never reached {sorted(statuses)}")

    async def post_mcp(self, http, token):
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        return await http.post(MCP_PATH, json=INITIALIZE, headers=headers)

    async def start(self, client, request_id="req-1", prompt="a brass clock on a cliff", **overrides):
        return await client.call_tool(
            "start_generation",
            {"prompt": prompt, "specs": specs(request_id, **overrides)},
        )


class DiscoveryTests(TransportCase):
    async def test_tools_resources_and_schemas_are_discoverable(self):
        factory, _ = scripted_factory([plan_reply(VALID_PLAN)])
        runtime, _ = make_runtime(self._tmp.name, model_factory=factory, submission=FakeSubmission())
        async with asgi_server(make_server(runtime)) as running:
            async with running.client(headers=auth_headers()) as client:
                tools = {tool.name: tool for tool in await client.list_tools()}
                self.assertEqual(
                    set(tools), {"start_generation", "get_generation"}
                )
                start_schema = tools["start_generation"].input_schema
                self.assertEqual(set(start_schema["required"]), {"prompt", "specs"})
                self.assertEqual(
                    start_schema["properties"]["prompt"]["type"], "string"
                )
                raw_specs = start_schema["properties"]["specs"]
                self.assertIs(raw_specs["additionalProperties"], True)
                self.assertNotIn("seed", raw_specs.get("properties", {}))
                for name in ("start_generation", "get_generation"):
                    generation_schema = tools[name].output_schema
                    self.assertIn("artifacts", generation_schema["properties"])
                    self.assertIn("vendor_job_id", generation_schema["properties"])
                    self.assertIn("diagnostics", generation_schema["properties"])

                templates = await client.list_resource_templates()
                self.assertEqual(
                    [str(template.uri_template) for template in templates],
                    ["media://jobs/{job_id}/artifacts/{artifact_id}"],
                )

    async def test_the_served_app_registers_all_three_route_groups(self):
        runtime, _ = make_runtime(self._tmp.name, generation_enabled=False)
        app = build_http_app(make_server(runtime))
        paths = {getattr(route, "path", None) for route in app.routes}
        self.assertLessEqual(
            {
                MCP_PATH,
                "/health",
                "/artifacts/{job_id}/{artifact_id}",
            },
            paths,
        )


class AuthenticationTests(TransportCase):
    async def test_missing_and_invalid_tokens_are_rejected_before_mcp(self):
        runtime, _ = make_runtime(self._tmp.name, generation_enabled=False)
        expired = issue_token(TEST_SECRET, ttl_seconds=60, now=time.time() - 120)
        variants = {
            "missing": None,
            "garbage": "not-a-jwt",
            "expired": expired,
            "wrong issuer": issue_token(TEST_SECRET, issuer="other"),
            "wrong audience": issue_token(TEST_SECRET, audience="other"),
            "missing scope": issue_token(TEST_SECRET, scope="media:read"),
            "wrong secret": issue_token("different-secret-" + "z" * 32),
        }
        async with asgi_server(make_server(runtime)) as running:
            async with running.http_client() as http:
                for label, token in variants.items():
                    with self.subTest(token=label):
                        response = await self.post_mcp(http, token)
                        self.assertEqual(response.status_code, 401)
                        self.assertIn("Bearer", response.headers.get("www-authenticate", ""))

    async def test_a_valid_token_reaches_the_mcp_endpoint(self):
        runtime, _ = make_runtime(self._tmp.name, generation_enabled=False)
        async with asgi_server(make_server(runtime)) as running:
            async with running.http_client() as http:
                response = await self.post_mcp(http, TEST_TOKEN)
                self.assertNotEqual(response.status_code, 401)
                self.assertEqual(response.status_code, 200)


class PlanningToolTests(TransportCase):
    async def test_start_generation_returns_the_planned_job_and_diagnostics(self):
        factory, _ = scripted_factory([plan_reply(VALID_PLAN)])
        runtime, _ = make_runtime(self._tmp.name, model_factory=factory, submission=FakeSubmission())
        async with asgi_server(make_server(runtime)) as running:
            async with running.client(headers=auth_headers()) as client:
                result = await self.start(client)
        data = result.structured_content
        self.assertEqual(data["status"], "queued")
        self.assertEqual(data["request_id"], "req-1")
        self.assertEqual(data["plan"]["seed"], 42)
        self.assertEqual(data["plan"]["width"], 832)
        self.assertEqual(data["plan"]["height"], 1216)
        self.assertEqual(len(data["package_digest"]), 64)
        self.assertLessEqual(
            set(data["diagnostics"]),
            {"duration_seconds", "model_requests", "agent_steps", "validated_previews", "state"},
        )
        self.assertEqual(data["diagnostics"]["model_requests"], 1)
        self.assertEqual(data["artifacts"], [])
        self.assertIsNone(data["error"])

    async def test_specs_overrides_replace_the_planned_values(self):
        factory, _ = scripted_factory([plan_reply(VALID_PLAN)])
        runtime, _ = make_runtime(self._tmp.name, model_factory=factory, submission=FakeSubmission())
        async with asgi_server(make_server(runtime)) as running:
            async with running.client(headers=auth_headers()) as client:
                result = await self.start(client, width=1024, height=1024, seed=7)
        plan = result.structured_content["plan"]
        self.assertEqual((plan["width"], plan["height"], plan["seed"]), (1024, 1024, 7))

    async def test_empty_request_is_code_prefixed(self):
        runtime, _ = make_runtime(self._tmp.name, generation_enabled=False)
        async with asgi_server(make_server(runtime)) as running:
            async with running.client(headers=auth_headers()) as client:
                with self.assertRaises(ToolError) as caught:
                    await self.start(client, prompt="   ")
        self.assertTrue(str(caught.exception).startswith("invalid_input:"))

    async def test_unknown_category_is_code_prefixed_and_lists_the_valid_ones(self):
        runtime, _ = make_runtime(self._tmp.name, generation_enabled=False)
        async with asgi_server(make_server(runtime)) as running:
            async with running.client(headers=auth_headers()) as client:
                with self.assertRaises(ToolError) as caught:
                    await self.start(client, category="sd-photoreal")
        message = str(caught.exception)
        self.assertTrue(message.startswith("invalid_input:"))
        self.assertIn("sd-semireal", message)

    async def test_unknown_specs_field_is_refused(self):
        runtime, _ = make_runtime(self._tmp.name, generation_enabled=False)
        async with asgi_server(make_server(runtime)) as running:
            async with running.client(headers=auth_headers()) as client:
                with self.assertRaises(ToolError) as caught:
                    await client.call_tool(
                        "start_generation",
                        {
                            "prompt": "a brass clock",
                            "specs": {"id": "req-1", "package_digest": "0" * 64},
                        },
                    )
        self.assertTrue(str(caught.exception).startswith("invalid_input:"))

    async def test_planning_failure_is_redacted_and_bounded(self):
        def broken_model():
            raise RuntimeError(
                f"provider body DO_NOT_PUBLISH_123 for {TEST_KEY} at /home/user/private/x"
            )

        runtime, _ = make_runtime(self._tmp.name, model_factory=broken_model)
        async with asgi_server(make_server(runtime)) as running:
            async with running.client(headers=auth_headers()) as client:
                with self.assertRaises(ToolError) as caught:
                    await self.start(client)
        message = str(caught.exception)
        self.assertTrue(message.startswith("plan_failed:"))
        self.assertNotIn(TEST_KEY, message)
        self.assertNotIn("/home/user", message)
        self.assertNotIn("DO_NOT_PUBLISH_123", message)
        self.assertNotIn("provider body", message)
        self.assertLessEqual(len(message), 400 + len("plan_failed: "))

    async def test_a_planning_failure_leaves_no_job_row(self):
        def broken_model():
            raise RuntimeError("provider body DO_NOT_PUBLISH_123")

        runtime, config = make_runtime(self._tmp.name, model_factory=broken_model)
        async with asgi_server(make_server(runtime)) as running:
            async with running.client(headers=auth_headers()) as client:
                with self.assertRaises(ToolError) as caught:
                    await self.start(client)
        self.assertTrue(str(caught.exception).startswith("plan_failed:"))
        connection = sqlite3.connect(config.database_path)
        try:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0
            )
        finally:
            connection.close()

    async def test_a_second_planner_is_rejected_as_busy(self):
        models = []

        def factory():
            model = BlockingScriptedModel([plan_reply(VALID_PLAN)])
            models.append(model)
            return model

        runtime, _ = make_runtime(self._tmp.name, model_factory=factory)
        try:
            async with asgi_server(make_server(runtime)) as running:
                async with running.client(headers=auth_headers()) as client:
                    first = asyncio.create_task(self.start(client, "req-1", "first brief"))
                    while not models or not models[0].started.is_set():
                        await asyncio.sleep(0.01)
                    with self.assertRaises(ToolError) as caught:
                        await self.start(client, "req-2", "second brief")
                    self.assertTrue(str(caught.exception).startswith("busy:"))
                    models[0].release.set()
                    self.assertEqual(
                        (await first).structured_content["plan"]["seed"], 42
                    )
        finally:
            for model in models:
                model.release.set()

    async def test_request_longer_than_the_limit_is_refused(self):
        runtime, _ = make_runtime(self._tmp.name, generation_enabled=False)
        async with asgi_server(make_server(runtime)) as running:
            async with running.client(headers=auth_headers()) as client:
                with self.assertRaises(ToolError) as caught:
                    await self.start(client, prompt="x" * 8193)
        self.assertTrue(str(caught.exception).startswith("invalid_input:"))
        self.assertIn("8192", str(caught.exception))

    async def test_specs_cannot_be_smuggled_as_untyped_json(self):
        runtime, _ = make_runtime(self._tmp.name, generation_enabled=False)
        async with asgi_server(make_server(runtime)) as running:
            async with running.client(headers=auth_headers()) as client:
                with self.assertRaises(ToolError):
                    await client.call_tool(
                        "start_generation",
                        {"prompt": "a brass clock", "specs": "not-an-object"},
                    )


class GenerationToolTests(TransportCase):
    async def test_generation_is_disabled_without_the_operator_flag(self):
        factory, models = scripted_factory([plan_reply(VALID_PLAN)])
        submission = FakeSubmission()
        runtime, _ = make_runtime(
            self._tmp.name,
            generation_enabled=False,
            model_factory=factory,
            submission=submission,
        )
        async with asgi_server(make_server(runtime)) as running:
            async with running.client(headers=auth_headers()) as client:
                with self.assertRaises(ToolError) as caught:
                    await self.start(client, "req-disabled")
        self.assertTrue(str(caught.exception).startswith("generation_disabled:"))
        self.assertEqual(submission.calls, [])
        self.assertEqual(models, [])

    async def test_generation_is_disabled_without_the_vendor_key(self):
        factory, _ = scripted_factory([plan_reply(VALID_PLAN)])
        runtime, _ = make_runtime(
            self._tmp.name,
            model_factory=factory,
            submission=FakeSubmission(),
            env={"MEDIAGEN_PROXY_API_KEY": "proxy"},
        )
        async with asgi_server(make_server(runtime)) as running:
            async with running.client(headers=auth_headers()) as client:
                with self.assertRaises(ToolError) as caught:
                    await self.start(client, "req-no-key")
        self.assertTrue(str(caught.exception).startswith("generation_disabled:"))

    async def test_numeric_strings_are_not_coerced_in_the_specs(self):
        factory, _ = scripted_factory([plan_reply(VALID_PLAN)])
        submission = FakeSubmission()
        runtime, _ = make_runtime(
            self._tmp.name, model_factory=factory, submission=submission
        )
        async with asgi_server(make_server(runtime)) as running:
            async with running.client(headers=auth_headers()) as client:
                with self.assertRaises(ToolError) as caught:
                    await client.call_tool(
                        "start_generation",
                        {
                            "prompt": "a brass clock",
                            "specs": {"id": "req-raw", "seed": "42"},
                        },
                    )
        self.assertTrue(str(caught.exception).startswith("invalid_input:"))
        self.assertIn("seed", str(caught.exception))
        self.assertEqual(submission.calls, [])

    async def test_duplicate_request_id_submits_once_and_changed_body_conflicts(self):
        factory, models = scripted_factory([plan_reply(VALID_PLAN)])
        submission = FakeSubmission()
        submission.block()
        runtime, _ = make_runtime(
            self._tmp.name, model_factory=factory, submission=submission
        )
        try:
            async with asgi_server(make_server(runtime)) as running:
                async with running.client(headers=auth_headers()) as client:
                    first = await self.start(client, "req-duplicate")
                    second = await self.start(client, "req-duplicate")
                    self.assertEqual(
                        first.structured_content["job_id"],
                        second.structured_content["job_id"],
                    )
                    with self.assertRaises(ToolError) as caught:
                        await self.start(client, "req-duplicate", "a different brief")
                    self.assertTrue(str(caught.exception).startswith("request_conflict:"))
                    submission.unblock()
                    job = await self.wait_for_status(
                        runtime, first.structured_content["job_id"]
                    )
            self.assertEqual(job.status, "succeeded")
            self.assertEqual(len(submission.calls), 1)
            # The replay was answered from the store, not by planning again.
            self.assertEqual(len(models), 1)
        finally:
            submission.unblock()

    async def test_a_vendor_body_is_not_exposed_through_get_generation(self):
        factory, _ = scripted_factory([plan_reply(VALID_PLAN)])
        runtime, _ = make_runtime(
            self._tmp.name,
            model_factory=factory,
            submission=FakeSubmission(behavior="raise_uncertain"),
        )
        async with asgi_server(make_server(runtime)) as running:
            async with running.client(headers=auth_headers()) as client:
                job = await self.start(client, "req-vendor-body")
                await self.wait_for_status(
                    runtime,
                    job.structured_content["job_id"],
                    statuses=("needs_attention",),
                )
                reread = await client.call_tool(
                    "get_generation", {"job_id": job.structured_content["job_id"]}
                )
        data = reread.structured_content
        self.assertEqual(data["status"], "needs_attention")
        self.assertEqual(data["vendor_job_id"], "vendor-77")
        self.assertEqual(data["error"]["code"], "submission_uncertain")
        self.assertTrue(data["error"]["may_have_submitted"])
        self.assertEqual(set(data["diagnostics"]),
                         {"duration_seconds", "model_requests", "agent_steps",
                          "validated_previews", "state"})
        for marker in (
            "DO_NOT_PUBLISH_123",
            "vendor_private_body",
            TEST_KEY,
            "/home/user",
        ):
            self.assertNotIn(marker, data["error"]["message"])

    async def test_unknown_job_is_code_prefixed(self):
        runtime, _ = make_runtime(self._tmp.name, generation_enabled=False)
        async with asgi_server(make_server(runtime)) as running:
            async with running.client(headers=auth_headers()) as client:
                with self.assertRaises(ToolError) as caught:
                    await client.call_tool("get_generation", {"job_id": "missing"})
        self.assertTrue(str(caught.exception).startswith("unknown_job:"))

    async def test_client_disconnect_does_not_cancel_a_paid_generation(self):
        factory, _ = scripted_factory([plan_reply(VALID_PLAN)])
        submission = FakeSubmission()
        submission.block()
        runtime, _ = make_runtime(
            self._tmp.name, model_factory=factory, submission=submission
        )
        try:
            async with asgi_server(make_server(runtime)) as running:
                async with running.client(headers=auth_headers()) as client:
                    job = await self.start(client, "req-disconnect")
                # The client and its HTTP session are gone; the worker is independent.
                submission.unblock()
                view = await self.wait_for_status(
                    runtime, job.structured_content["job_id"]
                )
                self.assertEqual(view.status, "succeeded")
                async with running.client(headers=auth_headers()) as client:
                    reread = await client.call_tool(
                        "get_generation", {"job_id": job.structured_content["job_id"]}
                    )
                    self.assertEqual(reread.structured_content["status"], "succeeded")
            self.assertEqual(len(submission.calls), 1)
        finally:
            submission.unblock()


class ArtifactTests(TransportCase):
    def make_completed_job(self, **config_overrides):
        factory, _ = scripted_factory([plan_reply(VALID_PLAN)])
        submission = FakeSubmission()
        runtime, config = make_runtime(
            self._tmp.name,
            model_factory=factory,
            submission=submission,
            **config_overrides,
        )
        server = make_server(runtime)
        return runtime, config, server

    async def start_and_wait(self, runtime, client):
        job = await self.start(client, "req-artifact")
        view = await self.wait_for_status(runtime, job.structured_content["job_id"])
        return job, job.structured_content["job_id"], view

    async def test_resource_returns_exact_validated_bytes_and_metadata(self):
        runtime, _, server = self.make_completed_job()
        async with asgi_server(server) as running:
            async with running.client(headers=auth_headers()) as client:
                _, job_id, view = await self.start_and_wait(runtime, client)
                metadata = view.artifacts[0]
                self.assertEqual(metadata.mime_type, "image/png")
                self.assertEqual(metadata.size_bytes, len(PNG_BYTES))
                self.assertEqual(metadata.sha256, hashlib.sha256(PNG_BYTES).hexdigest())
                contents = await client.read_resource(
                    f"media://jobs/{job_id}/artifacts/{metadata.artifact_id}"
                )
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0].mime_type, "image/png")
        self.assertEqual(base64.b64decode(contents[0].blob), PNG_BYTES)

    async def test_resource_rejects_unknown_ids_and_traversal(self):
        runtime, _, server = self.make_completed_job()
        async with asgi_server(server) as running:
            async with running.client(headers=auth_headers()) as client:
                _, job_id, view = await self.start_and_wait(runtime, client)
                for uri in (
                    "media://jobs/missing/artifacts/missing",
                    f"media://jobs/{job_id}/artifacts/missing",
                    "media://jobs/../artifacts/x",
                    f"media://jobs/{job_id}/artifacts/../../etc/passwd",
                ):
                    with self.subTest(uri=uri):
                        with self.assertRaises(Exception) as caught:
                            await client.read_resource(uri)
                        message = str(caught.exception)
                        self.assertNotIn("Traceback", message)
                        self.assertNotIn("root:", message)

    async def test_resource_inline_cap_sends_large_files_to_http(self):
        runtime, config, server = self.make_completed_job(max_inline_artifact_bytes=32)
        async with asgi_server(server) as running:
            async with running.client(headers=auth_headers()) as client:
                _, job_id, view = await self.start_and_wait(runtime, client)
                artifact_id = view.artifacts[0].artifact_id
                with self.assertRaises(MCPError) as caught:
                    await client.read_resource(
                        f"media://jobs/{job_id}/artifacts/{artifact_id}"
                    )
                self.assertIn("artifact_too_large", str(caught.exception))
            async with running.http_client(headers=auth_headers()) as http:
                response = await http.get(f"/artifacts/{job_id}/{artifact_id}")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, PNG_BYTES)

    async def test_http_artifact_route_verifies_its_own_token(self):
        runtime, _, server = self.make_completed_job()
        async with asgi_server(server) as running:
            async with running.client(headers=auth_headers()) as client:
                _, job_id, view = await self.start_and_wait(runtime, client)
            artifact_id = view.artifacts[0].artifact_id
            path = f"/artifacts/{job_id}/{artifact_id}"
            async with running.http_client() as http:
                for token in (
                    None,
                    "not-a-jwt",
                    issue_token(TEST_SECRET, issuer="other"),
                    issue_token(TEST_SECRET, scope="media:read"),
                    issue_token(TEST_SECRET, ttl_seconds=60, now=time.time() - 120),
                ):
                    with self.subTest(token=token):
                        headers = {} if token is None else {"Authorization": f"Bearer {token}"}
                        response = await http.get(path, headers=headers)
                        self.assertEqual(response.status_code, 401)
                        self.assertIn("Bearer", response.headers.get("www-authenticate", ""))
            async with running.http_client(headers=auth_headers()) as http:
                response = await http.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "image/png")
                self.assertEqual(response.headers["x-content-type-options"], "nosniff")
                self.assertEqual(response.content, PNG_BYTES)
                self.assertEqual(
                    hashlib.sha256(response.content).hexdigest(),
                    view.artifacts[0].sha256,
                )

    async def test_http_artifact_route_maps_unknown_and_corrupt(self):
        runtime, config, server = self.make_completed_job()
        async with asgi_server(server) as running:
            async with running.client(headers=auth_headers()) as client:
                _, job_id, view = await self.start_and_wait(runtime, client)
            artifact_id = view.artifacts[0].artifact_id
            async with running.http_client(headers=auth_headers()) as http:
                unknown = await http.get(f"/artifacts/{job_id}/missing")
                self.assertEqual(unknown.status_code, 404)
                self.assertTrue(unknown.text.startswith("unknown_artifact:"))
                # Remove the published file: the metadata now points at nothing.
                published = config.output_dir / job_id / "vendor-1" / "image_000.png"
                published.unlink()
                corrupt = await http.get(f"/artifacts/{job_id}/{artifact_id}")
                self.assertEqual(corrupt.status_code, 410)
                self.assertTrue(corrupt.text.startswith("artifact_corrupt:"))

                outside = config.output_dir.parent / "outside.png"
                outside.write_bytes(PNG_BYTES)
                published.symlink_to(outside)
                linked = await http.get(f"/artifacts/{job_id}/{artifact_id}")
                self.assertEqual(linked.status_code, 410)
                self.assertTrue(linked.text.startswith("artifact_corrupt:"))

    async def test_health_is_public_and_get_only(self):
        runtime, _ = make_runtime(self._tmp.name, generation_enabled=False)
        async with asgi_server(make_server(runtime)) as running:
            async with running.http_client() as http:
                response = await http.get("/health")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"status": "ok"})
                self.assertEqual(response.headers["cache-control"], "no-store")
                self.assertEqual((await http.post("/health")).status_code, 405)


class HostOriginTests(TransportCase):
    async def test_host_and_origin_guards_reject_foreign_values(self):
        runtime, _ = make_runtime(self._tmp.name, generation_enabled=False)
        async with asgi_server(
            make_server(runtime), host_origin_protection="auto"
        ) as running:
            async with running.http_client() as http:
                self.assertEqual((await http.get("/health")).status_code, 200)
                foreign_host = await http.get("/health", headers={"Host": "evil.example"})
                self.assertEqual(foreign_host.status_code, 421)
                foreign_origin = await http.get(
                    "/health", headers={"Origin": "http://evil.example"}
                )
                self.assertEqual(foreign_origin.status_code, 403)

    async def test_configured_host_and_origin_are_accepted(self):
        runtime, _ = make_runtime(self._tmp.name, generation_enabled=False)
        async with asgi_server(
            make_server(runtime),
            host_origin_protection="auto",
            allowed_hosts=["media.internal"],
            allowed_origins=["http://media.internal:8000"],
        ) as running:
            async with running.http_client() as raw:
                response = await raw.get(
                    "/health",
                    headers={
                        "Host": "media.internal",
                        "Origin": "http://media.internal:8000",
                    },
                )
                self.assertEqual(response.status_code, 200)


class LifespanTests(TransportCase):
    async def test_shutdown_releases_the_state_lock_for_a_new_instance(self):
        runtime, config = make_runtime(self._tmp.name, generation_enabled=False)
        async with asgi_server(make_server(runtime)):
            pass
        from distribute_agent.service.runtime import MediaRuntime

        replacement = MediaRuntime(config, env={"RUNPOD_API_KEY": TEST_KEY})
        replacement.start()
        try:
            self.assertIsNotNone(replacement.config.database_path)
        finally:
            self.assertTrue(replacement.stop())

    async def test_a_runtime_that_cannot_drain_makes_the_transport_exit_nonzero(self):
        class StuckRuntime:
            started = False
            stopped = False

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True
                return False

        stuck = StuckRuntime()
        with self.assertRaises(RuntimeError) as caught:
            async with asgi_server(make_server(stuck)):
                pass
        self.assertTrue(stuck.started)
        self.assertTrue(stuck.stopped)
        self.assertIn("did not release its state lock", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

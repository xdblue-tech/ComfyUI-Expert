"""Complete offline tracer: MCP -> CodeAgent -> sandbox -> bridge -> fake vendor.

The only substitutions are off the production path: a scripted model replaces
the hosted one, and the existing ``runner_prefix`` seam runs a test wrapper that
imports the pinned runner and points its vendor base at a localhost fake. No
production URL or credential changes; the wrapper and fake vendor are test-only.
"""

import asyncio
import base64
import hashlib
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fastmcp.utilities.tests import asgi_server

from distribute_agent.scripted_model import ScriptedModel, code_reply
from distribute_agent.service.config import ServiceConfig
from distribute_agent.service.runtime import MediaRuntime

from service_test_support import TEST_KEY, VALID_PLAN, build_png, copy_package
from transport_support import TEST_ENV, auth_headers, make_server

PROMPT = (
    "masterpiece, best quality, a lighthouse keeper winding a brass clock on a "
    "stormy cliff, oilskin coat, cold blue light, no text"
)
VENDOR_ID = "fake-job-42"
ARTIFACT_WIDTH = 832
ARTIFACT_HEIGHT = 1216


class FakeVendor:
    """A localhost stand-in for the RunPod jobs API."""

    def __init__(self, image_bytes):
        self.image_bytes = image_bytes
        self.posts = []
        self.gets = []
        self.unexpected = []
        vendor = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("content-length") or 0))
                vendor.posts.append(
                    {
                        "path": self.path,
                        "body": body,
                        "authorization": self.headers.get("authorization"),
                    }
                )
                vendor.reply(
                    self,
                    {"id": VENDOR_ID, "status": "IN_QUEUE"},
                )

            def do_GET(self):
                vendor.gets.append(self.path)
                if f"/status/{VENDOR_ID}" not in self.path:
                    vendor.unexpected.append(self.path)
                    vendor.reply(self, {"error": "unexpected path"}, status=404)
                    return
                vendor.reply(
                    self,
                    {
                        "id": VENDOR_ID,
                        "status": "COMPLETED",
                        "output": {
                            "images": [
                                {
                                    "filename": "image_000.png",
                                    "data": base64.b64encode(vendor.image_bytes).decode("ascii"),
                                }
                            ]
                        },
                    },
                )

            def log_message(self, *_args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, name="fake-vendor", daemon=True
        )

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v2"

    def start(self):
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    @staticmethod
    def reply(handler, payload, status=200):
        encoded = json.dumps(payload).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(encoded)))
        handler.end_headers()
        handler.wfile.write(encoded)


def write_runner_wrapper(path, state_dir, vendor_base):
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import importlib.util\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        f"STATE = Path({str(state_dir)!r})\n"
        "runners = sorted(STATE.glob('packages/*/bin/rp_oneshot.py'))\n"
        "if len(runners) != 1:\n"
        "    raise SystemExit(f'expected one pinned runner, found {len(runners)}')\n"
        "spec = importlib.util.spec_from_file_location('rp_oneshot_pinned', runners[0])\n"
        "runner = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(runner)\n"
        f"runner.RUNPOD_JOB_BASE = {vendor_base!r}\n"
        "runner.POLL_SECONDS = 0\n"
        "try:\n"
        "    sys.exit(runner.main(sys.argv[1:]))\n"
        "except runner.OneshotError as exc:\n"
        "    print(f'[FAIL] {exc}', file=sys.stderr)\n"
        "    sys.exit(1)\n",
        encoding="utf-8",
    )
    return path


class CompleteTracerTests(unittest.IsolatedAsyncioTestCase):
    async def test_plan_submit_generate_and_retrieve_offline(self):
        image = build_png(ARTIFACT_WIDTH, ARTIFACT_HEIGHT)
        vendor = FakeVendor(image)
        vendor.start()
        self.addCleanup(vendor.stop)
        with tempfile.TemporaryDirectory() as tmp:
            package = copy_package(f"{tmp}/package")
            manifest = json.loads(
                (package / "manifest.json").read_text(encoding="utf-8")
            )
            endpoints = json.loads(
                (package / "endpoints.json").read_text(encoding="utf-8")
            )
            state_dir = f"{tmp}/state"
            output_dir = f"{tmp}/output"
            wrapper = write_runner_wrapper(
                Path(tmp) / "runner_wrapper.py",
                state_dir,
                vendor.base_url,
            )
            config = ServiceConfig(
                package_root=package,
                state_dir=state_dir,
                output_dir=output_dir,
                generation_enabled=True,
                bridge_timeout_seconds=30.0,
                bridge_poll_timeout_seconds=10,
                shutdown_timeout_seconds=10.0,
            )
            script = code_reply(
                "plan = run_oneshot_preview(\n"
                '    template="unholy-sdxl",\n'
                f"    prompt={PROMPT!r},\n"
                "    seed=42,\n"
                "    width=832,\n"
                "    height=1216,\n"
                ")\n"
                "final_answer({\n"
                '    "template": "unholy-sdxl",\n'
                f"    \"prompt\": {PROMPT!r},\n"
                '    "seed": 42,\n'
                '    "width": 832,\n'
                '    "height": 1216,\n'
                "})"
            )
            runtime = MediaRuntime(
                config,
                model_factory=lambda: ScriptedModel([script]),
                env=TEST_ENV,
                runner_prefix=(sys.executable, str(wrapper)),
            )
            server = make_server(runtime)
            async with asgi_server(server) as running:
                async with running.client(headers=auth_headers()) as client:
                    plan = await client.call_tool("plan_media", {"request": PROMPT})
                    self.assertEqual(plan.structured_content["diagnostics"]["validated_previews"], 1)
                    body = {
                        "plan": plan.structured_content["plan"],
                        "package_digest": plan.structured_content["package_digest"],
                        "request_id": "tracer-request-1",
                    }
                    first, second = await asyncio.gather(
                        client.call_tool("start_generation", body),
                        client.call_tool("start_generation", body),
                    )
                    self.assertEqual(
                        first.structured_content["job_id"],
                        second.structured_content["job_id"],
                    )
                    job_id = first.structured_content["job_id"]
                    view = await self.wait_for_status(runtime, job_id)
                    self.assertEqual(view.status, "succeeded")
                    self.assertEqual(view.vendor_job_id, VENDOR_ID)
                    self.assertEqual(view.plan["seed"], 42)
                    metadata = view.artifacts[0]
                    self.assertEqual(metadata.mime_type, "image/png")
                    self.assertEqual(metadata.width, ARTIFACT_WIDTH)
                    self.assertEqual(metadata.height, ARTIFACT_HEIGHT)
                    self.assertEqual(metadata.size_bytes, len(image))
                    self.assertEqual(
                        metadata.sha256, hashlib.sha256(image).hexdigest()
                    )

                    contents = await client.read_resource(
                        f"media://jobs/{job_id}/artifacts/{metadata.artifact_id}"
                    )
                    self.assertEqual(contents[0].mime_type, "image/png")
                    self.assertEqual(base64.b64decode(contents[0].blob), image)

                    fetched = await client.call_tool(
                        "get_generation", {"job_id": job_id}
                    )
                    self.assertEqual(fetched.structured_content["status"], "succeeded")

                async with running.http_client(headers=auth_headers()) as http:
                    response = await http.get(
                        f"/artifacts/{job_id}/{metadata.artifact_id}"
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers["content-type"], "image/png")
                    self.assertEqual(response.content, image)
                    self.assertEqual(
                        hashlib.sha256(response.content).hexdigest(), metadata.sha256
                    )
                async with running.http_client() as unauthenticated:
                    denied = await unauthenticated.get(
                        f"/artifacts/{job_id}/{metadata.artifact_id}"
                    )
                    self.assertEqual(denied.status_code, 401)

        self.assertEqual(len(vendor.posts), 1, "duplicate request ids submitted twice")
        post = vendor.posts[0]
        endpoint_id = endpoints[VALID_PLAN["template"]]["endpoint_id"]
        self.assertEqual(post["path"], f"/v2/{endpoint_id}/run")
        self.assertEqual(post["authorization"], f"Bearer {TEST_KEY}")
        self.assertNotIn(TEST_KEY.encode(), post["body"])
        envelope = json.loads(post["body"])
        workflow = envelope["input"]["workflow"]
        template = manifest["templates"][VALID_PLAN["template"]]
        seed = template["seed"]
        size = template["size"]
        positive = template["positive"]
        self.assertEqual(workflow[seed["node"]]["inputs"][seed["field"]], 42)
        self.assertEqual(workflow[size["node"]]["inputs"][size["width_field"]], 832)
        self.assertEqual(workflow[size["node"]]["inputs"][size["height_field"]], 1216)
        self.assertEqual(workflow[positive["node"]]["inputs"][positive["field"]], PROMPT)
        self.assertEqual(vendor.unexpected, [])
        self.assertGreaterEqual(len(vendor.gets), 1)

    async def wait_for_status(self, runtime, job_id, timeout=30.0):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            view = await asyncio.to_thread(runtime.get_generation, job_id)
            if view.status in ("succeeded", "failed", "needs_attention"):
                return view
            await asyncio.sleep(0.05)
        raise AssertionError(f"job {job_id} did not finish before the tracer timeout")


if __name__ == "__main__":
    unittest.main()

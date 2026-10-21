#!/usr/bin/env python3
"""Run a oneshot ComfyUI workflow template on a RunPod serverless endpoint.

Standalone: standard library only, no imports from the maintainer's repo. The
package root is this script's parent's parent (scripts/..). Read manifest.json for
template metadata (override node ids, model family) and endpoints.json for the
template -> endpoint id mapping. The RunPod API key comes from the RUNPOD_API_KEY
environment variable ONLY — never a file.

Override flags are applied at the manifest-declared nodes, so callers never
hand-edit the workflow JSON. RunPod caps the whole request body (not one file):
10 MB for /run and 20 MB for /runsync, shared by the workflow and every
base64-embedded input image.

Usage:
    python3 scripts/rp_oneshot.py --template <name> [overrides] [--wait] [--dry-run]

    Overrides: --prompt T   --negative T   --seed N|random   --width N   --height N
               --set NODE.FIELD=VALUE (repeatable)   --image PATH (repeatable)
    Run:       --output DIR (required)  --sync (/runsync)  --wait (poll to completion)
               --timeout-s N  --project NAME (subdirectory of --output)
               --dry-run (print envelope, no network, no key)

Exit codes: 0 job completed and outputs downloaded; 1 job failed/cancelled/
timed out or a bad override; 2 missing RUNPOD_API_KEY.

Python 3.9+, standard library only.
"""

import argparse
import base64
import json
import mimetypes
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PACKAGE_ROOT / "manifest.json"
ENDPOINTS_PATH = PACKAGE_ROOT / "endpoints.json"
WORKFLOWS_DIR = PACKAGE_ROOT / "workflows"

RUNPOD_JOB_BASE = "https://api.runpod.ai/v2"
MAX_RUN_REQUEST_BYTES = 10 * 1024 * 1024
MAX_RUNSYNC_REQUEST_BYTES = 20 * 1024 * 1024
POLL_SECONDS = 5
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
OUTPUT_KEYS = ("images", "videos", "gifs", "audio", "files")
DEFAULT_EXTENSIONS = {"images": ".png", "videos": ".mp4", "gifs": ".gif", "audio": ".wav"}
RETRIES = 2


class OneshotError(Exception):
    """User-facing error; the message names the fix."""


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise OneshotError(f"missing {path} — the package is incomplete; "
                           "get a fresh copy from the maintainer")
    except ValueError as exc:
        raise OneshotError(f"{path} is not valid JSON: {exc}")


def http_json(method, url, payload=None, key=None, timeout=120, retry=True):
    """One JSON request, retried on 5xx/network errors when `retry` (GETs only
    in practice: a retried submission could enqueue a second billable job)."""
    last_error = None
    for attempt in range(RETRIES + 1):
        try:
            data = None if payload is None else json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(url, data=data, method=method)
            request.add_header("Content-Type", "application/json")
            if key:
                request.add_header("Authorization", f"Bearer {key}")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            if retry and 500 <= exc.code < 600 and attempt < RETRIES:
                last_error = f"HTTP {exc.code}: {body}"
                continue
            raise OneshotError(f"HTTP {exc.code} on {method} {url}: {body}")
        except (urllib.error.URLError, TimeoutError) as exc:
            if retry and attempt < RETRIES:
                last_error = str(exc)
                time.sleep(2 * (attempt + 1))
                continue
            if payload is not None:
                raise OneshotError(
                    f"network error submitting to {url}: {exc} — the job MAY have been "
                    "accepted anyway; check the endpoint's jobs on RunPod before rerunning"
                )
            raise OneshotError(f"network error on {method} {url}: {exc}")
    raise OneshotError(f"retries exhausted on {method} {url}: {last_error}")


def set_input(graph, node_id, field, value):
    if node_id not in graph:
        raise OneshotError(f"node {node_id!r} not in the template — check "
                           "manifest.json override nodes")
    graph[node_id]["inputs"][field] = value


def apply_overrides(workflow, meta, args):
    """Apply --prompt/--negative/--seed/--width/--height/--set at manifest nodes."""
    for section, attr in (("positive", "prompt"), ("negative", "negative")):
        text = getattr(args, attr)
        if text is not None:
            spec = meta.get(section) or {}
            set_input(workflow, spec.get("node"), spec.get("field", "text"), text)
    if args.seed is not None or (meta.get("default_seed_policy") == "random"):
        spec = meta.get("seed") or {}
        if args.seed in (None, "random"):
            seed = random.randint(0, 2**48)
        else:
            seed = int(args.seed)
            if not 0 <= seed < 2**64:
                raise OneshotError(f"--seed {seed}: must be 0..{2**64 - 1}")
        set_input(workflow, spec.get("node"), spec.get("field", "seed"), seed)
    if args.width is not None or args.height is not None:
        spec = meta.get("size") or {}
        if not spec.get("node"):
            raise OneshotError("--width/--height: this template declares no size node")
        if args.width is not None:
            set_input(workflow, spec["node"], spec.get("width_field", "width"), args.width)
        if args.height is not None:
            set_input(workflow, spec["node"], spec.get("height_field", "height"), args.height)
    for override in args.set or []:
        if "=" not in override or "." not in override.split("=", 1)[0]:
            raise OneshotError(f"--set {override!r}: expected NODE.FIELD=VALUE")
        node_field, raw_value = override.split("=", 1)
        node_id, field = node_field.split(".", 1)
        value = int(raw_value) if re.fullmatch(r"-?\d+", raw_value) else raw_value
        set_input(workflow, node_id, field, value)


def encode_image(path, limit_bytes):
    target = Path(path)
    if not target.is_file():
        raise OneshotError(f"input image not found: {target}")
    payload = target.read_bytes()
    if len(payload) > limit_bytes:
        raise OneshotError(
            f"{target} is {len(payload)} bytes; the whole request body is capped at "
            f"{limit_bytes} bytes — shrink the image or host it at a URL instead"
        )
    mime = mimetypes.guess_type(str(target))[0] or "image/png"
    return {
        "name": target.name,
        "image": f"data:{mime};base64," + base64.b64encode(payload).decode("ascii"),
    }


def output_dir(base, project, job_id):
    if project:
        return Path(base) / project / job_id
    return Path(base) / job_id


def describe_output(item):
    """Normalize one worker output entry to {'filename', 'data'|'url'} (or None)."""
    if isinstance(item, str):
        if item.startswith("http"):
            return {"filename": Path(item.split("?")[0]).name, "url": item}
        return {"filename": None, "data": item}
    if not isinstance(item, dict):
        return None
    filename = item.get("filename") or item.get("name")
    url = item.get("url") or item.get("video_url") or item.get("image_url")
    if url:
        return {"filename": filename, "url": url}
    data = item.get("data") or item.get("image")
    if str(item.get("type", "")).lower() == "s3_url":
        return {"filename": filename, "url": data} if data else None
    if data:
        return {"filename": filename, "data": data}
    return None


def collect_outputs(output):
    """All recognized artifacts from a worker response, tagged with key/index."""
    items = []
    if isinstance(output, dict):
        for key in OUTPUT_KEYS:
            for index, item in enumerate(output.get(key) or []):
                described = describe_output(item)
                if described:
                    described.setdefault("key", key)
                    described.setdefault("index", index)
                    items.append(described)
        if not items and "output" in output:
            return collect_outputs(output["output"])
    elif isinstance(output, list):
        for index, item in enumerate(output):
            described = describe_output(item)
            if described:
                described.setdefault("key", "output")
                described.setdefault("index", index)
                items.append(described)
    return items


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a oneshot ComfyUI workflow template on RunPod."
    )
    parser.add_argument("--template", required=True, help="template name (from manifest.json)")
    parser.add_argument("--prompt", help="replace the positive prompt text")
    parser.add_argument("--negative", help="replace the negative prompt text")
    parser.add_argument("--seed", help="integer, or 'random' (default: random)")
    parser.add_argument("--width", type=int, help="output width")
    parser.add_argument("--height", type=int, help="output height")
    parser.add_argument("--set", action="append", metavar="NODE.FIELD=VALUE",
                        help="raw override, repeatable")
    parser.add_argument("--image", action="append", help="input image file, repeatable")
    parser.add_argument("--sync", action="store_true", help="use /runsync instead of /run")
    parser.add_argument("--wait", action="store_true", help="poll until the job ends")
    parser.add_argument("--timeout-s", type=int, default=1800, help="wait timeout (default 1800)")
    parser.add_argument("--output", required=True, metavar="DIR",
                        help="directory for this job's outputs (required)")
    parser.add_argument("--project", help="save under a subdirectory of --output")
    parser.add_argument("--dry-run", action="store_true", help="print the envelope; no network")
    args = parser.parse_args(argv)
    if args.timeout_s <= 0:
        raise OneshotError(f"--timeout-s {args.timeout_s}: must be positive")

    manifest = load_json(MANIFEST_PATH)
    endpoints = load_json(ENDPOINTS_PATH)
    templates = manifest.get("templates") or {}
    if args.template not in templates:
        raise SystemExit(
            f"unknown template {args.template!r}; shipped: {', '.join(sorted(templates)) or '(none)'}"
        )
    if args.template not in endpoints:
        raise OneshotError(f"no endpoint recorded for template {args.template!r} — "
                           "get a refreshed package from the maintainer")
    meta = templates[args.template]
    workflow = load_json(WORKFLOWS_DIR / f"{args.template}.json")
    apply_overrides(workflow, meta, args)

    limit = MAX_RUNSYNC_REQUEST_BYTES if args.sync else MAX_RUN_REQUEST_BYTES
    images = [encode_image(p, limit) for p in args.image or []]
    envelope = {"input": {"workflow": workflow}}
    if images:
        envelope["input"]["images"] = images
    encoded = json.dumps(envelope).encode("utf-8")
    route = "runsync" if args.sync else "run"
    oversized = len(encoded) > limit

    if args.dry_run:
        print(json.dumps(envelope, indent=2))
        if oversized:
            raise OneshotError(
                f"request body is {len(encoded)} bytes; the {route} cap is {limit} — "
                "shrink input images or pass URLs"
            )
        return 0

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("[FAIL] RUNPOD_API_KEY is not set — export it before running jobs",
              file=sys.stderr)
        return 2

    endpoint_id = endpoints[args.template]["endpoint_id"]
    if oversized:
        raise OneshotError(
            f"request body is {len(encoded)} bytes; the {route} cap is {limit} — "
            "shrink input images or pass URLs"
        )
    submitted = http_json("POST", f"{RUNPOD_JOB_BASE}/{endpoint_id}/{route}",
                          envelope, key=api_key, retry=False)
    job_id = submitted.get("id")
    if not job_id:
        raise OneshotError(f"no job id in submission response: {submitted}")

    if not args.wait and not args.sync:
        print(job_id)
        return 0
    if args.sync and submitted.get("status") in TERMINAL_STATES:
        remote = submitted
    else:
        deadline = time.monotonic() + args.timeout_s
        remote = None
        while time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
            status = http_json("GET", f"{RUNPOD_JOB_BASE}/{endpoint_id}/status/{job_id}",
                              key=api_key)
            if status.get("status") in TERMINAL_STATES:
                remote = status
                break
            print(f"[{time.strftime('%H:%M:%S')}] {status.get('status')}", file=sys.stderr)
        if remote is None:
            raise OneshotError(f"job {job_id} did not finish within {args.timeout_s} s — "
                               "note the id; the output may still appear on RunPod")

    state = remote.get("status")
    if state != "COMPLETED":
        print(f"[FAIL] job {job_id} ended {state}: "
              f"{json.dumps(remote.get('error') or remote.get('output') or remote)}",
              file=sys.stderr)
        return 1

    destination = output_dir(args.output, args.project, job_id)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "response.json").write_text(json.dumps(remote, indent=2), encoding="utf-8")
    saved = 0
    unrecognized = 0
    for item in collect_outputs(remote.get("output")):
        key = item.get("key", "output")
        index = item.get("index", 0)
        filename = Path(item.get("filename")
                        or f"{key}_{index:03d}{DEFAULT_EXTENSIONS.get(key, '')}").name
        target = destination / filename
        try:
            if item.get("url"):
                with urllib.request.urlopen(item["url"], timeout=300) as response:
                    target.write_bytes(response.read())
            elif item.get("data"):
                data = item["data"]
                if data.startswith("data:") and "," in data:
                    data = data.split(",", 1)[1]
                target.write_bytes(base64.b64decode(data))
            else:
                unrecognized += 1
                continue
        except (ValueError, OSError, urllib.error.URLError) as exc:
            print(f"[warn] could not write {target.name}: {exc}", file=sys.stderr)
            continue
        print(f"saved {target}")
        saved += 1
    if unrecognized:
        print(f"[warn] {unrecognized} output item(s) not recognized; see response.json",
              file=sys.stderr)
    print(destination.resolve())
    return 0 if saved else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OneshotError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
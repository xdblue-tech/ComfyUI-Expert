#!/usr/bin/env python3
"""Run a oneshot ComfyUI workflow template on a RunPod serverless endpoint.

Standalone: standard library only, no imports from the maintainer's repo. The
package root is this script's parent's parent (bin/..). Read manifest.json for
template metadata (override node ids, model family) and endpoints.json for the
template -> endpoint id mapping. The RunPod API key comes from the RUNPOD_API_KEY
environment variable ONLY — never a file.

Override flags are applied at the manifest-declared nodes, so callers never
hand-edit the workflow JSON. Everything is validated before a request leaves the
machine: template, endpoint mapping, seed range, dimensions (positive, multiple
of 8), and image input support (manifest templates declare an `image` node only
when the workflow accepts one). RunPod caps the whole request body (not one
file): 10 MB for /run and 20 MB for /runsync, shared by the workflow and every
base64-embedded input image.

Usage:
    python3 bin/rp_oneshot.py --template <name> [overrides] [--wait] [--dry-run]

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
MAX_DIMENSION = 16384
DIMENSION_STEP = 8
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
OUTPUT_KEYS = ("images", "videos", "gifs", "audio", "files")
DEFAULT_EXTENSIONS = {"images": ".png", "videos": ".mp4", "gifs": ".gif", "audio": ".wav"}
RETRIES = 2
# Fixed markers the host bridge parses. Change them and bridge.py must change too:
# a wrong guess here can make a caller submit a second billable job.
SUBMISSION_ACCEPTED = "[info] job {job_id} accepted"
SUBMISSION_UNKNOWN = "submission outcome is UNKNOWN"


class OneshotError(Exception):
    """User-facing error; the message names the fix."""


def load_json(path, label="file"):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise OneshotError(f"missing {label} {path} — the package is incomplete; "
                           "get a fresh copy from the maintainer")
    except ValueError as exc:
        raise OneshotError(f"{label} {path} is not valid JSON: {exc}")


def require_mapping(data, label, path):
    if not isinstance(data, dict):
        raise OneshotError(f"{label} {path} must be a JSON object, got "
                           f"{type(data).__name__}")
    return data


def _http_open(target, timeout):
    """The single network seam. Tests replace it to exercise the real logic offline."""
    return urllib.request.urlopen(target, timeout=timeout)


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
            with _http_open(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            if exc.code in (401, 403):
                raise OneshotError(
                    f"authentication failed (HTTP {exc.code}) on {method} {url}: {body} — "
                    "check that RUNPOD_API_KEY is the right key for this endpoint"
                )
            if retry and 500 <= exc.code < 600 and attempt < RETRIES:
                last_error = f"HTTP {exc.code}: {body}"
                continue
            # A server-side failure on a POST can still have enqueued the job.
            outcome = ""
            if payload is not None and exc.code >= 500:
                outcome = f" — {SUBMISSION_UNKNOWN}"
            raise OneshotError(f"HTTP {exc.code} on {method} {url}: {body}{outcome}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if retry and attempt < RETRIES:
                last_error = str(exc)
                time.sleep(2 * (attempt + 1))
                continue
            if payload is not None:
                raise OneshotError(
                    f"network error submitting to {url}: {exc} — the job MAY have been "
                    "accepted anyway; check the endpoint's jobs on RunPod before rerunning "
                    f"— {SUBMISSION_UNKNOWN}"
                )
            raise OneshotError(f"network error on {method} {url}: {exc}")
        try:
            return json.loads(raw)
        except ValueError as exc:
            outcome = ""
            if payload is not None:
                outcome = (
                    f" — {SUBMISSION_UNKNOWN}; check the endpoint's jobs before rerunning"
                )
            raise OneshotError(
                f"malformed JSON from {method} {url}: {exc}; body starts {raw[:120]!r}{outcome}"
            )
    raise OneshotError(f"retries exhausted on {method} {url}: {last_error}")


def set_input(graph, node_id, field, value):
    if node_id not in graph:
        raise OneshotError(f"node {node_id!r} not in the template — check "
                           "manifest.json override nodes")
    inputs = graph[node_id].get("inputs")
    if not isinstance(inputs, dict):
        raise OneshotError(f"node {node_id!r} has no inputs object — check "
                           "manifest.json override nodes")
    inputs[field] = value


def parse_seed(raw):
    """Return a concrete seed; reject anything that is not an integer in range."""
    if raw in (None, "random"):
        return random.randint(0, 2**48)
    text = str(raw).strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise OneshotError(
            f"--seed {raw!r}: expected a non-negative integer or 'random'"
        )
    seed = int(text)
    if seed >= 2**64:
        raise OneshotError(f"--seed {seed}: must be 0..{2**64 - 1}")
    return seed


def validate_dimension(flag, value):
    if value is None:
        return
    if value <= 0 or value > MAX_DIMENSION:
        raise OneshotError(
            f"{flag} {value}: must be between 1 and {MAX_DIMENSION} pixels"
        )
    if value % DIMENSION_STEP:
        raise OneshotError(
            f"{flag} {value}: must be a multiple of {DIMENSION_STEP} pixels"
        )


def apply_overrides(workflow, meta, args):
    """Apply --prompt/--negative/--seed/--width/--height/--set at manifest nodes."""
    for section, attr, flag in (("positive", "prompt", "--prompt"),
                                ("negative", "negative", "--negative")):
        text = getattr(args, attr)
        if text is not None:
            spec = meta.get(section) or {}
            if not spec.get("node"):
                raise OneshotError(
                    f"{flag}: template {args.template!r} declares no {section} node "
                    "in manifest.json"
                )
            set_input(workflow, spec["node"], spec.get("field", "text"), text)
    if args.seed is not None or (meta.get("default_seed_policy") == "random"):
        spec = meta.get("seed") or {}
        if not spec.get("node"):
            raise OneshotError(
                f"--seed: template {args.template!r} declares no seed node in manifest.json"
            )
        set_input(workflow, spec["node"], spec.get("field", "seed"),
                  parse_seed(args.seed))
    if args.width is not None or args.height is not None:
        spec = meta.get("size") or {}
        if not spec.get("node"):
            raise OneshotError(
                f"--width/--height: template {args.template!r} declares no size node in "
                "manifest.json"
            )
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
    if "data" in item or "image" in item:
        return {"filename": filename, "data": data if data is not None else ""}
    return None


def _empty_entry(item):
    """Keep the reported filename when an entry carries neither data nor a URL."""
    name = (item.get("filename") or item.get("name")) if isinstance(item, dict) else None
    return {"filename": name}


def collect_outputs(output):
    """Every artifact entry from a worker response, tagged with key/index.

    Entries that carry neither data nor a URL are kept so the caller reports
    them instead of silently dropping part of the response.
    """
    items = []
    if isinstance(output, dict):
        for key in OUTPUT_KEYS:
            for index, item in enumerate(output.get(key) or []):
                described = describe_output(item) or _empty_entry(item)
                described.setdefault("key", key)
                described.setdefault("index", index)
                items.append(described)
        if not items and "output" in output:
            return collect_outputs(output["output"])
    elif isinstance(output, list):
        for index, item in enumerate(output):
            described = describe_output(item) or _empty_entry(item)
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
    validate_dimension("--width", args.width)
    validate_dimension("--height", args.height)

    manifest = require_mapping(load_json(MANIFEST_PATH, "manifest"),
                               "manifest", MANIFEST_PATH)
    endpoints = require_mapping(load_json(ENDPOINTS_PATH, "endpoint snapshot"),
                                "endpoint snapshot", ENDPOINTS_PATH)
    templates = manifest.get("templates")
    if not isinstance(templates, dict) or not templates:
        raise OneshotError(f"manifest {MANIFEST_PATH} has no 'templates' object — "
                           "the package is incomplete")
    if args.template not in templates:
        raise OneshotError(
            f"unknown template {args.template!r}; shipped: "
            f"{', '.join(sorted(templates)) or '(none)'}"
        )
    if args.template not in endpoints:
        raise OneshotError(f"no endpoint recorded for template {args.template!r} — "
                           "get a refreshed package from the maintainer")
    meta = templates[args.template]
    if not isinstance(meta, dict):
        raise OneshotError(f"manifest template {args.template!r} must be a JSON object")
    if args.image and not (meta.get("image") or {}).get("node"):
        raise OneshotError(
            f"--image: template {args.template!r} is text-only — manifest.json declares "
            "no image node for it; drop --image or use a template that accepts one"
        )
    workflow = load_json(WORKFLOWS_DIR / f"{args.template}.json",
                         f"workflow for template {args.template!r}")
    if not isinstance(workflow, dict):
        raise OneshotError(f"workflow for template {args.template!r} must be a JSON object")
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

    if oversized:
        raise OneshotError(
            f"request body is {len(encoded)} bytes; the {route} cap is {limit} — "
            "shrink input images or pass URLs"
        )
    endpoint_entry = endpoints[args.template]
    if not isinstance(endpoint_entry, dict) or not endpoint_entry.get("endpoint_id"):
        raise OneshotError(f"no endpoint recorded for template {args.template!r} — "
                           "get a refreshed package from the maintainer")
    endpoint_id = endpoint_entry["endpoint_id"]

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("[FAIL] RUNPOD_API_KEY is not set — export it before running jobs",
              file=sys.stderr)
        return 2

    submitted = http_json("POST", f"{RUNPOD_JOB_BASE}/{endpoint_id}/{route}",
                          envelope, key=api_key, retry=False)
    if not isinstance(submitted, dict):
        raise OneshotError(
            f"submission response from {RUNPOD_JOB_BASE} must be a JSON object, got "
            f"{type(submitted).__name__} — {SUBMISSION_UNKNOWN}"
        )
    job_id = submitted.get("id")
    if not job_id:
        raise OneshotError(
            f"no job id in submission response: {submitted} — {SUBMISSION_UNKNOWN}"
        )
    print(SUBMISSION_ACCEPTED.format(job_id=job_id), file=sys.stderr)

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
            if not isinstance(status, dict):
                raise OneshotError(
                    f"status response for job {job_id} must be a JSON object, got "
                    f"{type(status).__name__}"
                )
            if status.get("status") in TERMINAL_STATES:
                remote = status
                break
            print(f"[{time.strftime('%H:%M:%S')}] {status.get('status')}", file=sys.stderr)
        if remote is None:
            raise OneshotError(f"job {job_id} did not finish within {args.timeout_s} s — "
                               "note the id; the output may still appear on RunPod")

    state = remote.get("status")
    if state != "COMPLETED":
        detail = json.dumps(remote.get("error") or remote.get("output") or remote)
        if state == "FAILED":
            message = f"job {job_id} failed: {detail}"
        elif state == "CANCELLED":
            message = f"job {job_id} was cancelled before it finished: {detail}"
        elif state == "TIMED_OUT":
            message = f"job {job_id} timed out on the worker: {detail}"
        else:
            message = f"job {job_id} ended in unexpected state {state!r}: {detail}"
        print(f"[FAIL] {message}", file=sys.stderr)
        return 1

    destination = output_dir(args.output, args.project, job_id)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "response.json").write_text(json.dumps(remote, indent=2), encoding="utf-8")
    items = collect_outputs(remote.get("output"))
    if not items:
        print(f"[FAIL] job {job_id} completed but the response carried no outputs; "
              f"see {destination / 'response.json'}", file=sys.stderr)
        return 1
    saved = 0
    failures = []
    for item in items:
        key = item.get("key", "output")
        index = item.get("index", 0)
        filename = Path(item.get("filename")
                        or f"{key}_{index:03d}{DEFAULT_EXTENSIONS.get(key, '')}").name
        target = destination / filename
        try:
            if item.get("url"):
                with _http_open(item["url"], timeout=300) as response:
                    payload = response.read()
            elif item.get("data"):
                data = item["data"]
                if data.startswith("data:") and "," in data:
                    data = data.split(",", 1)[1]
                # Workers may wrap base64; whitespace is not payload, other junk is.
                payload = base64.b64decode("".join(data.split()), validate=True)
            else:
                failures.append(f"{filename}: unrecognized item; see response.json")
                continue
        except (ValueError, OSError, urllib.error.URLError) as exc:
            failures.append(f"{filename}: {exc}")
            continue
        if not payload:
            failures.append(f"{filename}: empty payload")
            continue
        try:
            target.write_bytes(payload)
        except OSError as exc:
            failures.append(f"{filename}: could not write {target}: {exc}")
            continue
        print(f"saved {target}")
        saved += 1
    if failures:
        print(f"[FAIL] saved {saved} of {len(items)} output(s); failed: "
              + "; ".join(failures), file=sys.stderr)
        return 1
    print(destination.resolve())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OneshotError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
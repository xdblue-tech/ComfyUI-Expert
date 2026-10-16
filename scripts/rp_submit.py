#!/usr/bin/env python3
"""Submit a ComfyUI API-format workflow to a RunPod serverless ComfyUI endpoint.

The workflow rides the job payload (no image rebuild needed). The envelope is the
worker-comfyui input contract:

    {"input": {"workflow": {...},
               "images": [{"name": "in.png", "image": "data:image/png;base64,..."}]}}

`images` appears only when --image is given. RunPod caps the whole request body, not
one file: 10 MB for /run and 20 MB for /runsync, so base64 images plus the workflow
share that budget. rp_submit.py checks the serialized envelope before POSTing; larger
inputs should be passed as URLs (the worker fetches http(s) URLs itself).

Usage:
    python3 scripts/rp_submit.py --endpoint <id-or-name> --workflow <path.json>
        [--image in.png ...] [--project NAME] [--sync] [--wait] [--timeout-s N] [--dry-run]

--sync uses /runsync (single held request); otherwise /run + /status polling with
--wait. Video jobs run 10-20 minutes, so the default wait is 30 minutes.

Python 3.9+, standard library only.
"""

import argparse
import base64
import json
import mimetypes
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rp_common import (  # noqa: E402  (path set up above)
    ConfigError,
    ENDPOINTS_STATE,
    REPO_ROOT,
    RUNPOD_API_BASE,
    RUNPOD_JOB_BASE,
    api_key,
    api_request,
    encode_payload,
    load_config,
    load_state,
    utc_now,
)

# Whole-request limits, shared by the workflow and every attached image
# (worker-comfyui README: "10MB for /run, 20MB for /runsync").
MAX_RUN_REQUEST_BYTES = 10 * 1024 * 1024
MAX_RUNSYNC_REQUEST_BYTES = 20 * 1024 * 1024
POLL_SECONDS = 5
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
OUTPUT_KEYS = ("images", "videos", "gifs", "audio", "files")
DEFAULT_EXTENSIONS = {"images": ".png", "videos": ".mp4", "gifs": ".gif", "audio": ".wav"}


def load_workflow(path):
    """Load and sanity-check a ComfyUI API-format workflow."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"workflow file not found: {path}")
    try:
        workflow = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read workflow {path}: {exc}")
    if not isinstance(workflow, dict) or not workflow:
        raise ConfigError(f"{path} is not a ComfyUI API-format workflow object")
    if "nodes" in workflow and isinstance(workflow.get("nodes"), list):
        raise ConfigError(
            f"{path} is UI-format; export it with Workflow > Export (API), then retry"
        )
    if "workflow" in workflow or set(workflow) == {"input"}:
        raise ConfigError(
            f"{path} already looks wrapped for RunPod; pass the bare API prompt map "
            "(the object with node ids at the top level)"
        )
    if not any(isinstance(node, dict) and "class_type" in node for node in workflow.values()):
        raise ConfigError(f"{path} has no nodes with class_type; is this an API prompt?")
    return workflow


def request_limit(sync):
    """(byte limit, route) for the chosen submit route."""
    if sync:
        return MAX_RUNSYNC_REQUEST_BYTES, "/runsync"
    return MAX_RUN_REQUEST_BYTES, "/run"


def encode_image(path, limit_bytes, route):
    """Return {'name', 'image'} for one input image, or raise when it cannot fit."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"input image not found: {path}")
    size = path.stat().st_size
    # Preliminary guard: base64 is 4 bytes per 3 raw bytes, so refuse before encoding a
    # file that could not fit the whole-request limit even as the only image.
    if size * 4 // 3 >= limit_bytes:
        raise ConfigError(
            f"{path} is {size / 1048576:.1f} MB; base64 would exceed even the "
            f"{limit_bytes / 1048576:.0f} MB whole-request limit for {route}. Upload it "
            "somewhere reachable and pass the http(s) URL as the image value instead."
        )
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"name": path.name, "image": f"data:{mime};base64,{payload}"}


def build_envelope(workflow, images):
    body = {"workflow": workflow}
    if images:
        body["images"] = images
    return {"input": body}


def check_request_size(envelope, sync):
    """Raise when the serialized envelope exceeds the route's documented request limit.

    RunPod applies the limit to the whole body, so several individually-small images
    can still overflow it once base64 and the workflow are counted. Returns the body size.
    """
    limit_bytes, route = request_limit(sync)
    size = len(encode_payload(envelope))
    if size > limit_bytes:
        raise ConfigError(
            f"serialized envelope is {size / 1048576:.1f} MB, over the "
            f"{limit_bytes / 1048576:.0f} MB request limit for {route}: trim the workflow "
            "or pass input images as http(s) URLs"
        )
    return size


def resolve_endpoint(value, state, config, allow_network=True):
    """Resolve an endpoint name to an id, falling back to a live list when allowed.

    With allow_network=False the registry is the only source: --dry-run must not call
    RunPod, so an unknown target is returned exactly as the caller typed it.
    """
    records = (state or {}).get("endpoints") or {}
    if value in records:
        return value
    for endpoint_id, record in records.items():
        if record.get("name") == value:
            return endpoint_id
    if not allow_network:
        return value
    try:
        key = api_key(config, required=False)
    except ConfigError:
        key = None
    if key:
        for remote in api_request("GET", f"{RUNPOD_API_BASE}/endpoints", key=key) or []:
            if remote.get("name") == value and remote.get("id"):
                return remote["id"]
    if " " not in value and value:
        return value
    raise ConfigError(f"no endpoint matches {value!r}; try: rp_endpoints.py list")


def describe_output(item):
    """Normalize one worker output entry to {'filename', 'data'|'url'}.

    The worker tags each entry with `type`: "base64" (data holds base64) or "s3_url"
    (data holds an https URL), so a URL is never handed to the base64 decoder.
    """
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
    items = []
    if isinstance(output, dict):
        for key in OUTPUT_KEYS:
            value = output.get(key)
            if isinstance(value, list):
                for index, item in enumerate(value):
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


def decode_data(data):
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


def download(url):
    with urllib.request.urlopen(url, timeout=600) as response:
        return response.read()


def save_outputs(output, destination):
    """Write every recognized output artifact. Returns (written, failed) path lists."""
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    failed = []
    for item in collect_outputs(output):
        key = item.get("key", "output")
        index = item.get("index", 0)
        filename = item.get("filename") or f"{key}_{index:03d}{DEFAULT_EXTENSIONS.get(key, '')}"
        target = destination / Path(filename).name
        try:
            if item.get("url"):
                target.write_bytes(download(item["url"]))
            elif item.get("data"):
                target.write_bytes(decode_data(item["data"]))
            else:
                continue
        except (ValueError, OSError) as exc:
            print(f"[warn] could not write {target.name}: {exc}")
            failed.append(target)
            continue
        written.append(target)
        print(f"[OK] wrote {target}")
    return written, failed


def poll(endpoint_id, job_id, key, timeout_s):
    deadline = time.monotonic() + timeout_s
    status = None
    previous = None
    while True:
        payload = api_request(
            "GET", f"{RUNPOD_JOB_BASE}/{endpoint_id}/status/{job_id}", key=key
        ) or {}
        status = payload.get("status", "UNKNOWN")
        if status != previous:
            elapsed = int(timeout_s - (deadline - time.monotonic()))
            print(f"[info] {status} after {elapsed}s")
            previous = status
        if status in TERMINAL_STATES:
            return payload
        if time.monotonic() >= deadline:
            raise ConfigError(
                f"job {job_id} still {status} after {timeout_s}s; check it later with "
                f"GET {RUNPOD_JOB_BASE}/{endpoint_id}/status/{job_id}"
            )
        time.sleep(POLL_SECONDS)


def output_dir(project, job_id):
    if project:
        return REPO_ROOT / "projects" / project / "output" / job_id
    return REPO_ROOT / "state" / "output" / job_id


def submit(args, config, state):
    workflow = load_workflow(args.workflow)
    limit_bytes, route = request_limit(args.sync)
    images = [encode_image(path, limit_bytes, route) for path in args.image or []]
    envelope = build_envelope(workflow, images)
    body_size = check_request_size(envelope, args.sync)
    endpoint_id = resolve_endpoint(
        args.endpoint, state, config, allow_network=not args.dry_run
    )
    run_path = "runsync" if args.sync else "run"
    url = f"{RUNPOD_JOB_BASE}/{endpoint_id}/{run_path}"

    if args.dry_run:
        registered = (state or {}).get("endpoints") or {}
        if endpoint_id not in registered:
            print(
                f"[dry-run] {args.endpoint!r} is not in {ENDPOINTS_STATE}; using it as typed",
                file=sys.stderr,
            )
        printable = {
            "input": {
                "workflow": workflow,
                "images": [
                    {"name": image["name"], "image": f"<base64 {len(image['image'])} chars>"}
                    for image in images
                ],
            }
        }
        if not images:
            printable = envelope
        print(json.dumps(printable, indent=2))
        # Note goes to stderr so `--dry-run | jq .` sees only the envelope.
        print(
            f"[dry-run] would POST {url} ({body_size / 1048576:.2f} MB body, "
            f"{route} limit {limit_bytes // 1048576} MB)",
            file=sys.stderr,
        )
        return 0

    key = api_key(config)
    print(f"[info] POST {url}")
    result = api_request("POST", url, envelope, key) or {}
    job_id = result.get("id")
    if not job_id:
        raise ConfigError(f"submit returned no job id: {result}")
    print(f"[OK] job {job_id} status {result.get('status', 'IN_QUEUE')}")

    status = result.get("status")
    if args.wait or args.sync:
        if status not in TERMINAL_STATES:
            result = poll(endpoint_id, job_id, key, args.timeout_s)
            status = result.get("status")
    else:
        print(
            f"[info] not waiting; poll with: GET "
            f"{RUNPOD_JOB_BASE}/{endpoint_id}/status/{job_id}"
        )
        return 0

    execution = result.get("executionTime")
    if execution is not None:
        print(f"[info] executionTime {execution} ms")
    if status != "COMPLETED":
        print(f"[FAIL] job {job_id} ended {status}")
        error = result.get("error") or result.get("output") or result
        print(json.dumps(error, indent=2)[:2000] if isinstance(error, (dict, list)) else error)
        return 1

    destination = output_dir(args.project, job_id)
    destination.mkdir(parents=True, exist_ok=True)
    # Keep the raw response before extracting, so an unrecognized output shape or a
    # failed download still leaves a recoverable result next to the job record.
    result_path = destination / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written, failed = save_outputs(result.get("output"), destination)
    record = {
        "job_id": job_id,
        "endpoint_id": endpoint_id,
        "status": status,
        "workflow": str(Path(args.workflow).resolve()),
        "images": [image["name"] for image in images],
        "project": args.project or "",
        "execution_time_ms": execution,
        "submitted_at": utc_now(),
        "output_dir": str(destination),
        "outputs": [str(path) for path in written],
    }
    (destination / "job.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[OK] job record {destination / 'job.json'}")
    print(f"[OK] raw response {result_path}")
    if failed:
        print(
            f"[FAIL] {len(failed)} output file(s) could not be written "
            f"({', '.join(path.name for path in failed)}); response kept in {result_path}",
            file=sys.stderr,
        )
        return 1
    if not written:
        print(
            f"[warn] no image/video files recognized in the job output; "
            f"response kept in {result_path}"
        )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Submit a ComfyUI API workflow to a RunPod serverless endpoint."
    )
    parser.add_argument("--endpoint", required=True, help="endpoint id or name")
    parser.add_argument("--workflow", required=True, help="ComfyUI API-format workflow JSON")
    parser.add_argument("--image", action="append", help="input image file; repeatable")
    parser.add_argument("--project", help="save outputs under projects/<name>/output/")
    parser.add_argument("--sync", action="store_true", help="use /runsync instead of /run")
    parser.add_argument("--wait", action="store_true", help="poll /status until the job ends")
    parser.add_argument("--timeout-s", type=int, default=1800, help="wait timeout (default 1800)")
    parser.add_argument("--dry-run", action="store_true", help="print the envelope; no network")
    args = parser.parse_args(argv)

    try:
        config = load_config()
        state = load_state(ENDPOINTS_STATE) or {}
        return submit(args, config, state)
    except ConfigError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

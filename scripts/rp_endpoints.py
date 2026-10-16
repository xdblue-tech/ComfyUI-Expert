#!/usr/bin/env python3
"""Register and sync RunPod serverless endpoints for the baked ComfyUI images.

Every endpoint is created as a template (image + container disk + env) plus an
endpoint (template + GPU types + worker bounds + timeouts) over the REST v1
control plane, then recorded in state/endpoints.json.

The trap this script exists for: RunPod silently reduces an idle endpoint's
max workers (to 2 after ~3 idle days, 0 after ~7). `sync` detects that drop, restores
the recorded workersMax with a PATCH on the v2 serverless route, and confirms the
new value with a fresh read; an endpoint it could not repair or confirm keeps its
`drifted`/`unknown` status and makes `sync` exit nonzero.

Usage:
    python3 scripts/rp_endpoints.py create <image-tag> [--profile video] [--gpu ID ...]
    python3 scripts/rp_endpoints.py sync
    python3 scripts/rp_endpoints.py list
    python3 scripts/rp_endpoints.py health <id-or-name>
    python3 scripts/rp_endpoints.py delete <id-or-name>

The API key comes from RUNPOD_API_KEY or api_key_file in config/rp.yaml.
Python 3.9+, standard library only.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rp_common import (  # noqa: E402  (path set up above)
    ConfigError,
    ENDPOINTS_STATE,
    RUNPOD_API_BASE,
    RUNPOD_JOB_BASE,
    RUNPOD_V2_BASE,
    api_key,
    api_request,
    load_config,
    load_state,
    parse_image_tag,
    state_transaction,
    utc_now,
)

VIDEO_MARKERS = ("video", "wan", "framepack")


def profile_for(tag):
    """Infer the config profile from the image tag (video tags carry a video marker)."""
    lowered = tag.lower()
    return "video" if any(marker in lowered for marker in VIDEO_MARKERS) else "image-gen"


def profile_settings(config, profile):
    profiles = config.get("profiles") or {}
    if profile not in profiles:
        raise ConfigError(
            f"config/rp.yaml has no profile {profile!r}; expected one of "
            f"{', '.join(sorted(profiles))}"
        )
    return profiles[profile]


def endpoint_records(state):
    return state.setdefault("endpoints", {})


def find_endpoint(state, id_or_name, key=None):
    """Resolve an endpoint id or name. Falls back to a live list when the name is unknown."""
    records = endpoint_records(state)
    if id_or_name in records:
        return id_or_name, records[id_or_name]
    for endpoint_id, record in records.items():
        if record.get("name") == id_or_name:
            return endpoint_id, record
    if key:
        for remote in api_request("GET", f"{RUNPOD_API_BASE}/endpoints", key=key) or []:
            if remote.get("name") == id_or_name:
                return remote.get("id"), records.get(remote.get("id"), {})
    if id_or_name and " " not in id_or_name:
        return id_or_name, records.get(id_or_name, {})
    raise ConfigError(f"no endpoint matches {id_or_name!r} in {ENDPOINTS_STATE}")


def remote_workers_max(remote):
    """Read workersMax from an endpoint object (v1 flat `workersMax` or v2 `workers.max`)."""
    if not isinstance(remote, dict):
        return None
    if isinstance(remote.get("workersMax"), int):
        return remote["workersMax"]
    workers = remote.get("workers")
    if isinstance(workers, dict) and isinstance(workers.get("max"), int):
        return workers["max"]
    return None


def read_workers_max(endpoint_id, key):
    """Current workersMax from a single-endpoint GET; None when RunPod does not report one."""
    try:
        remote = api_request(
            "GET", f"{RUNPOD_API_BASE}/endpoints/{endpoint_id}", key=key
        )
    except ConfigError as exc:
        print(f"[warn] could not read {endpoint_id} back: {exc}")
        return None
    return remote_workers_max(remote)


def default_endpoint_name(repo):
    """Endpoint name for an image repository: comfyui-video-wan -> comfyui-video-wan."""
    short = repo.rsplit("/", 1)[-1]
    return short if short.startswith("comfyui-") else f"comfyui-{short}"


def create(args, config, state):
    # The image tag is validated before any remote resource is created.
    repo, _version = parse_image_tag(args.image_tag)
    tag = args.image_tag
    profile = args.profile or profile_for(tag)
    settings = profile_settings(config, profile)
    gpu_types = args.gpu or settings.get("gpu_type_ids")
    if not gpu_types:
        raise ConfigError(f"profile {profile!r} has no gpu_type_ids and no --gpu given")
    key = api_key(config)

    name = args.name or default_endpoint_name(repo)
    env = {}
    for item in args.env or []:
        item_key, sep, value = item.partition("=")
        if not sep:
            raise ConfigError(f"--env expects KEY=VALUE, got {item!r}")
        env[item_key] = value

    template_body = {
        "name": f"{name}-tpl",
        "imageName": tag,
        "containerDiskInGb": args.container_disk_gb or settings.get("container_disk_gb", 50),
        "isServerless": True,
        "env": env,
    }
    template = api_request("POST", f"{RUNPOD_API_BASE}/templates", template_body, key)
    template_id = (template or {}).get("id")
    if not template_id:
        raise ConfigError(f"template create returned no id: {template}")
    print(f"[OK] template {template_id} ({template_body['name']})")

    endpoint_body = {
        "name": name,
        "templateId": template_id,
        "gpuTypeIds": list(gpu_types),
        "gpuCount": 1,
        "workersMin": args.workers_min,
        "workersMax": args.workers_max,
        "idleTimeout": settings.get("idle_timeout_s", 10),
        "executionTimeoutMs": args.exec_timeout or settings.get("execution_timeout_ms"),
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 1,
        "flashboot": bool(config.get("flashboot", True)),
    }
    endpoint = api_request("POST", f"{RUNPOD_API_BASE}/endpoints", endpoint_body, key)
    endpoint_id = (endpoint or {}).get("id")
    if not endpoint_id:
        raise ConfigError(f"endpoint create returned no id: {endpoint}")

    now = utc_now()
    with state_transaction(ENDPOINTS_STATE) as registry:
        endpoint_records(registry)[endpoint_id] = {
            "endpoint_id": endpoint_id,
            "name": name,
            "template_id": template_id,
            "image_tag": tag,
            "gpu_types": list(gpu_types),
            "workers_min": args.workers_min,
            "workers_max": args.workers_max,
            "exec_timeout_ms": endpoint_body["executionTimeoutMs"],
            "status": "active",
            "created_at": now,
            "last_synced_at": now,
        }
    print(f"[OK] endpoint {endpoint_id} ({name}) profile={profile}")
    print(f"     run    POST {RUNPOD_JOB_BASE}/{endpoint_id}/run")
    print(f"     health GET  {RUNPOD_JOB_BASE}/{endpoint_id}/health")
    print(f"     ~${settings.get('hourly_usd', '?')}/hr while a worker is up")
    print(f"[OK] recorded in {ENDPOINTS_STATE}")
    return 0


def sync(args, config, state):
    key = api_key(config)
    remote_list = api_request("GET", f"{RUNPOD_API_BASE}/endpoints", key=key) or []
    remote_by_id = {item.get("id"): item for item in remote_list if item.get("id")}
    now = utc_now()
    unresolved = 0

    with state_transaction(ENDPOINTS_STATE) as registry:
        records = endpoint_records(registry)
        for endpoint_id, record in sorted(records.items()):
            record["last_synced_at"] = now
            if record.get("status") == "deleted":
                continue
            remote = remote_by_id.get(endpoint_id)
            if remote is None:
                record["status"] = "deleted"
                record["drift_note"] = "not in the RunPod account; deleted outside the registry"
                print(f"[warn] {endpoint_id} ({record.get('name')}): deleted outside the registry")
                continue
            wanted_max = record.get("workers_max")
            if not isinstance(wanted_max, int):
                record["status"] = "unknown"
                record["drift_note"] = (
                    "registry has no workers_max; set it with rp_endpoints.py create"
                )
                print(
                    f"[warn] {endpoint_id} ({record.get('name')}): registry has no workers_max"
                )
                unresolved += 1
                continue
            current_max = remote_workers_max(remote)
            if current_max is None:
                # The list view can omit the field; only the single-endpoint read settles it.
                current_max = read_workers_max(endpoint_id, key)
            if current_max is None:
                record["status"] = "unknown"
                record["drift_note"] = (
                    "RunPod did not report workersMax; verify with rp_endpoints.py health"
                )
                print(
                    f"[warn] {endpoint_id} ({record.get('name')}): workersMax unreadable; "
                    "registry left unknown rather than active"
                )
                unresolved += 1
                continue
            if current_max == wanted_max:
                record["status"] = "active"
                record.pop("drift_note", None)
                continue
            record["status"] = "drifted"
            record["drift_note"] = (
                f"RunPod reports workersMax={current_max}, registry has {wanted_max}"
            )
            print(
                f"[drift] {endpoint_id} ({record.get('name')}): workersMax {wanted_max} -> "
                f"{current_max} (idle drop); restoring"
            )
            # Documented update route: the REST v2 serverless PATCH takes a partial body,
            # so only workers.max changes (skills/runpod-migrate/reference/graphql-to-v2.md,
            # skills/runpod/golden-paths/15-monitor-and-debug.md). REST v1 documents no
            # endpoint-update route at all.
            try:
                api_request(
                    "PATCH",
                    f"{RUNPOD_V2_BASE}/serverless/{endpoint_id}",
                    {"workers": {"max": wanted_max}},
                    key,
                )
            except ConfigError as exc:
                print(f"[warn] could not restore workersMax: {exc}")
                print(
                    "       restore it in the Console, or with "
                    f"`runpodctl serverless update {endpoint_id} --workers-max {wanted_max}`"
                )
                unresolved += 1
                continue
            # A PATCH response is not a read: only a fresh GET can confirm the repair.
            confirmed_max = read_workers_max(endpoint_id, key)
            if confirmed_max == wanted_max:
                record["status"] = "active"
                record["drift_note"] = (
                    f"restored workersMax={wanted_max} after an idle drop (confirmed by read-back)"
                )
                print(f"[OK] restored workersMax={wanted_max} on {endpoint_id}")
                continue
            shown = (
                "no workersMax"
                if confirmed_max is None
                else f"workersMax={confirmed_max}"
            )
            record["status"] = "drifted"
            record["drift_note"] = (
                f"PATCH sent workersMax={wanted_max}, read-back shows {shown}"
            )
            print(f"[warn] {endpoint_id}: PATCH sent, read-back shows {shown}")
            unresolved += 1

        print(
            f"[OK] synced {len(records)} endpoints against {len(remote_by_id)} in the account"
        )
    if unresolved:
        print(
            f"[FAIL] {unresolved} endpoint(s) unresolved: not at the recorded workersMax or "
            "not readable; see the drift_note in "
            f"{ENDPOINTS_STATE}",
            file=sys.stderr,
        )
        return 1
    return 0


def list_endpoints(args, config, state):
    records = (state or {}).get("endpoints") or {}
    if not records:
        print(f"no endpoints recorded in {ENDPOINTS_STATE}; run: rp_endpoints.py create ...")
        return 0
    for endpoint_id, record in sorted(records.items()):
        print(
            f"{endpoint_id:16} {str(record.get('name', '-')):28} "
            f"{str(record.get('status', '-')):8} workers {record.get('workers_min', '?')}-"
            f"{record.get('workers_max', '?')} timeout {record.get('exec_timeout_ms', '?')}ms "
            f"{record.get('image_tag', '-')}"
        )
        if record.get("drift_note"):
            print(f"{'':16} note: {record['drift_note']}")
    return 0


def health(args, config, state):
    key = api_key(config)
    endpoint_id, record = find_endpoint(state, args.endpoint, key)
    endpoint = api_request("GET", f"{RUNPOD_API_BASE}/endpoints/{endpoint_id}", key=key) or {}
    print(f"endpoint {endpoint_id} ({endpoint.get('name', record.get('name', '?'))})")
    print(
        f"  workers     {endpoint.get('workersMin', record.get('workers_min'))}-"
        f"{endpoint.get('workersMax', record.get('workers_max'))}"
    )
    print(
        f"  timeouts    idle {endpoint.get('idleTimeout', '?')}s, "
        f"execution {endpoint.get('executionTimeoutMs', record.get('exec_timeout_ms'))}ms"
    )
    template = endpoint.get("template") or {}
    print(f"  image       {template.get('imageName', record.get('image_tag', '?'))}")
    try:
        workers = api_request("GET", f"{RUNPOD_V2_BASE}/serverless/{endpoint_id}/workers", key=key)
    except ConfigError as exc:
        print(f"  live counts unavailable: {exc}")
        return 0
    items = workers.get("workers") if isinstance(workers, dict) else workers
    if not isinstance(items, list):
        print(f"  live counts: {workers}")
        return 0
    counts = {}
    for worker in items:
        state_name = (worker or {}).get("status", "unknown")
        counts[state_name] = counts.get(state_name, 0) + 1
    summary = ", ".join(f"{name} {count}" for name, count in sorted(counts.items())) or "none"
    print(f"  live workers {len(items)}: {summary}")
    return 0


def delete(args, config, state):
    key = api_key(config)
    endpoint_id, record = find_endpoint(state, args.endpoint, key)
    api_request("DELETE", f"{RUNPOD_API_BASE}/endpoints/{endpoint_id}", key=key)
    with state_transaction(ENDPOINTS_STATE) as registry:
        records = endpoint_records(registry)
        record = records.get(endpoint_id)
        if record is not None:
            record["status"] = "deleted"
            record["last_synced_at"] = utc_now()
            record["drift_note"] = "deleted through rp_endpoints.py"
    print(f"[OK] deleted endpoint {endpoint_id}; registry status=deleted")
    print("     the template is left in place; remove it in the Console if unused")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create/sync RunPod serverless endpoints for the baked ComfyUI images."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    create_parser = sub.add_parser("create", help="create template + endpoint")
    create_parser.add_argument(
        "image_tag", help="pushed image tag, e.g. ns/comfyui-video-wan:v1.0.0"
    )
    create_parser.add_argument("--name", help="endpoint name (default: from the image tag)")
    create_parser.add_argument("--profile", choices=["image-gen", "video"], help="config profile")
    create_parser.add_argument("--gpu", action="append", help="GPU type id; repeat for fallbacks")
    create_parser.add_argument("--workers-min", type=int, default=0)
    create_parser.add_argument("--workers-max", type=int, default=1)
    create_parser.add_argument("--exec-timeout", type=int, help="executionTimeoutMs override")
    create_parser.add_argument("--container-disk-gb", type=int, help="template container disk")
    create_parser.add_argument("--env", action="append", help="template env var KEY=VALUE")

    sub.add_parser("sync", help="diff state/endpoints.json against RunPod; restore idle drops")
    sub.add_parser("list", help="list endpoints recorded in state/endpoints.json")

    health_parser = sub.add_parser("health", help="endpoint config + live worker counts")
    health_parser.add_argument("endpoint", help="endpoint id or name")

    delete_parser = sub.add_parser("delete", help="delete an endpoint and mark it deleted")
    delete_parser.add_argument("endpoint", help="endpoint id or name")

    args = parser.parse_args(argv)

    try:
        config = load_config()
        state = load_state(ENDPOINTS_STATE) or {}
        handlers = {
            "create": create,
            "sync": sync,
            "list": list_endpoints,
            "health": health,
            "delete": delete,
        }
        return handlers[args.command](args, config, state)
    except ConfigError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

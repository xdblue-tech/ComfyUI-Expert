#!/usr/bin/env python3
"""Shared helpers for the RunPod ComfyUI production CLIs.

Used by scripts/rp_images.py, scripts/rp_endpoints.py, and scripts/rp_submit.py.
Standard library only: config/rp.yaml and the images/*/models.yaml manifests are
read with a small YAML-subset parser instead of PyYAML.

Python 3.9+, standard library only.
"""

import contextlib
import fcntl
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "rp.yaml"
IMAGES_DIR = REPO_ROOT / "images"
STATE_DIR = REPO_ROOT / "state"
IMAGES_STATE = STATE_DIR / "images.json"
ENDPOINTS_STATE = STATE_DIR / "endpoints.json"

# REST v1 control plane (templates, endpoints) and the serverless job API.
RUNPOD_API_BASE = "https://rest.runpod.io/v1"
RUNPOD_JOB_BASE = "https://api.runpod.ai/v2"
# Live-verified worker read for endpoint health (skills/runpod/golden-paths/15).
RUNPOD_V2_BASE = "https://v2-rest.runpod.io/v2"

PLACEHOLDER_NAMESPACE = "PLACEHOLDER_DOCKERHUB_NS"

# Image tags are semantic and immutable in practice, so `latest` is never accepted.
VERSION_RE = re.compile(r"v\d+\.\d+\.\d+")

# How long a registry transaction waits for another writer before giving up.
STATE_LOCK_TIMEOUT_S = 30


class ConfigError(Exception):
    """A config, manifest, or argument problem the user must fix."""


def utc_now():
    """ISO-8601 UTC timestamp, matching the state/ files written elsewhere in the repo."""
    return datetime.now(timezone.utc).isoformat()


def _scalar(value):
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_scalar(part) for part in inner.split(",")]
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    if value == "true":
        return True
    if value == "false":
        return False
    if value in ("null", "~"):
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def load_yaml(text):
    """Parse the YAML subset used by config/rp.yaml and images/*/models.yaml.

    Handles nested maps, block lists ("- item"), inline lists ("[a, b]"), plain and
    quoted scalars, and "#" comments. Commas inside inline lists are separators, so
    do not put a comma inside an inline list value.
    """
    logical = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            logical.append((len(line) - len(line.lstrip()), line.strip()))

    root = {}
    stack = [(-1, root)]
    for index, (indent, item) in enumerate(logical):
        while indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]
        if item.startswith("- "):
            if not isinstance(container, list):
                raise ConfigError(f"list item has no list key: {item!r}")
            container.append(_scalar(item[2:]))
            continue
        key, sep, value = item.partition(":")
        if not sep or not key.strip():
            raise ConfigError(f"expected 'key: value', got: {item!r}")
        key = key.strip()
        value = value.strip()
        if value:
            container[key] = _scalar(value)
            continue
        following = logical[index + 1] if index + 1 < len(logical) else None
        if following and following[0] > indent and following[1].startswith("- "):
            container[key] = []
        else:
            container[key] = {}
        stack.append((indent, container[key]))
    return root


def load_yaml_file(path):
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    return load_yaml(text)


def load_config(path=None):
    """Load config/rp.yaml. Raises ConfigError naming the file when it is missing."""
    return load_yaml_file(Path(path) if path else CONFIG_PATH)


def load_state(path):
    """Read a state JSON document; missing file returns an empty document."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc


def save_state(path, data):
    """Write a state JSON document atomically: temp file in place, then os.replace.

    Readers never observe partial JSON, and a crash leaves the previous document
    intact instead of a truncated one. Concurrency between writers is handled by
    state_transaction(), which serializes the read-modify-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["last_updated"] = utc_now()
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    handle, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


def _acquire_lock(stream, lock_path, timeout):
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise ConfigError(
                    f"timed out after {timeout}s waiting for the lock on {lock_path}; "
                    "another rp_* command is still writing that registry"
                )
            time.sleep(0.1)


@contextlib.contextmanager
def state_transaction(path, timeout=STATE_LOCK_TIMEOUT_S):
    """Serialize a registry read-modify-write and commit it atomically.

    Holds <registry>.lock, reloads the current document (so a concurrent writer's
    records are merged rather than overwritten), yields it for mutation, and saves
    it back on a clean exit. An exception inside the block discards the changes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as stream:
        _acquire_lock(stream, lock_path, timeout)
        try:
            data = load_state(path) or {}
            yield data
            save_state(path, data)
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate_version(version, source="--version"):
    """Raise unless version is a semantic vX.Y.Z image tag."""
    if not VERSION_RE.fullmatch(str(version or "")):
        raise ConfigError(
            f"{source} must be a semantic tag like v1.2.3, got {version!r}; "
            "`latest` and untagged images are never used (RunPod caches images per host)"
        )
    return version


def parse_image_tag(tag):
    """Split 'ns/comfyui-video-wan:v1.0.0' into ('ns/comfyui-video-wan', 'v1.0.0')."""
    repo, sep, version = str(tag or "").rpartition(":")
    if not sep or not repo:
        raise ConfigError(
            f"image tag {tag!r} must be <namespace>/<repo>:vX.Y.Z (e.g. ns/comfyui-video-wan:v1.0.0)"
        )
    return repo, validate_version(version, source=f"image tag {tag!r}")


def encode_payload(payload):
    """The exact request body api_request posts for a payload (so callers can size it)."""
    return json.dumps(payload).encode("utf-8")


KEY_HELP = (
    "No RunPod API key. Set RUNPOD_API_KEY in the environment, or write the key "
    "(one line, no quotes) to the api_key_file in config/rp.yaml "
    "(default ~/.runpod/api_key). Create one at "
    "https://www.runpod.io/console/user/settings"
)


def api_key(config=None, required=True):
    """Return the RunPod API key from RUNPOD_API_KEY or config's api_key_file.

    Raises ConfigError with instructions when required and no key is found.
    """
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key and config:
        key_file = Path(os.path.expanduser(str(config.get("api_key_file", ""))))
        if str(key_file) not in ("", ".") and key_file.is_file():
            for line in key_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    key = line
                    break
    if not key and required:
        raise ConfigError(KEY_HELP)
    return key


def api_request(method, url, payload=None, key=None, timeout=120):
    """Call a RunPod JSON endpoint. Returns the decoded body (None when empty)."""
    body = encode_payload(payload) if payload is not None else None
    headers = {
        "Content-Type": "application/json",
        # RunPod's REST front (Cloudflare) rejects default Python UA with error 1010.
        "User-Agent": "comfyui-expert-rp-scripts/1.0",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise ConfigError(f"{method} {url} -> HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ConfigError(f"{method} {url} -> {exc.reason}") from exc
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ConfigError(f"{method} {url} -> non-JSON response: {text[:200]!r}") from exc

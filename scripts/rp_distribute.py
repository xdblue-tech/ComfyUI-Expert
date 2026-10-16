#!/usr/bin/env python3
"""Build the distribute/ package: a standalone, third-party-agent-consumable
export of the RunPod ComfyUI oneshot system.

The package contains the workflow templates, the manifest (compiled from the
workflows/<name>.meta.yaml sidecars), an endpoint index snapshot, the standalone
run tool (bin/rp_oneshot.py, no repo imports), the packaged skill
(skills/runpod-oneshot/), and a README. It carries no secrets: the packaged tool
reads RUNPOD_API_KEY from the environment only. distribute/ is git-tracked.

A template ships only when it has BOTH a meta sidecar and an active endpoint in
state/endpoints.json whose "name" equals the template base name. Templates
missing either are skipped with a warning; zero shippable templates is an error.

Usage:
    python3 scripts/rp_distribute.py build
    python3 scripts/rp_distribute.py check
    python3 scripts/rp_distribute.py list

Python 3.9+, standard library only.
"""

import argparse
import json
import py_compile
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rp_common import (  # noqa: E402  (path set up above)
    ConfigError,
    ENDPOINTS_STATE,
    REPO_ROOT,
    load_state,
    load_yaml_file,
    utc_now,
)

DISTRIBUTE_DIR = REPO_ROOT / "distribute"
WORKFLOWS_DIR = REPO_ROOT / "workflows"
MANIFEST_PATH = DISTRIBUTE_DIR / "manifest.json"
ENDPOINTS_OUT_PATH = DISTRIBUTE_DIR / "endpoints.json"
TOOL_PATH = DISTRIBUTE_DIR / "bin" / "rp_oneshot.py"
SKILL_PATH = DISTRIBUTE_DIR / "skills" / "runpod-oneshot" / "SKILL.md"
README_PATH = DISTRIBUTE_DIR / "README.md"
MANIFEST_VERSION = 1

# Meta keys copied verbatim from the sidecar into the manifest.
META_KEYS = (
    "schema",
    "model_family",
    "model_info",
    "positive",
    "negative",
    "seed",
    "size",
    "default_seed_policy",
)


def shippable_templates():
    """[(name, workflow_path, meta_dict)] for templates that can ship."""
    endpoints = (load_state(ENDPOINTS_STATE) or {}).get("endpoints", {})
    active = {r.get("name"): r for r in endpoints.values() if r.get("status") == "active"}
    shipped = []
    skipped = []
    for workflow_path in sorted(WORKFLOWS_DIR.glob("*.json")):
        name = workflow_path.stem
        meta_path = workflow_path.with_suffix(".meta.yaml")
        if not meta_path.is_file():
            skipped.append(f"{name}: no {meta_path.name} sidecar")
            continue
        if name not in active:
            skipped.append(f"{name}: no active endpoint named {name!r} in state")
            continue
        shipped.append((name, workflow_path, load_yaml_file(meta_path)))
    for reason in skipped:
        print(f"[warn] skip {reason}")
    if not shipped:
        raise ConfigError(
            "no shippable templates: every workflow lacks a .meta.yaml sidecar or "
            "an active endpoint named after it — create the sidecar and/or "
            "`python3 scripts/rp_endpoints.py create` first"
        )
    return shipped


def validate_meta(name, meta, workflow_path):
    """Fail if a sidecar references nodes the template does not have."""
    if not isinstance(meta, dict):
        raise ConfigError(f"workflows/{name}.meta.yaml: not a mapping")
    if meta.get("schema") != MANIFEST_VERSION:
        raise ConfigError(
            f"workflows/{name}.meta.yaml: schema {meta.get('schema')!r} != supported "
            f"{MANIFEST_VERSION}"
        )
    graph = json.loads(workflow_path.read_text(encoding="utf-8"))
    for key in ("positive", "negative", "seed"):
        node_id = (meta.get(key) or {}).get("node")
        if node_id and node_id not in graph:
            raise ConfigError(
                f"workflows/{name}.meta.yaml: {key} node {node_id!r} not in "
                f"{workflow_path.name} — fix the sidecar node ids"
            )
    size_node = (meta.get("size") or {}).get("node")
    if size_node and size_node not in graph:
        raise ConfigError(
            f"workflows/{name}.meta.yaml: size node {size_node!r} not in "
            f"{workflow_path.name} — fix the sidecar node ids"
        )


def build(_args):
    shipped = shippable_templates()
    templates_manifest = {}
    endpoints_out = {}
    record = (load_state(ENDPOINTS_STATE) or {}).get("endpoints", {})
    for name, workflow_path, meta in shipped:
        validate_meta(name, meta, workflow_path)
        out_dir = DISTRIBUTE_DIR / "workflows"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / workflow_path.name).write_bytes(workflow_path.read_bytes())
        templates_manifest[name] = {key: meta.get(key) for key in META_KEYS}
        endpoint = next(
            r for r in record.values()
            if r.get("name") == name and r.get("status") == "active"
        )
        endpoints_out[name] = {
            "endpoint_id": endpoint["endpoint_id"],
            "image_tag": endpoint.get("image_tag"),
            "gpu_types": endpoint.get("gpu_types", []),
            "generated_at": utc_now(),
        }
    manifest = {
        "version": MANIFEST_VERSION,
        "built_at": utc_now(),
        "templates": templates_manifest,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    ENDPOINTS_OUT_PATH.write_text(json.dumps(endpoints_out, indent=2) + "\n", encoding="utf-8")
    packaged = DISTRIBUTE_DIR / "workflows"
    for stale in sorted(packaged.glob("*.json")):
        if stale.stem not in templates_manifest:
            stale.unlink()
    if not shipped:
        raise ConfigError(
            "no shippable templates left: distribute/ indexes regenerated empty and "
            "retired templates pruned — add a .meta.yaml sidecar and/or an active "
            "endpoint to ship again"
        )
    print(f"[OK] distribute/: {len(templates_manifest)} template(s) "
          f"({', '.join(sorted(templates_manifest))})")
    missing_static = [
        str(p) for p in (TOOL_PATH, SKILL_PATH, README_PATH) if not p.is_file()
    ]
    if missing_static:
        print("[warn] distribute/ missing static file(s): " + ", ".join(missing_static))


def check(_args):
    """Non-mutating validation of sources and the built package."""
    shipped = shippable_templates()
    expected_manifest = {}
    expected_endpoints = {}
    record = (load_state(ENDPOINTS_STATE) or {}).get("endpoints", {})
    problems = []
    for name, workflow_path, meta in shipped:
        validate_meta(name, meta, workflow_path)
        expected_manifest[name] = {key: meta.get(key) for key in META_KEYS}
        endpoint = next(
            r for r in record.values()
            if r.get("name") == name and r.get("status") == "active"
        )
        expected_endpoints[name] = {
            "endpoint_id": endpoint["endpoint_id"],
            "image_tag": endpoint.get("image_tag"),
            "gpu_types": endpoint.get("gpu_types", []),
        }
        source = max(workflow_path.stat().st_mtime,
                     workflow_path.with_suffix(".meta.yaml").stat().st_mtime)
        packaged = DISTRIBUTE_DIR / "workflows" / workflow_path.name
        if not packaged.is_file() or packaged.stat().st_mtime < source:
            problems.append(f"stale: distribute/workflows/{workflow_path.name} older "
                            "than its source")
    if not MANIFEST_PATH.is_file():
        raise ConfigError(f"missing {MANIFEST_PATH} — run `rp_distribute.py build`")
    if not ENDPOINTS_OUT_PATH.is_file():
        raise ConfigError(f"missing {ENDPOINTS_OUT_PATH} — run `rp_distribute.py build`")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    packaged_manifest = manifest.get("templates")
    if packaged_manifest != expected_manifest:
        problems.append("manifest.json templates differ from the .meta.yaml sources "
                        "(added/removed/edited templates)")
    packaged_endpoints = {
        name: {k: v for k, v in entry.items() if k != "generated_at"}
        for name, entry in json.loads(
            ENDPOINTS_OUT_PATH.read_text(encoding="utf-8")).items()
    }
    if packaged_endpoints != expected_endpoints:
        problems.append("endpoints.json differs from state/endpoints.json "
                        "(added/removed/changed endpoints)")
    packaged_workflows = {p.stem for p in (DISTRIBUTE_DIR / "workflows").glob("*.json")}
    if packaged_workflows != set(expected_manifest):
        problems.append("distribute/workflows/ contains templates not in the manifest "
                        "(or missing ones)")
    py_compile.compile(str(TOOL_PATH), doraise=True)
    if shipped:
        name = shipped[0][0]
        result = subprocess.run(
            [sys.executable, str(TOOL_PATH), "--template", name, "--prompt", "check",
             "--output", str(DISTRIBUTE_DIR / "output"), "--dry-run"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise ConfigError(
                f"{TOOL_PATH} --dry-run failed (exit {result.returncode}): "
                f"{(result.stderr or result.stdout).strip()[:500]}"
            )
    for problem in problems:
        print(f"[FAIL] {problem}")
    if problems:
        raise ConfigError("package is out of date — run `rp_distribute.py build`")
    print(f"[OK] {len(shipped)} template(s) valid, package matches sources, "
          "tool dry-run passes")


def list_templates(_args):
    record = (load_state(ENDPOINTS_STATE) or {}).get("endpoints", {})
    for name, _workflow_path, meta in shippable_templates():
        endpoint = next(
            r["endpoint_id"] for r in record.values()
            if r.get("name") == name and r.get("status") == "active"
        )
        print(f"{name}  endpoint={endpoint}  family={meta.get('model_family')}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build/check the standalone distribute/ package."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="regenerate distribute/ (templates, manifest, endpoints)")
    sub.add_parser("check", help="validate sources and the built package")
    sub.add_parser("list", help="list shippable templates")
    args = parser.parse_args(argv)
    try:
        {"build": build, "check": check, "list": list_templates}[args.command](args)
    except ConfigError as exc:
        print(f"[FAIL] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
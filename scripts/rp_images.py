#!/usr/bin/env python3
"""Build, check, and push the baked ComfyUI worker images.

Specs live in images/<spec>/ (Dockerfile + models.yaml). The Docker build context
is comfyui_models_path from config/rp.yaml, so Dockerfile COPY lines name model
files as they sit on disk. Versions are semantic tags (<ns>/comfyui-<name>:vX.Y.Z),
read and incremented from state/images.json; `latest` is never used.

Usage:
    python3 scripts/rp_images.py build [SPEC] [--version vX.Y.Z] [--check] [--no-cache]
    python3 scripts/rp_images.py push SPEC [--version vX.Y.Z]
    python3 scripts/rp_images.py list

Without SPEC, `build --check` validates every spec directory.
Python 3.9+, standard library only.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rp_common import (  # noqa: E402  (path set up above)
    ConfigError,
    IMAGES_DIR,
    IMAGES_STATE,
    PLACEHOLDER_NAMESPACE,
    load_config,
    load_state,
    load_yaml_file,
    state_transaction,
    utc_now,
    validate_version,
)

COPY_RE = re.compile(r"^COPY\s+(\S+)\s+(\S+)\s*$", re.IGNORECASE)
COPY_JSON_RE = re.compile(
    r'^COPY\s+\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]\s*$', re.IGNORECASE
)
# Model references in ComfyUI loader node inputs (checkpoint, LoRA, VAE, CLIP, ...).
MODEL_FILE_RE = re.compile(r"\.(safetensors|ckpt|pt|bin|gguf)$", re.IGNORECASE)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Category resolution for loader model references. Keys are (class_type, field) for
# ambiguous fields and bare field names for unambiguous ones; values are the ComfyUI
# model directories that satisfy the reference. A reference whose (class, field) or
# field has no entry is matched by filename only, with a warning, so custom loaders
# never receive a hard failure from a wrong guess.
LOADER_FIELD_CATEGORIES = {
    ("CheckpointLoaderSimple", "ckpt_name"): {"checkpoints"},
    ("LoraLoader", "lora_name"): {"loras"},
    ("UNETLoader", "unet_name"): {"diffusion_models", "unet"},
    ("VAELoader", "vae_name"): {"vae"},
    ("CLIPLoader", "clip_name"): {"clip", "text_encoders"},
    ("DualCLIPLoader", "clip_name"): {"clip", "text_encoders"},
    ("CLIPVisionLoader", "clip_name"): {"clip_vision"},
    ("UpscaleModelLoader", "model_name"): {"upscale_models"},
    ("UpscaleModelLoader", "model_path"): {"upscale_models"},
}
# Unambiguous fields: category holds regardless of loader class (custom forks reuse
# the standard input names). Ambiguous fields (clip_name, model_name, model_path) are
# NOT listed here — they resolve only through LOADER_FIELD_CATEGORIES above.
FIELD_CATEGORIES = {
    "ckpt_name": {"checkpoints"},
    "lora_name": {"loras"},
    "unet_name": {"diffusion_models", "unet"},
    "vae_name": {"vae"},
}


def resolve_categories(class_type, field):
    """Expected model directories for a loader input, or None when unknown."""
    return LOADER_FIELD_CATEGORIES.get((class_type, field)) or FIELD_CATEGORIES.get(field)


def check_workflows(spec, workflows, manifest, manifest_path, notes):
    """Verify every declared workflow's Loader model reference is baked.

    A reference matches a manifest entry only when the filename matches AND the
    entry's category (manifest key) is one the loader field accepts, so a same-named
    file in the wrong model directory is a failure, not a pass.
    """
    baked = {}
    for category, filenames in manifest.items():
        for filename in filenames:
            baked.setdefault(Path(filename).name, set()).add(category)
    warned = set()
    for workflow_rel in workflows:
        workflow_path = Path(workflow_rel) if os.path.isabs(workflow_rel) else REPO_ROOT / workflow_rel
        if not workflow_path.is_file():
            raise ConfigError(f"{spec}: declared workflow not found: {workflow_path}")
        try:
            graph = json.loads(workflow_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ConfigError(f"{spec}: {workflow_path} is not valid JSON: {exc}")
        for node in graph.values() if isinstance(graph, dict) else []:
            if not isinstance(node, dict) or "Loader" not in str(node.get("class_type", "")):
                continue
            class_type = str(node.get("class_type", ""))
            for field, value in node.get("inputs", {}).items():
                if not (isinstance(value, str) and MODEL_FILE_RE.search(value)):
                    continue
                name = Path(value).name
                categories = resolve_categories(class_type, field)
                baked_categories = baked.get(name)
                if baked_categories is None:
                    raise ConfigError(
                        f"{spec}: {workflow_path} loads {value!r} ({class_type}.{field}) "
                        f"but it is not baked in {manifest_path} — add it to the manifest "
                        "and Dockerfile, or the endpoint fails at runtime"
                    )
                if categories is None:
                    warned.add(f"{class_type}.{field}")
                elif not categories & baked_categories:
                    raise ConfigError(
                        f"{spec}: {workflow_path} loads {value!r} ({class_type}.{field}, "
                        f"expects {', '.join(sorted(categories))}) but it is only baked "
                        f"under {', '.join(sorted(baked_categories))} — fix the manifest "
                        "category or the endpoint fails at runtime"
                    )
    if warned:
        notes.append(
            f"[warn] {spec}: {', '.join(sorted(warned))} have no category map; "
            "matched by filename only"
        )
    notes.append(f"{spec}: {len(workflows)} declared workflow(s) covered by the manifest")


def _copy_source(token):
    """Strip surrounding double quotes from a COPY source token (filenames with spaces)."""
    if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    return token


def resolve_spec(name, config):
    """Map 'sdxl' or 'images-sdxl' to an existing directory under images/."""
    for candidate in (name, f"images-{name}"):
        if (IMAGES_DIR / candidate).is_dir():
            return candidate
    raise ConfigError(
        f"unknown image spec {name!r}: no images/{name}/ or images/images-{name}/. "
        f"Available: {', '.join(spec_names())}"
    )


def spec_names():
    if not IMAGES_DIR.is_dir():
        return []
    return sorted(p.name for p in IMAGES_DIR.iterdir() if p.is_dir())


def image_name(spec, config=None):
    """Docker repository name for a spec.

    Config's repo_names overrides the comfyui-<short> default, e.g.
    repo_names: {images-sdxl: unholy-sdxl} in config/rp.yaml.
    """
    override = (config or {}).get("repo_names", {}).get(spec)
    if override:
        return str(override)
    short = spec[len("images-"):] if spec.startswith("images-") else spec
    return f"comfyui-{short}"


def parse_dockerfile(spec):
    """Return (base_image, [(source, destination), ...]) from the spec's Dockerfile."""
    path = IMAGES_DIR / spec / "Dockerfile"
    if not path.is_file():
        raise ConfigError(f"missing {path}")
    base = None
    copies = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.upper().startswith("FROM "):
            base = line.split()[1]
        match = COPY_RE.match(line)
        json_match = COPY_JSON_RE.match(line)
        if json_match:
            copies.append((json_match.group(1), json_match.group(2)))
        elif match:
            copies.append((_copy_source(match.group(1)), match.group(2)))
    if not base:
        raise ConfigError(f"{path} has no FROM line")
    if not copies:
        raise ConfigError(f"{path} has no COPY lines")
    return base, copies


def check_spec(spec, config):
    """Validate one spec: manifest, Dockerfile, local model files, drift. Returns notes."""
    models_path = Path(os.path.expanduser(str(config.get("comfyui_models_path", ""))))
    if not models_path.is_dir():
        raise ConfigError(f"comfyui_models_path is not a directory: {models_path}")
    manifest_path = IMAGES_DIR / spec / "models.yaml"
    if not manifest_path.is_file():
        raise ConfigError(f"missing {manifest_path}")
    manifest = load_yaml_file(manifest_path)
    if not isinstance(manifest, dict) or not manifest:
        raise ConfigError(f"{manifest_path} must map model types to filename lists")

    workflows = manifest.pop("workflows", None)
    if workflows is not None and not (isinstance(workflows, list) and workflows):
        raise ConfigError(f"{manifest_path}: 'workflows' must be a non-empty list of workflow files")

    base, copies = parse_dockerfile(spec)
    expected = []
    missing = []
    for model_type, filenames in manifest.items():
        if not isinstance(filenames, list) or not filenames:
            raise ConfigError(f"{manifest_path}: {model_type!r} must be a non-empty list")
        for filename in filenames:
            source = f"{model_type}/{filename}"
            expected.append(source)
            target = models_path / model_type / filename
            if not target.is_file():
                detail = "broken symlink" if target.is_symlink() else "missing"
                missing.append(f"{target} ({detail})")
    if missing:
        raise ConfigError(
            f"{spec}: {len(missing)} model file(s) unusable from {manifest_path}: "
            + "; ".join(missing)
            + " — restore the file or fix the symlink before building"
        )

    copy_sources = [source for source, _ in copies]
    only_manifest = sorted(set(expected) - set(copy_sources))
    only_dockerfile = sorted(set(copy_sources) - set(expected))
    if only_manifest or only_dockerfile:
        detail = []
        if only_manifest:
            detail.append(f"in models.yaml but not COPYed: {', '.join(only_manifest)}")
        if only_dockerfile:
            detail.append(f"COPYed but not in models.yaml: {', '.join(only_dockerfile)}")
        raise ConfigError(f"{spec}: Dockerfile and models.yaml drift — " + "; ".join(detail))

    config_base = config.get("base_image")
    if base != config_base:
        raise ConfigError(
            f"{spec}: Dockerfile FROM {base} != base_image {config_base} in config/rp.yaml"
        )

    notes = [f"{spec}: {len(expected)} model files present, Dockerfile matches models.yaml"]
    if workflows:
        check_workflows(spec, workflows, manifest, manifest_path, notes)
    if config.get("docker_namespace") == PLACEHOLDER_NAMESPACE:
        notes.append(
            "[warn] docker_namespace is still "
            f"{PLACEHOLDER_NAMESPACE}; fill config/rp.yaml before push"
        )
    return notes


def next_version(spec, state):
    """First build is v1.0.0; later builds bump the minor version."""
    record = state.get("images", {}).get(spec) or {}
    current = str(record.get("version", ""))
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", current)
    if not match:
        return "v1.0.0"
    major, minor, _patch = (int(part) for part in match.groups())
    return f"v{major}.{minor + 1}.0"


def run_docker(args, capture=True):
    """Run a docker subcommand. With capture=False, output is inherited and stdout is None."""
    if not shutil.which("docker"):
        raise ConfigError("docker not found on PATH; install Docker (with buildx) first")
    result = subprocess.run(
        ["docker", *args], capture_output=capture, text=True
    )
    if result.returncode != 0:
        tail = (result.stdout or "")[-2000:] + (result.stderr or "")[-2000:]
        raise ConfigError(f"docker {' '.join(args)} failed:\n{tail.strip()}")
    return (result.stdout or "").strip()


def build(args, config, state):
    spec = resolve_spec(args.spec, config)
    namespace = config.get("docker_namespace")
    if namespace == PLACEHOLDER_NAMESPACE:
        raise ConfigError(
            "docker_namespace is still "
            f"{PLACEHOLDER_NAMESPACE} in config/rp.yaml; set it and run `docker login`"
        )
    version = args.version or next_version(spec, state)
    validate_version(version)

    tag = f"{namespace}/{image_name(spec, config)}:{version}"
    command = [
        "build",
        "--platform",
        "linux/amd64",
        "-f",
        str(IMAGES_DIR / spec / "Dockerfile"),
        "-t",
        tag,
    ]
    if args.no_cache:
        command.append("--no-cache")
    command.append(os.path.expanduser(config["comfyui_models_path"]))
    print(f"[info] docker {' '.join(command)}")
    run_docker(command, capture=False)
    print(f"[OK] built {tag}")

    with state_transaction(IMAGES_STATE) as registry:
        registry.setdefault("images", {})[spec] = {
            "spec": spec,
            "repo": f"{namespace}/{image_name(spec, config)}",
            "version": version,
            "tag": tag,
            "status": "built",
            "built_at": utc_now(),
        }
    print(f"[OK] recorded {version} for {spec} in {IMAGES_STATE}")
    return 0


def push(args, config, state):
    spec = resolve_spec(args.spec, config)
    record = (state.get("images") or {}).get(spec)
    if not record:
        raise ConfigError(f"no build recorded for {spec}; run: rp_images.py build {spec}")
    version = validate_version(
        args.version or record.get("version"),
        source="--version (or the recorded build version)",
    )
    tag = f"{record['repo']}:{version}"

    run_docker(["push", tag], capture=False)
    digest_line = run_docker(["image", "inspect", tag, "--format", "{{index .RepoDigests 0}}"])
    size_line = run_docker(["image", "inspect", tag, "--format", "{{.Size}}"])
    created_line = run_docker(["image", "inspect", tag, "--format", "{{.Created}}"])
    digest = digest_line.split("@", 1)[1] if "@" in digest_line else digest_line

    # Merge into whichever record is current, so a concurrent build for another spec
    # is not overwritten by this one's snapshot.
    with state_transaction(IMAGES_STATE) as registry:
        record = registry.setdefault("images", {}).setdefault(spec, dict(record))
        record.update(
            {
                "version": version,
                "tag": tag,
                "digest": digest,
                "size_mb": round(int(size_line) / 1048576),
                "created_at": created_line,
                "pushed_at": utc_now(),
                "status": "pushed",
            }
        )
    print(f"[OK] pushed {tag}")
    print(f"     digest {digest}  size {record['size_mb']} MB")
    print(f"[OK] recorded in {IMAGES_STATE}")
    return 0


def list_images(args, config, state):
    images = (state or {}).get("images") or {}
    if not images:
        print(f"no images recorded in {IMAGES_STATE}; run: rp_images.py build <spec>")
        return 0
    for spec in sorted(images):
        record = images[spec]
        pushed = record.get("pushed_at", record.get("built_at", "-"))
        print(
            f"{spec:14} {record.get('tag', '-'):48} {record.get('status', '-'):7} "
            f"{record.get('size_mb', '-')} MB  {pushed}"
        )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build/push baked ComfyUI worker images.")
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="build a spec image (semantic tag)")
    build_parser.add_argument("spec", nargs="?", help="spec directory name, e.g. sdxl")
    build_parser.add_argument("--version", help="explicit tag version, e.g. v1.2.0")
    build_parser.add_argument(
        "--check",
        action="store_true",
        help="validate manifests/Dockerfile/model files only; no docker invocation",
    )
    build_parser.add_argument("--no-cache", action="store_true", help="pass --no-cache")

    push_parser = sub.add_parser("push", help="push a built image and record its digest")
    push_parser.add_argument("spec", help="spec directory name, e.g. sdxl")
    push_parser.add_argument("--version", help="explicit tag version, default: last built")

    sub.add_parser("list", help="list images recorded in state/images.json")
    args = parser.parse_args(argv)

    try:
        config = load_config()
        state = load_state(IMAGES_STATE) or {}
        if args.command == "build" and args.check:
            specs = [resolve_spec(args.spec, config)] if args.spec else spec_names()
            if not specs:
                raise ConfigError(f"no spec directories under {IMAGES_DIR}")
            failures = 0
            for spec in specs:
                try:
                    for note in check_spec(spec, config):
                        print(f"[OK] {note}" if not note.startswith("[warn]") else note)
                except ConfigError as exc:
                    print(f"[FAIL] {exc}", file=sys.stderr)
                    failures += 1
            return 1 if failures else 0
        if args.command == "build":
            if not args.spec:
                raise ConfigError("build needs a spec (or --check to validate all specs)")
            return build(args, config, state)
        if args.command == "push":
            return push(args, config, state)
        return list_images(args, config, state)
    except ConfigError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Scan a ComfyUI installation for models, custom nodes, and system info.

Modes:
  online  - query the running ComfyUI REST API (preferred)
  offline - walk the ComfyUI directory on disk
  auto    - try online, fall back to offline (default)

Output: state/inventory.json (schema documented in docs/architecture.md).
Python 3.9+, standard library only.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "state" / "inventory.json"
DEFAULT_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")

MODEL_TYPES = [
    "checkpoints", "loras", "vae", "controlnet", "clip", "clip_vision",
    "upscale_models", "diffusion_models", "ipadapter", "instantid",
    "insightface", "facerestore_models", "embeddings",
]
MODEL_EXTENSIONS = {
    "checkpoints": (".safetensors", ".ckpt"),
    "loras": (".safetensors",),
    "vae": (".safetensors", ".pt"),
    "controlnet": (".safetensors", ".pth"),
    "clip": (".safetensors",),
    "clip_vision": (".safetensors",),
    "upscale_models": (".pth", ".safetensors"),
    "diffusion_models": (".safetensors", ".gguf"),
    "ipadapter": (".safetensors", ".bin"),
    "instantid": (".bin",),
    "insightface": (".onnx",),
    "facerestore_models": (".pth",),
    "embeddings": (".pt", ".safetensors"),
}
CANDIDATE_PATHS = [
    "~/ComfyUI", "~/comfyui", "/opt/ComfyUI", "/workspace/ComfyUI",
    "C:/ComfyUI", "C:/ComfyUI-Easy-Install/ComfyUI",
    "E:/ComfyUI-Easy-Install/ComfyUI", "E:/ComfyUI",
]


def detect_comfyui_path(candidates=None):
    """Return the first existing ComfyUI directory, or None."""
    env = os.environ.get("COMFYUI_PATH")
    paths = ([Path(env)] if env else []) + list(candidates or [Path(p).expanduser() for p in CANDIDATE_PATHS])
    for path in paths:
        if (path / "models").is_dir() or (path / "main.py").is_file():
            return path
    return None


def http_json(url, timeout=10):
    """GET a URL and return parsed JSON. Raises OSError/ValueError on failure."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def scan_online(url, fetch=http_json):
    """Query the running ComfyUI API. `fetch` is injectable for tests."""
    stats = fetch(url.rstrip("/") + "/system_stats")
    device = (stats.get("devices") or [{}])[0]
    system = {
        "gpu": device.get("name", "unknown"),
        "vram_total_gb": round(device.get("vram_total", 0) / 1e9, 1),
        "vram_free_gb": round(device.get("vram_free", 0) / 1e9, 1),
    }
    models = {}
    for model_type in MODEL_TYPES:
        try:
            models[model_type] = sorted(fetch(url.rstrip("/") + "/models/" + model_type))
        except (OSError, ValueError):
            models[model_type] = []
    node_classes = sorted(fetch(url.rstrip("/") + "/object_info"))
    return {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "mode": "online",
        "comfyui_version": stats.get("system", {}).get("comfyui_version", "unknown"),
        "comfyui_path": "",
        "comfyui_url": url,
        "system": system,
        "models": models,
        "custom_nodes": [],
        "node_classes": node_classes,
    }


def scan_offline(root):
    """Walk a ComfyUI directory. `root` is a str or Path."""
    root = Path(root)
    models = {}
    models_dir = root / "models"
    for model_type in MODEL_TYPES:
        exts = MODEL_EXTENSIONS[model_type]
        directory = models_dir / model_type
        files = []
        if directory.is_dir():
            for entry in directory.rglob("*"):
                if entry.is_file() and entry.suffix.lower() in exts:
                    files.append(entry.name)
        models[model_type] = sorted(files)

    detection = models_dir / "ultralytics" / "bbox"
    models["detection"] = sorted(p.name for p in detection.glob("*.pt")) if detection.is_dir() else []

    motion = root / "custom_nodes" / "ComfyUI-AnimateDiff-Evolved" / "models"
    models["animatediff_motion"] = sorted(
        p.name for p in motion.glob("*") if p.suffix.lower() in (".ckpt", ".safetensors")
    ) if motion.is_dir() else []

    custom_nodes = []
    nodes_dir = root / "custom_nodes"
    if nodes_dir.is_dir():
        custom_nodes = sorted(
            p.name for p in nodes_dir.iterdir()
            if p.is_dir() and p.name != "__pycache__" and not p.name.startswith(".")
        )

    return {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "mode": "offline",
        "comfyui_version": "unknown",
        "comfyui_path": str(root),
        "comfyui_url": "",
        "system": {"gpu": "unknown (offline scan)", "vram_total_gb": 0, "vram_free_gb": 0},
        "models": models,
        "custom_nodes": custom_nodes,
        "node_classes": [],
    }


def write_inventory(data, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Scan a ComfyUI installation.")
    parser.add_argument("--mode", choices=["auto", "online", "offline"], default="auto")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--comfyui-path", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    data = None
    if args.mode in ("auto", "online"):
        try:
            data = scan_online(args.url)
            print("[OK] scanned online at " + args.url)
        except (OSError, ValueError) as exc:
            if args.mode == "online":
                print(f"[FAIL] cannot reach ComfyUI at {args.url}: {exc}")
                return 1
            print(f"[info] ComfyUI not reachable ({exc}); falling back to offline scan")

    if data is None:
        root = Path(args.comfyui_path) if args.comfyui_path else detect_comfyui_path()
        if root is None or not Path(root).is_dir():
            print("[FAIL] no ComfyUI directory found. Pass --comfyui-path or set COMFYUI_PATH.")
            return 1
        data = scan_offline(root)
        print("[OK] scanned offline at " + str(root))

    output = write_inventory(data, args.output)
    total = sum(len(v) for v in data["models"].values())
    print(
        f"mode={data['mode']} models={total} custom_nodes={len(data['custom_nodes'])} "
        f"node_classes={len(data['node_classes'])}"
    )
    print("inventory written to " + str(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())

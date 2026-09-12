# ComfyUI Expert — Agent-Agnostic Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repo fully usable by any AgentSkills-compatible agent (pi first) when opened at the repo root, and remove all harness-specific launch machinery.

**Architecture:** One canonical `AGENTS.md` orchestrator. Skills stay in `skills/` and are exposed to modern harnesses via a committed `.agents/skills` symlink. Five PowerShell scripts and the Windows `.bat` launcher are replaced by two stdlib-only Python scripts (`scan_inventory.py`, `session.py`). Harness-specific adapters (`openclaw/`, global Claude sync scripts) are deleted; per-harness instructions move to `docs/compatibility.md`.

**Tech Stack:** Markdown (Agent Skills standard, agentskills.io), Python 3.9+ (stdlib only, no pip deps), git.

**Spec:** This plan's “Requirements & Decisions” section below is the spec (no separate spec doc exists). Source: user request 2026-09-12 — “plan those fixes, goal is to make it fully useful for you from projects root and remove any unnecessary component, make it a fully agent agnostic product.”

## Requirements & Decisions (spec)

Verified facts driving this plan:

- ComfyUI **is running** at `http://127.0.0.1:8188` (v0.33.1, linux, Python 3.14.7), source tree at `/home/xdblue/ComfyUI`.
- Real GPU is **RTX 5070 Ti, 16.7 GB VRAM** — docs claiming RTX 5090 / 32 GB are wrong.
- `pwsh` is **not installed** → all `.ps1` scripts are dead components here.
- Installed models: 8 SDXL-family checkpoints, 24 LoRAs (API view); `diffusion_models`/`unet`/`text_encoders` empty → video skills are dormant until models exist.
- `state/session.json` and `state/inventory.json` do not exist; nothing writes them anymore.
- `pi` auto-discovers skills from `.agents/skills/` in cwd. **Verified**: a symlinked `.agents/skills` → repo `skills/` exposed all 15 comfyui-* skills to a headless pi session.
- pi loads `AGENTS.md` **or** `CLAUDE.md` from the project root; other harnesses (Codex, Cursor, OpenClaw) read `AGENTS.md`.

Decisions:

- **Keep** `skills/`, `foundation/`, `references/`, `projects/`, `state/` layout (no path churn).
- **Add** `.agents/skills` symlink (committed) — primary discovery mechanism, zero config.
- **Replace** 5 PowerShell scripts + `video-agent.bat` with 2 Python scripts (stdlib only).
- **Delete** `openclaw/` adapter dir, `agent/AGENT.md` (duplicate), global-sync scripts (`deploy.ps1`, `sync-skills.ps1`).
- **Keep** `CLAUDE.md` as a 3-line pointer to `AGENTS.md` (Claude Code compat).
- **Out of scope (deferred):** merging `comfyui-video-pipeline` vs `comfyui-video-production` (needs content review); installing video models.

## Global Constraints

- Python 3.9+ standard library only. No pip dependencies anywhere.
- `AGENTS.md` is canonical. No harness-specific launch machinery may remain.
- No skill directory moves. `.agents/skills` is a symlink to `../skills`.
- Keep all existing `references/` content.
- Conventional commit style, matching repo history (`feat:`, `fix:`, `docs:`).
- Any script the plan adds must run on Linux and Windows (use `pathlib`, no shelling out).
- Any file the plan deletes must have zero remaining references in the repo when its task completes (grep check is part of each task).

---

### Task 1: Enable pi skill discovery

**Files:**

- Create: `.agents/skills` (symlink → `../skills`)
- Modify: `.gitignore`
- Modify (local only, NOT committed): `.pi/settings.json`

**Interfaces:**

- Consumes: existing `skills/` directory.
- Produces: `.agents/skills` path that Task 7 and all future sessions rely on.

- [ ] **Step 1: Create the symlink**

```bash
cd /home/xdblue/Repos/ComfyUI-Expert
mkdir -p .agents
ln -s ../skills .agents/skills
ls -la .agents/skills
```

Expected: `skills -> ../skills`.

- [ ] **Step 2: Add personal pi config to .gitignore**

Append to `.gitignore`:

```gitignore
# pi personal project config (machine-specific)
.pi/settings.json
__pycache__/
```

- [ ] **Step 3: Add skills entry to the local pi settings (belt and braces)**

The file already has a `skills` array. Append `"../skills"` to it — do not add a second `skills` key:

```json
"skills": ["../skills", "<keep existing entries>"]
```

Do not commit this file.

- [ ] **Step 4: Verify pi discovery from the repo root**

Run:

```bash
cd /home/xdblue/Repos/ComfyUI-Expert
timeout 120 pi -p --approve --no-extensions "List the names of skills available to you, comma-separated, nothing else."
```

Expected: output contains all of `comfyui-api, comfyui-character-gen, comfyui-inventory, comfyui-lora-training, comfyui-prompt-engineer, comfyui-prompt-interview, comfyui-research, comfyui-troubleshooter, comfyui-video-pipeline, comfyui-video-production, comfyui-voice-pipeline, comfyui-workflow-builder, project-manager, video-assembly, video-publisher`.

If the symlink is not followed: fallback = revert the symlink and commit `.pi/settings.json` containing only `{"skills": ["../skills"]}` (never the machine's other entries).

- [ ] **Step 5: Commit**

```bash
git add .agents/skills .gitignore
git commit -m "feat: expose skills to pi and other harnesses via .agents/skills"
```

Note for Windows users (also added to `docs/compatibility.md` in Task 7): symlinks require Developer Mode or `git config core.symlinks true`; otherwise copy `skills/` to `.agents/skills/`.

---

### Task 2: Replace scan-inventory.ps1 with a cross-platform Python scanner

**Files:**

- Create: `scripts/scan_inventory.py`
- Create: `tests/test_scan_inventory.py`
- Delete (in Task 8): `scripts/scan-inventory.ps1`

**Interfaces:**

- Consumes: nothing (standalone).
- Produces: CLI `python3 scripts/scan_inventory.py [--mode auto|online|offline] [--url URL] [--comfyui-path PATH] [--output PATH]`, exit code 0 on success / 1 on failure. Writes `state/inventory.json` with this exact schema (Task 4's skill text and `docs/architecture.md` reference it):

```json
{
  "last_updated": "2026-09-12T18:00:00+00:00",
  "mode": "online",
  "comfyui_version": "0.33.1",
  "comfyui_path": "/home/xdblue/ComfyUI",
  "comfyui_url": "http://127.0.0.1:8188",
  "system": {"gpu": "cuda:0 NVIDIA GeForce RTX 5070 Ti", "vram_total_gb": 16.7, "vram_free_gb": 15.5},
  "models": {"checkpoints": ["a.safetensors"], "loras": []},
  "custom_nodes": ["SomeNode"],
  "node_classes": ["KSampler"]
}
```

`node_classes` is populated only in online mode; `custom_nodes` only in offline mode; both default to `[]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scan_inventory.py`:

```python
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import scan_inventory as si


class OfflineScanTests(unittest.TestCase):
    def test_scan_offline_finds_models_and_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models" / "checkpoints").mkdir(parents=True)
            (root / "models" / "checkpoints" / "a.safetensors").write_bytes(b"x")
            (root / "models" / "loras").mkdir(parents=True)
            (root / "models" / "loras" / "b.safetensors").write_bytes(b"x")
            (root / "models" / "loras" / "ignore.txt").write_text("x")
            (root / "custom_nodes" / "SomeNode").mkdir(parents=True)
            (root / "custom_nodes" / "__pycache__").mkdir()
            (root / "custom_nodes" / "__pycache__" / "junk.pyc").write_bytes(b"x")

            data = si.scan_offline(root)

            self.assertEqual(data["mode"], "offline")
            self.assertEqual(data["models"]["checkpoints"], ["a.safetensors"])
            self.assertEqual(data["models"]["loras"], ["b.safetensors"])
            self.assertEqual(data["custom_nodes"], ["SomeNode"])
            self.assertEqual(data["node_classes"], [])
            self.assertEqual(data["comfyui_path"], str(root))


class OnlineScanTests(unittest.TestCase):
    def test_scan_online_shape(self):
        def fake_fetch(url, timeout=10):
            if url.endswith("/system_stats"):
                return {
                    "system": {"comfyui_version": "9.9.9"},
                    "devices": [{"name": "cuda:0 Fake", "vram_total": 1000000000, "vram_free": 500000000}],
                }
            if "/models/" in url:
                return ["m.safetensors"] if url.endswith("/checkpoints") else []
            if url.endswith("/object_info"):
                return {"KSampler": {}, "CLIPTextEncode": {}}
            raise AssertionError("unexpected url " + url)

        data = si.scan_online("http://example:8188", fetch=fake_fetch)

        self.assertEqual(data["mode"], "online")
        self.assertEqual(data["comfyui_version"], "9.9.9")
        self.assertEqual(data["system"]["gpu"], "cuda:0 Fake")
        self.assertEqual(data["system"]["vram_total_gb"], 1.0)
        self.assertEqual(data["models"]["checkpoints"], ["m.safetensors"])
        self.assertEqual(data["node_classes"], ["CLIPTextEncode", "KSampler"])
        self.assertEqual(data["custom_nodes"], [])


class PathDetectionTests(unittest.TestCase):
    def test_detect_uses_env_var(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "models").mkdir()
            with mock.patch.dict(os.environ, {"COMFYUI_PATH": tmp}):
                self.assertEqual(si.detect_comfyui_path(), Path(tmp))

    def test_detect_skips_non_comfyui_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(si.detect_comfyui_path(candidates=[Path(tmp)]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scan_inventory'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/scan_inventory.py`:

```python
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
                print("[FAIL] cannot reach ComfyUI at %s: %s" % (args.url, exc))
                return 1
            print("[info] ComfyUI not reachable (%s); falling back to offline scan" % exc)

    if data is None:
        root = Path(args.comfyui_path) if args.comfyui_path else detect_comfyui_path()
        if root is None or not Path(root).is_dir():
            print("[FAIL] no ComfyUI directory found. Pass --comfyui-path or set COMFYUI_PATH.")
            return 1
        data = scan_offline(root)
        print("[OK] scanned offline at " + str(root))

    output = write_inventory(data, args.output)
    total = sum(len(v) for v in data["models"].values())
    print("mode=%s models=%d custom_nodes=%d node_classes=%d" % (
        data["mode"], total, len(data["custom_nodes"]), len(data["node_classes"])))
    print("inventory written to " + str(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -v`
Expected: 4 tests, OK.

- [ ] **Step 5: Verify against the real machine**

Run: `python3 scripts/scan_inventory.py`
Expected: `mode=online`, writes `state/inventory.json`, and `state/inventory.json` contains `"gpu": "cuda:0 NVIDIA GeForce RTX 5070 Ti ..."`.

- [ ] **Step 6: Commit**

```bash
git add scripts/scan_inventory.py tests/test_scan_inventory.py
git commit -m "feat: cross-platform inventory scanner (replaces scan-inventory.ps1)"
```

---

### Task 3: Bootstrap script (session state + staleness + connectivity)

**Files:**

- Create: `scripts/session.py`
- Create: `tests/test_session.py`
- Delete (in Task 8): `scripts/connect-comfyui.ps1`, `scripts/staleness-check.ps1`

**Interfaces:**

- Consumes: `references/staleness-report.md` (parses `**Date**: YYYY-MM-DD` and the literal `Not yet run`); `state/inventory.json` if present.
- Produces: CLI `python3 scripts/session.py [--project NAME] [--comfyui-url URL]`, writes `state/session.json` with keys `{comfyui_url, active_project, started, host, cwd}`, and prints a human/agent-readable summary. Exit code always 0 (bootstrap must not fail a session).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_session.py`:

```python
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import session as sess


class StalenessTests(unittest.TestCase):
    def test_fresh_report(self):
        text = "# Staleness Report\n**Date**: 2026-09-01\n"
        result = sess.parse_staleness(text, today=date(2026, 9, 12))
        self.assertEqual(result["last_run"], date(2026, 9, 1))
        self.assertEqual(result["days"], 11)
        self.assertFalse(result["stale"])

    def test_old_report(self):
        text = "**Date**: 2026-08-01\n"
        result = sess.parse_staleness(text, today=date(2026, 9, 12))
        self.assertTrue(result["stale"])

    def test_never_run(self):
        result = sess.parse_staleness("Not yet run\n", today=date(2026, 9, 12))
        self.assertIsNone(result["last_run"])
        self.assertIn("never", result["note"].lower())

    def test_missing_date(self):
        result = sess.parse_staleness("no date here", today=date(2026, 9, 12))
        self.assertIsNone(result["last_run"])


class SessionFileTests(unittest.TestCase):
    def test_write_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            sess.write_session("my-video", "http://127.0.0.1:8188", path=path)
            data = json.loads(path.read_text())
            self.assertEqual(data["active_project"], "my-video")
            self.assertEqual(data["comfyui_url"], "http://127.0.0.1:8188")
            self.assertIn("started", data)


class ConnectivityTests(unittest.TestCase):
    def test_unreachable_returns_none(self):
        self.assertIsNone(sess.check_comfyui("http://127.0.0.1:9"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'session'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/session.py`:

```python
#!/usr/bin/env python3
"""Bootstrap a ComfyUI Expert session.

Writes state/session.json and prints a summary: ComfyUI reachability,
GPU/VRAM, model counts, inventory freshness, research staleness.

Usage:
    python3 scripts/session.py [--project NAME] [--comfyui-url URL]

Python 3.9+, standard library only.
"""
import argparse
import json
import os
import re
import socket
import sys
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_PATH = REPO_ROOT / "state" / "session.json"
INVENTORY_PATH = REPO_ROOT / "state" / "inventory.json"
STALENESS_REPORT = REPO_ROOT / "references" / "staleness-report.md"
DEFAULT_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
STALE_AFTER_DAYS = 14


def parse_staleness(text, today=None):
    """Parse references/staleness-report.md. Returns dict with last_run/days/stale/note."""
    today = today or date.today()
    if "Not yet run" in text:
        return {"last_run": None, "days": None, "stale": True,
                "note": "Research has never been run. Consider: 'research latest ComfyUI updates'"}
    match = re.search(r"\*\*Date\*\*:\s*(\d{4}-\d{2}-\d{2})", text)
    if not match:
        return {"last_run": None, "days": None, "stale": False,
                "note": "No research date found in staleness report."}
    last_run = date.fromisoformat(match.group(1))
    days = (today - last_run).days
    stale = days > STALE_AFTER_DAYS
    note = "Research data is %d days old." % days
    if stale:
        note += " Consider: 'research latest ComfyUI updates'"
    return {"last_run": last_run, "days": days, "stale": stale, "note": note}


def check_comfyui(url, timeout=3):
    """GET /system_stats. Returns dict or None when unreachable."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/system_stats", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError):
        return None


def write_session(project, url, path=SESSION_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "comfyui_url": url,
        "active_project": project or "",
        "started": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cwd": str(Path.cwd()),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def print_summary(data, stats, staleness):
    print("ComfyUI Expert session")
    print("  session file : %s" % SESSION_PATH)
    print("  comfyui url  : %s" % data["comfyui_url"])
    print("  project      : %s" % (data["active_project"] or "(none)"))
    if stats is None:
        print("  comfyui      : unreachable (start ComfyUI or pass --comfyui-url)")
    else:
        device = (stats.get("devices") or [{}])[0]
        print("  comfyui      : OK v%s" % stats.get("system", {}).get("comfyui_version", "?"))
        print("  gpu          : %s (%.1f GB free of %.1f GB)" % (
            device.get("name", "unknown"),
            device.get("vram_free", 0) / 1e9,
            device.get("vram_total", 0) / 1e9))
    if INVENTORY_PATH.is_file():
        inv = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        total = sum(len(v) for v in inv.get("models", {}).values())
        print("  inventory    : %s, %s models (mode=%s)" % (
            inv.get("last_updated", "?"), total, inv.get("mode", "?")))
    else:
        print("  inventory    : missing. Run: python3 scripts/scan_inventory.py")
    print("  research     : %s" % staleness["note"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bootstrap a ComfyUI Expert session.")
    parser.add_argument("--project", default=None)
    parser.add_argument("--comfyui-url", default=DEFAULT_URL)
    args = parser.parse_args(argv)

    if STALENESS_REPORT.is_file():
        staleness = parse_staleness(STALENESS_REPORT.read_text(encoding="utf-8"))
    else:
        staleness = {"last_run": None, "days": None, "stale": True,
                     "note": "references/staleness-report.md is missing."}

    data = write_session(args.project, args.comfyui_url)
    print_summary(data, check_comfyui(args.comfyui_url), staleness)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -v`
Expected: 5 session tests + 4 scanner tests, OK.

- [ ] **Step 5: Verify against the real machine**

Run: `python3 scripts/session.py --project smoke-test`
Expected output includes `comfyui : OK v0.33.1`, the RTX 5070 Ti line, `inventory : ...` (written by Task 2), and a research staleness note. `state/session.json` exists.

- [ ] **Step 6: Commit**

```bash
git add scripts/session.py tests/test_session.py
git commit -m "feat: cross-platform session bootstrap (replaces connect-comfyui and staleness hooks)"
```

---

### Task 4: Remove harness-specific references from skills and foundation

**Files:**

- Modify: `skills/comfyui-inventory/SKILL.md`
- Modify: `skills/comfyui-api/SKILL.md:58`
- Modify: `foundation/skill-registry.md:24`
- Modify: `skills/comfyui-video-production/references/instance-management.md` (only if it presents Windows as the sole path)

**Interfaces:**

- Consumes: Task 2 CLI (`python3 scripts/scan_inventory.py`) and Task 3 CLI (`python3 scripts/session.py`).
- Produces: skills that reference only current components; no `pwsh`, `.ps1`, `video-agent.bat`, or `~/.claude` dependency.

- [ ] **Step 1: Rewrite the inventory skill body**

In `skills/comfyui-inventory/SKILL.md`:

- Frontmatter: change `"requires":{"bins":["pwsh"]}` to `"requires":{"anyBins":["python3","python"]}` and `"primaryEnv":"COMFYUI_PATH"`.
- Replace every `scan-inventory.ps1` / `pwsh` instruction with:

```bash
python3 scripts/scan_inventory.py                  # auto: online first, offline fallback
python3 scripts/scan_inventory.py --mode offline --comfyui-path "$COMFYUI_PATH"
python3 scripts/scan_inventory.py --url http://127.0.0.1:8188
```

- Document the output schema pointer: “Schema: see docs/architecture.md (Inventory)”.

- [ ] **Step 2: Fix remaining references**

- `skills/comfyui-api/SKILL.md:58`: change `"client_id": "video-agent"` → `"client_id": "comfyui-expert"`.
- `foundation/skill-registry.md:24`: change the `comfyui-character-gen` row to `| skills/comfyui-character-gen/ | ... |` (it now lives in this repo).
- Add the missing `comfyui-video-production` row to the registry. Every skill directory must appear: the number of registry rows must equal `ls skills | wc -l` (15).
- `skills/comfyui-video-production/references/instance-management.md`: if the multi-instance recipe only shows PowerShell, add a POSIX equivalent block (three lines: `cd /path/ComfyUI && python main.py --port 8188` and the second instance on `--port 8189`). Do not delete existing content.

- [ ] **Step 3: Verify no stale references remain**

Run:

```bash
grep -rn 'pwsh\|\.ps1\|video-agent\|~/.claude' skills/ foundation/
```

Expected: only `skills/comfyui-video-production/references/instance-management.md` may retain a Windows sample, and it must also contain the POSIX block. No other hits. Also run `ls skills | wc -l` and confirm it equals the number of rows in `foundation/skill-registry.md` (15).

- [ ] **Step 4: Commit**

```bash
git add skills/ foundation/
git commit -m "fix: drop PowerShell/Claude-only references from skills and registry"
```

---

### Task 5: AGENTS.md is the canonical orchestrator

**Files:**

- Create: `AGENTS.md` (merge of `CLAUDE.md`, `agent/AGENT.md`, `openclaw/AGENTS.md`)
- Modify: `CLAUDE.md` → 3-line pointer
- Delete (in Task 8): `agent/AGENT.md`

**Interfaces:**

- Consumes: `scripts/session.py` and `scripts/scan_inventory.py` command lines from Tasks 2–3.
- Produces: the single file every harness loads. Later docs quote its Session Start block verbatim.

- [ ] **Step 1: Write AGENTS.md**

Structure (keep under ~150 lines, in this order):

- Title + one-paragraph role (VideoAgent / ComfyUI production orchestrator).
- **Hardware**: `RTX 5070 Ti, 16.7 GB VRAM, Linux, ComfyUI at http://127.0.0.1:8188` — details in `foundation/hardware-profile.md`.
- **Session start**: run `python3 scripts/session.py --project <name-if-known>`, then read in order: `foundation/skill-registry.md`, `foundation/model-landscape.md`, `foundation/hardware-profile.md`, plus `projects/<name>/manifest.yaml` when a project is active.
- **Critical rule**: before generating any workflow, ensure `state/inventory.json` exists and is fresh; if missing run `python3 scripts/scan_inventory.py`; never use a model/node absent from the inventory.
- **Routing table** (copy from current `CLAUDE.md` — all 12 rows).
- **Authority matrix** (copy from `CLAUDE.md`).
- **Multi-step pipeline pattern** (copy 5 bullets from `CLAUDE.md`).
- **Error recovery** → `skills/comfyui-troubleshooter/SKILL.md`.
- **Context tiers** table (foundation / projects / references).
- **Files** map listing the final layout (see Task 7 target tree; no `agent/`, no `openclaw/`, no `.bat`).

Sources to merge: `CLAUDE.md` (routing, authority, tiers), `agent/AGENT.md` (persona, decision tree, staleness, integration points, error recovery), `openclaw/AGENTS.md` (inventory rule, compact routing table).

- [ ] **Step 2: Replace CLAUDE.md with a pointer**

Full new content of `CLAUDE.md`:

```markdown
# ComfyUI Expert

This project's agent instructions live in [AGENTS.md](AGENTS.md). Read that file.
```

- [ ] **Step 3: Verify**

```bash
grep -rn 'agent/AGENT.md\|video-agent.bat\|\.claude/settings' AGENTS.md CLAUDE.md
```

Expected: no output. Then run a fresh headless check:

```bash
timeout 120 pi -p --approve --no-extensions "In one sentence: what is this project and what do you do at session start?"
```

Expected: mentions ComfyUI workflows and running `scripts/session.py` / reading foundation files.

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md CLAUDE.md
git commit -m "feat: canonical AGENTS.md orchestrator, CLAUDE.md becomes a pointer"
```

---

### Task 6: Correct the hardware profile

**Files:**

- Rewrite: `foundation/hardware-profile.md`
- Modify: `foundation/model-landscape.md` (only stale hardware claims)
- Modify: `README.md` hardware section (trimmed again in Task 7)

**Interfaces:**

- Consumes: live values from `curl http://127.0.0.1:8188/system_stats`.
- Produces: capability guidance that matches 16.7 GB VRAM.

- [ ] **Step 1: Rewrite hardware-profile.md**

Replace the GPU section with the verified values:

```markdown
## GPU
- **Model**: NVIDIA GeForce RTX 5070 Ti
- **VRAM**: 16.7 GB (reported by CUDA; plan for ~16 GB usable)
- **ComfyUI**: 0.33.1 at http://127.0.0.1:8188, source at /home/xdblue/ComfyUI
- **OS**: Linux, Python 3.14
```

Replace the “Capabilities at 32 GB” section with a 16 GB table. Required rows (values are the guidance the agent must follow):

| Task class | 16.7 GB verdict |
| --- | --- |
| SDXL / Pony / Illustrious checkpoints | Full quality, no offload |
| Flux dev (fp8 or GGUF Q8) | Viable, expect offload with large batches |
| SD1.5 / SDXL ControlNet stacks | Viable |
| LoRA training (SDXL) | Viable with small batch + gradient checkpointing |
| 14B-class video models (Wan etc.) | Not installed; only GGUF quantized + offload, slow |
| Simultaneous video + upscaler pipelines | Avoid; run sequentially |

Replace launch flags section: default flags only; `--highvram` is not appropriate for 16 GB. Recommended: `python main.py --listen` (no highvram), optionally `--fp8_e4m3fn-unet` for Flux.

- [ ] **Step 2: Sweep stale claims**

```bash
grep -rn '5090\|32GB\|32 GB' foundation/ README.md AGENTS.md
```

Fix every hit (the number is wrong for this machine). Keep an optional sentence in `hardware-profile.md`: “If you move this repo to a 32 GB card, restore a 32 GB capability table.”

- [ ] **Step 3: Verify**

```bash
python3 scripts/session.py | grep -i '5070' 
grep -rni '5090\|32gb' foundation/ | wc -l
```

Expected: first command prints the 5070 Ti line; second prints `0`.

- [ ] **Step 4: Commit**

```bash
git add foundation/ README.md
git commit -m "fix: hardware profile matches the actual RTX 5070 Ti / 16.7 GB"
```

---

### Task 7: Rewrite README and docs for any harness

**Files:**

- Rewrite: `README.md`
- Rewrite: `docs/getting-started.md`
- Rewrite: `docs/architecture.md`
- Create: `docs/compatibility.md`

**Interfaces:**

- Consumes: everything above; final CLI list is `python3 scripts/scan_inventory.py` and `python3 scripts/session.py`.
- Produces: docs with zero references to deleted files. Task 8's grep gate depends on this.

- [ ] **Step 1: Write docs/compatibility.md**

Content: one table plus per-harness notes.

| Harness | Instructions file | Skills discovery | Notes |
| --- | --- | --- | --- |
| pi | `AGENTS.md` (auto) | `.agents/skills` symlink (auto) | works with zero config from repo root |
| Claude Code | `CLAUDE.md` → pointer to `AGENTS.md` | create `.claude/skills` symlink → `../skills` (optional) | |
| OpenAI Codex CLI | `AGENTS.md` (auto) | agent reads `skills/*/SKILL.md` on demand | |
| Cursor | `AGENTS.md` (auto) | agent reads on demand | |
| OpenClaw | point workspace instructions at `AGENTS.md` | point workspace skills dir at `skills/` | no adapter shipped; skills are already AgentSkills |
| Any other agent | paste `AGENTS.md` as context | point it at `skills/` | |

Include the Windows symlink note from Task 1.

- [ ] **Step 2: Rewrite docs/getting-started.md**

Sections: Prerequisites (any AgentSkills-compatible agent; Python 3.9+; ComfyUI running; ffmpeg optional), Quick start:

```bash
cd ComfyUI-Expert
python3 scripts/session.py                 # bootstrap + status
python3 scripts/scan_inventory.py          # first run / after installing models
# now open your agent in this directory and ask for what you want
```

plus example prompts, the file locations table (updated: no launcher, scripts = 2 Python files, symlink), troubleshooting table (ComfyUI unreachable → `--comfyui-url`; skills not visible → check `ls .agents/skills`; stale research → `references/staleness-report.md`).

- [ ] **Step 3: Rewrite docs/architecture.md**

Keep: 3-tier context system, skill dependency graph, skill invocation flow. Replace: “How It Loads” diagram (agent → AGENTS.md → `.agents/skills` + foundation reads + `session.py`); “Global vs Local” (skills are repo-local; harnesses pointed at them); add **Inventory** section documenting the JSON schema from Task 2 and `session.py` outputs. Delete `deploy.ps1` section and all `.bat`/hook text.

- [ ] **Step 4: Rewrite README.md**

Sections, in order: What this is / Why (rewrite of current “Why This Exists”), How it works (5 bullets), Requirements, Quick start (same 3 commands as getting-started), The 15 skills (keep current per-skill table, update inventory row text), Scripts (`scan_inventory.py`, `session.py` — one line each), Repo layout (final tree, see Task 8), Compatibility (link `docs/compatibility.md`), License. Delete: the entire OpenClaw section, PowerShell prerequisites, `video-agent.bat` quick start, hardware section (now the 5070 Ti version, one short block).

- [ ] **Step 5: Verify**

```bash
grep -rn --exclude-dir=plans 'video-agent.bat\|pwsh\|PowerShell 7\|scan-inventory.ps1\|connect-comfyui\|deploy.ps1\|sync-skills\|staleness-check' README.md docs/
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/
git commit -m "docs: rewrite README and docs for agent-agnostic setup"
```

---

### Task 8: Delete dead components and run the final sweep

**Files:**

- Delete: `video-agent.bat`
- Delete: `scripts/scan-inventory.ps1`, `scripts/connect-comfyui.ps1`, `scripts/staleness-check.ps1`, `scripts/deploy.ps1`, `scripts/sync-skills.ps1`
- Delete: `openclaw/` (directory), `agent/` (directory)
- Modify: `.gitignore` (already updated in Task 1)

**Interfaces:**

- Consumes: completed Tasks 4, 5, 7 (all references fixed).
- Produces: the final tree:

```text
ComfyUI-Expert/
|-- .agents/skills -> ../skills
|-- AGENTS.md
|-- CLAUDE.md            (pointer)
|-- README.md
|-- docs/                (architecture, getting-started, compatibility, plans/)
|-- foundation/
|-- projects/
|-- references/
|-- scripts/             (scan_inventory.py, session.py)
|-- skills/              (15 skills)
|-- state/               (runtime, gitignored contents)
|-- tests/               (test_scan_inventory.py, test_session.py)
```

- [ ] **Step 1: Confirm zero references first**

```bash
grep -rn 'video-agent\.bat\|scan-inventory\.ps1\|connect-comfyui\|staleness-check\|deploy\.ps1\|sync-skills\|openclaw\|agent/AGENT' --include='*.md' --include='*.py' . | grep -v '^./.git/' | grep -v 'docs/plans/'
```

Expected: no output. If hits exist, fix them before deleting.

- [ ] **Step 2: Delete**

```bash
git rm video-agent.bat
git rm scripts/*.ps1
git rm -r openclaw agent
```

- [ ] **Step 3: Final verification**

```bash
python3 -m unittest discover -s tests -v
python3 scripts/scan_inventory.py && python3 scripts/session.py
git status --short
```

Expected: tests pass; both scripts succeed; `git status` shows only the intended deletions.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove Windows/Claude-specific launch machinery"
```

---

## Self-Review

- **Spec coverage:** “useful from repo root” → Tasks 1, 5 (pi loads AGENTS.md + all 15 skills, verified headless). “Remove unnecessary components” → Tasks 4, 8 (5 PS1 + bat + openclaw + duplicate agent dir). “Agent-agnostic” → Tasks 5, 7 (AGENTS.md canonical, compatibility matrix, no harness machinery). “Plan those fixes” (prior turn’s 4 fixes) → hardware fix (Task 6), skill registration (Task 1), Python inventory (Task 2), Linux launcher replacement (Task 3). No coverage gaps found.
- **Placeholder scan:** none — all code and test bodies are complete; doc tasks specify section lists and exact commands.
- **Type consistency:** `scan_inventory.py` schema keys (`models`, `custom_nodes`, `node_classes`, `system.vram_free_gb`) match `session.py` reads and the architecture doc text; CLI flags (`--project`, `--comfyui-url`, `--comfyui-path`, `--output`) are used consistently in Tasks 4, 5, 7.

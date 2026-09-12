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
import urllib.parse
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
    """Parse references/staleness-report.md. Returns last_run/days/stale/note."""
    today = today or date.today()
    if "Not yet run" in text:
        return {
            "last_run": None,
            "days": None,
            "stale": True,
            "note": (
                "Research has never been run. Consider: "
                "'research latest ComfyUI updates'"
            ),
        }
    match = re.search(r"\*\*Date\*\*:\s*(\d{4}-\d{2}-\d{2})", text)
    if not match:
        return {
            "last_run": None,
            "days": None,
            "stale": False,
            "note": "No research date found in staleness report.",
        }
    last_run = date.fromisoformat(match.group(1))
    days = (today - last_run).days
    stale = days > STALE_AFTER_DAYS
    note = f"Research data is {days} days old."
    if stale:
        note += " Consider: 'research latest ComfyUI updates'"
    return {"last_run": last_run, "days": days, "stale": stale, "note": note}


def check_comfyui(url, timeout=3):
    """GET /system_stats over HTTP(S). Return parsed data or None when unreachable."""
    endpoint = url.rstrip("/") + "/system_stats"
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    try:
        with urllib.request.urlopen(endpoint, timeout=timeout) as resp:  # noqa: S310, RUF100
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError):
        return None


def write_session(project, url, path=SESSION_PATH):
    """Write session metadata and return it."""
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
    """Print a human- and agent-readable session summary."""
    print("ComfyUI Expert session")
    print(f"  session file : {SESSION_PATH}")
    print(f"  comfyui url  : {data['comfyui_url']}")
    print(f"  project      : {data['active_project'] or '(none)'}")
    if stats is None:
        print("  comfyui      : unreachable (start ComfyUI or pass --comfyui-url)")
    else:
        device = (stats.get("devices") or [{}])[0]
        print(
            f"  comfyui      : OK v{stats.get('system', {}).get('comfyui_version', '?')}"
        )
        print(
            f"  gpu          : {device.get('name', 'unknown')} "
            f"({device.get('vram_free', 0) / 1e9:.1f} GB free of "
            f"{device.get('vram_total', 0) / 1e9:.1f} GB)"
        )
    if INVENTORY_PATH.is_file():
        try:
            inv = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
            total = sum(len(v) for v in inv.get("models", {}).values())
        except (OSError, ValueError, AttributeError, TypeError):
            print(
                "  inventory    : unreadable — re-run "
                "python3 scripts/scan_inventory.py"
            )
        else:
            print(
                f"  inventory    : {inv.get('last_updated', '?')}, {total} models "
                f"(mode={inv.get('mode', '?')})"
            )
    else:
        print("  inventory    : missing. Run: python3 scripts/scan_inventory.py")
    print(f"  research     : {staleness['note']}")


def main(argv=None):
    """Run the non-failing session bootstrap."""
    parser = argparse.ArgumentParser(description="Bootstrap a ComfyUI Expert session.")
    parser.add_argument("--project", default=None)
    parser.add_argument("--comfyui-url", default=DEFAULT_URL)
    args = parser.parse_args(argv)

    if STALENESS_REPORT.is_file():
        try:
            staleness = parse_staleness(
                STALENESS_REPORT.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            staleness = {
                "last_run": None,
                "days": None,
                "stale": False,
                "note": "references/staleness-report.md is unreadable.",
            }
    else:
        staleness = {
            "last_run": None,
            "days": None,
            "stale": True,
            "note": "references/staleness-report.md is missing.",
        }

    data = write_session(args.project, args.comfyui_url)
    print_summary(data, check_comfyui(args.comfyui_url), staleness)
    return 0


if __name__ == "__main__":
    sys.exit(main())

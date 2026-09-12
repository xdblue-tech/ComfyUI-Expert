# ComfyUI Expert Architecture

## Overview

ComfyUI Expert is a repository-local production system for agents that work with ComfyUI. `AGENTS.md` supplies the canonical operating instructions, and 15 AgentSkills provide task-specific procedures for character images, video, voice, LoRA training, assembly, publishing, and troubleshooting.

## Core design decisions

1. **Repository-local instructions and skills** — project behavior lives in versioned files rather than global installation state.
2. **Agent-agnostic discovery** — harnesses that auto-load `AGENTS.md` use it directly; others can be pointed at the same file and `skills/` directory.
3. **Read skills on demand** — the agent reads `skills/{name}/SKILL.md` only after request routing selects it.
4. **Inventory before workflows** — generated workflows must use models and nodes present in `state/inventory.json`.
5. **Incremental context** — small foundation files are read first; project and reference material are loaded only when needed.

## How it loads

```text
Agent opened at repository root
  |
  +-- reads AGENTS.md
  |     |
  |     +-- discovers .agents/skills -> ../skills when supported
  |     +-- reads foundation guidance at session start
  |     +-- routes a request to skills/{name}/SKILL.md
  |
  +-- python3 scripts/session.py
        |
        +-- writes state/session.json
        +-- reports ComfyUI, inventory, and research status
```

See [compatibility.md](compatibility.md) for harness-specific discovery notes.

## 3-tier context system

### Tier 1: Foundation

Small files read at session start for current operating guidance.

| File | Purpose |
| --- | --- |
| `foundation/hardware-profile.md` | GPU, VRAM, and launch guidance |
| `foundation/model-landscape.md` | Current model choices by category |
| `foundation/skill-registry.md` | Skill list and dependency map |
| `foundation/api-quick-ref.md` | ComfyUI REST API reference |

### Tier 2: Working

Files read while working on an active project.

| File | Purpose |
| --- | --- |
| `projects/{name}/manifest.yaml` | Project settings, defaults, and status |
| `projects/{name}/characters/{char}/profile.yaml` | Character appearance, voice, LoRA, and history |
| `projects/{name}/notes.md` | Successful settings and production notes |

### Tier 3: Reference

Large files read only when the selected skill needs detailed material.

| File | Purpose |
| --- | --- |
| `references/models.md` | Model specifications and download links |
| `references/workflows.md` | Workflow node configurations |
| `references/lora-training.md` | Training parameters and practices |
| `references/voice-synthesis.md` | Voice tools in depth |
| `references/prompt-templates.md` | Model-specific prompt strategies |
| `references/troubleshooting.md` | Error database and recovery guidance |
| `references/research-log.md` | Technique survey and research history |

## Skill dependency graph

```text
AGENTS.md (canonical instructions)
    |
    +-- Discovery
    |   +-- comfyui-prompt-interview
    |
    +-- Foundation skills (no dependencies)
    |   +-- comfyui-api
    |   +-- comfyui-inventory
    |   +-- project-manager
    |
    +-- Research (independent)
    |   +-- comfyui-research
    |
    +-- Core creation (depends on inventory)
    |   +-- comfyui-prompt-engineer
    |   +-- comfyui-workflow-builder
    |   +-- comfyui-character-gen
    |
    +-- Production (depends on inventory and creation)
    |   +-- comfyui-video-pipeline
    |   +-- comfyui-video-production
    |   +-- comfyui-voice-pipeline
    |   +-- comfyui-lora-training
    |
    +-- Output (depends on production)
    |   +-- video-assembly
    |   +-- video-publisher
    |
    +-- Support
        +-- comfyui-troubleshooter
```

The authoritative current skill list is `foundation/skill-registry.md`.

## Skill invocation flow

```text
User: "Generate a character portrait"
  |
  +-- AGENTS.md routing table selects comfyui-workflow-builder
  |
  +-- agent reads skills/comfyui-workflow-builder/SKILL.md
  |
  +-- skill requires state/inventory.json
  |     +-- missing or stale: run python3 scripts/scan_inventory.py
  |
  +-- agent validates models and node classes against inventory
  |
  +-- skill requests reference material only when needed
  |
  +-- agent produces or queues the validated workflow through comfyui-api
```

## Inventory

`python3 scripts/scan_inventory.py` writes `state/inventory.json`. The default `auto` mode queries a running ComfyUI API first and falls back to a local ComfyUI installation when one can be detected. Use `--mode online`, `--mode offline`, `--url URL`, `--comfyui-path PATH`, and `--output PATH` when needed.

The inventory schema is:

```json
{
  "last_updated": "2026-09-12T18:00:00+00:00",
  "mode": "online",
  "comfyui_version": "0.33.1",
  "comfyui_path": "/home/user/ComfyUI",
  "comfyui_url": "http://127.0.0.1:8188",
  "system": {
    "gpu": "cuda:0 NVIDIA GeForce RTX 5070 Ti",
    "vram_total_gb": 16.7,
    "vram_free_gb": 15.5
  },
  "models": {
    "checkpoints": ["a.safetensors"],
    "loras": []
  },
  "custom_nodes": ["SomeNode"],
  "node_classes": ["KSampler"]
}
```

`node_classes` is populated only by an online scan, and `custom_nodes` is populated only by an offline scan; both default to empty lists. The scanner supplies the complete supported model-category map under `models`, so agents should use the actual cached keys rather than assume only the example categories.

`python3 scripts/session.py` writes `state/session.json` and prints a summary containing the selected ComfyUI URL, active project, ComfyUI reachability and GPU details when available, inventory timestamp/model count/mode, and research-staleness guidance. It accepts `--project NAME` and `--comfyui-url URL`.

## Global vs local

| Scope | Contents | Setup |
| --- | --- | --- |
| Repository local | `AGENTS.md`, `skills/`, `foundation/`, `references/`, project state, and scripts | Open an agent at the repository root. |
| Harness configuration | Optional discovery pointers to the repository-local files | Point the harness at `AGENTS.md` and `skills/`; do not copy behavior into global agent state unless that harness requires it. |

The skills remain repository-local. pi discovers them through `.agents/skills -> ../skills`; compatible harnesses may use their own optional symlink or on-demand file reads while consuming the same source files.

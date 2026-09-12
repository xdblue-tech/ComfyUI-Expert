# ComfyUI Expert

## What this is

ComfyUI Expert is a repository-local AI production assistant for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Its instructions and 15 [AgentSkills](https://agentskills.io)-compatible skills help an agent plan and validate character images, video, voice, LoRA training, assembly, and publishing work.

## Why

ComfyUI video production combines models, custom nodes, prompts, hardware limits, and project state. ComfyUI Expert keeps that operating knowledge in versioned files, so you can ask for a production outcome instead of memorizing node names or VRAM budgets. The agent checks the installed inventory before proposing a workflow and reads detailed reference material only when the selected skill needs it.

## How it works

- `AGENTS.md` is the canonical repository-root instruction file; `CLAUDE.md` points compatible Claude Code sessions to it.
- The `.agents/skills` symlink exposes the repository's `skills/` directory to pi, while other harnesses can read the same skill files on demand.
- `python3 scripts/session.py` writes session state and prints ComfyUI, inventory, and research-freshness status.
- `python3 scripts/scan_inventory.py` records installed models, custom nodes, and online node classes in `state/inventory.json`.
- The agent reads the matching `SKILL.md`, validates required models and nodes against the inventory, then follows that skill's production workflow.

## Requirements

- An AgentSkills-compatible agent that can read repository files.
- Python 3.9 or newer for the included scripts.
- A running local or remote ComfyUI instance for online scanning and workflow execution.
- [FFmpeg](https://ffmpeg.org/) is optional, but needed for local video assembly tasks.

> **Current hardware profile:** The checked-in guidance targets Linux with an NVIDIA GeForce RTX 5070 Ti reporting 16.7 GB VRAM. Read `foundation/hardware-profile.md` for current capability guidance.

## Quick start

```bash
cd ComfyUI-Expert
python3 scripts/session.py                 # bootstrap + status
python3 scripts/scan_inventory.py          # first run / after installing models
# now open your agent in this directory and ask for what you want
```

Example prompts:

```text
Create a project called "Character Showcase" and add a character named Sage.
Generate a photorealistic portrait of Sage with identity preservation.
Create a short talking-head video from this character image and script.
Scan my ComfyUI installation, then build a workflow using only installed models.
Research current ComfyUI video models and update the reference notes.
```

## The 15 skills

| Skill | Use it for |
| --- | --- |
| `comfyui-api` | Connect to ComfyUI, queue workflows, monitor execution, and retrieve results. |
| `comfyui-character-gen` | Build identity-preserving character-generation workflows and pipelines. |
| `comfyui-inventory` | Scan installed models, offline custom nodes, and online node classes with `scripts/scan_inventory.py`; cache the result in `state/inventory.json`. |
| `comfyui-lora-training` | Prepare datasets and configure character-consistency LoRA training. |
| `comfyui-prompt-engineer` | Craft model-specific prompts for the selected checkpoint and identity method. |
| `comfyui-prompt-interview` | Turn a creative idea into a complete image-generation brief through guided questions. |
| `comfyui-research` | Research ComfyUI models and techniques, update references, and flag stale information. |
| `comfyui-troubleshooter` | Diagnose ComfyUI server, workflow, quality, and performance failures. |
| `comfyui-video-pipeline` | Generate image-to-video, text-to-video, talking-head, and motion-controlled video workflows. |
| `comfyui-video-production` | Plan and orchestrate validated multi-shot ComfyUI video-production pipelines. |
| `comfyui-voice-pipeline` | Generate speech, clone voices, and plan lip-sync workflows. |
| `comfyui-workflow-builder` | Build validated ComfyUI workflow JSON from natural-language requirements. |
| `project-manager` | Manage project manifests, character profiles, workflow history, and assets. |
| `video-assembly` | Assemble generated clips, audio, titles, transitions, and exports with FFmpeg or Remotion. |
| `video-publisher` | Prepare completed videos for YouTube and other publishing targets. |

## Scripts

- `scripts/scan_inventory.py` scans a running or on-disk ComfyUI installation and writes the inventory cache.
- `scripts/session.py` writes session metadata and reports ComfyUI connectivity, inventory freshness, and research status.

## Repo layout

```text
ComfyUI-Expert/
|-- .agents/skills -> ../skills  Skill discovery symlink
|-- AGENTS.md                    Canonical agent instructions
|-- CLAUDE.md                    Pointer to AGENTS.md
|-- LICENSE                      Repository license (MIT)
|-- README.md
|-- docs/
|   |-- architecture.md
|   |-- compatibility.md
|   |-- getting-started.md
|   +-- plans/
|-- foundation/                  Tier 1 guidance
|-- projects/                    Per-project manifests and assets
|-- references/                  Deep reference material
|-- scripts/
|   |-- scan_inventory.py
|   +-- session.py
|-- skills/                      15 AgentSkills
|-- state/                       Runtime state; contents are gitignored
+-- tests/
    |-- test_scan_inventory.py
    +-- test_session.py
```

## Compatibility

See [docs/compatibility.md](docs/compatibility.md) for pi, Claude Code, OpenAI Codex CLI, Cursor, OpenClaw, and other-agent setup notes.

## License

MIT

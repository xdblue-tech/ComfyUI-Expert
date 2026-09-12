# Getting Started

## Prerequisites

- Any [AgentSkills](https://agentskills.io)-compatible agent that can read this repository.
- Python 3.9 or newer.
- ComfyUI running locally or at a reachable remote URL.
- [FFmpeg](https://ffmpeg.org/) is optional and needed only for local video assembly.

## Quick start

```bash
cd ComfyUI-Expert
python3 scripts/session.py                 # bootstrap + status
python3 scripts/scan_inventory.py          # first run / after installing models
# now open your agent in this directory and ask for what you want
```

`session.py` accepts `--project NAME` to set an active project and `--comfyui-url URL` to check a non-default ComfyUI endpoint. `scan_inventory.py` tries the configured ComfyUI URL first, then can scan an on-disk installation when given `--comfyui-path PATH`.

## Example prompts

```text
Create a project called "Character Showcase" and add a character named Sage.
Generate a photorealistic portrait of Sage using an installed identity method.
Create a talking-head video from this character image and this script.
Train a LoRA from these reference images.
Research the latest ComfyUI video models.
```

## File locations

| What | Where |
| --- | --- |
| Canonical instructions | `AGENTS.md` |
| Claude Code compatibility pointer | `CLAUDE.md` |
| Skill discovery symlink | `.agents/skills` → `../skills` |
| Skills | `skills/*/SKILL.md` |
| Foundation context | `foundation/` |
| Deep references | `references/` |
| Project state | `projects/` |
| Inventory cache | `state/inventory.json` |
| Session metadata | `state/session.json` |
| Python scripts | `scripts/scan_inventory.py`, `scripts/session.py` |

## Troubleshooting

| Issue | Action |
| --- | --- |
| ComfyUI is unreachable | Start ComfyUI, or run `python3 scripts/session.py --comfyui-url http://host:8188`. |
| Inventory scan cannot find ComfyUI | Pass `--comfyui-path PATH`, set `COMFYUI_PATH`, or start the server for an online scan. |
| Skills are not visible to pi | From the repository root, run `ls .agents/skills` and confirm it resolves to `../skills`. |
| Research guidance is stale | Read `references/staleness-report.md`, then ask the agent to research current ComfyUI updates. |
| A workflow names a missing model or node | Run `python3 scripts/scan_inventory.py` after installation, then ask the agent to rebuild the workflow. |

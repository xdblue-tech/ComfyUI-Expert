# Harness Compatibility

| Harness | Instructions file | Skills discovery | Notes |
| --- | --- | --- | --- |
| pi | `AGENTS.md` (auto) | `.agents/skills` symlink (auto) | Works with zero configuration from the repository root. |
| Claude Code | `CLAUDE.md` → pointer to `AGENTS.md` | Create `.claude/skills` symlink → `../skills` (optional) | |
| OpenAI Codex CLI | `AGENTS.md` (auto) | Agent reads `skills/*/SKILL.md` on demand | |
| Cursor | `AGENTS.md` (auto) | Agent reads on demand | |
| OpenClaw | Point workspace instructions at `AGENTS.md` | Point workspace skills directory at `skills/` | No adapter is shipped; the skills are already AgentSkills. |
| Any other agent | Paste `AGENTS.md` as context | Point it at `skills/` | |

## pi

Open pi from the repository root. It automatically reads `AGENTS.md` and discovers the `.agents/skills` symlink.

## Claude Code

`CLAUDE.md` is a short compatibility pointer to `AGENTS.md`. Claude Code can read skills directly from `skills/`; creating `.claude/skills` as a symlink to `../skills` is optional when that local discovery path is useful.

## OpenAI Codex CLI

Open Codex CLI in the repository root. It automatically reads `AGENTS.md`; the agent reads the matching `skills/*/SKILL.md` file on demand.

## Cursor

Open the repository root in Cursor. It automatically reads `AGENTS.md`; direct the agent to the matching skill file when a task requires it.

## OpenClaw

Configure the workspace to use `AGENTS.md` for instructions and `skills/` for skills. No OpenClaw-specific adapter is needed because the skills already follow the AgentSkills format.

## Any other agent

Provide `AGENTS.md` as context and point the agent at `skills/`. The repository-local scripts, foundation files, projects, and references remain available without a harness-specific launcher.

## Windows symlink note

Git symlinks on Windows require Developer Mode or `git config core.symlinks true`. If symlinks are unavailable, copy `skills/` to `.agents/skills/` instead.

# Installing skills — the CLI, the bundles, and why nothing installs transitively

This file covers **getting skills onto a machine**. It explains the `skills` CLI as this suite uses it
and lists the per-playbook install bundles. It also covers troubleshooting, including the one failure
mode that is structural rather than a bug.

## Contents

1. [The structural fact: no dependencies](#1-the-structural-fact-no-dependencies)
2. [The commands you actually need](#2-the-commands-you-actually-need)
3. [Bundles by playbook](#3-bundles-by-playbook)
4. [Scope, agents, and how files land](#4-scope-agents-and-how-files-land)
5. [Reading a skill without installing it](#5-reading-a-skill-without-installing-it)
6. [Keeping skills current](#6-keeping-skills-current)
7. [The Claude Code plugin route](#7-the-claude-code-plugin-route)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. The structural fact: no dependencies

**A `SKILL.md` cannot declare that it needs another skill, and the CLI resolves nothing
transitively.** The required frontmatter fields are `name` and `description`. The only meaningful
optional field is `metadata.internal`. There is no dependency field, no lockfile of dependencies,
and no install-time resolution.

This structural fact shapes how the suite is written, in two ways:

- **Cross-skill links resolve only when both skills are installed.** Every link in this repo is
  relative, in the form `../sibling/`, and installing flattens the domain folder away. So a link to
  `../z-image/` resolves when `z-image` is installed, and dangles otherwise. **A dangling link is
  not a bug. It means you have not yet run the install command for that sibling.**
- **[`generative-media-atlas`](../) carries what it would otherwise route to a sibling**, because it
  is the one skill designed to be installed alone. That is a deliberate deviation from how the rest
  of the suite is written, and it is why this skill runs longer than its cross-cutting siblings.

---

## 2. The commands you actually need

```bash
# Browse the catalogue without installing
npx skills add ryannel/skills --list

# One skill
npx skills add ryannel/skills --skill generative-media-atlas

# Several, in one command (repeat the flag)
npx skills add ryannel/skills --skill z-image --skill character-lora-training

# Interactive pick from the full list
npx skills add ryannel/skills

# Non-interactive, e.g. in CI
npx skills add ryannel/skills --skill z-image -g -a claude-code -y
```

**Ask the user before installing.** This matters especially with `-g`, or when installing into a
repository they did not ask you to modify. Installing writes files into their agent directories, so
it changes their machine. It is not just a lookup.

The CLI resolves sources in several ways: GitHub shorthand such as `ryannel/skills`, a full
GitHub/GitLab/git URL, a direct tree URL to one skill, a local path, or a `.zip`/`.tar` download.
Private repositories use whatever Git authentication is already configured. The CLI tries the Git
credential helper first, then `gh repo clone`, then SSH.

---

## 3. Bundles by playbook

These bundles come from [`playbooks.md`](playbooks.md). Add `-g` to install globally, or
`-a claude-code` to target one agent.

| Playbook | Command |
|---|---|
| **A** — realistic invented character, on RunPod | `npx skills add ryannel/skills --skill generative-media-atlas --skill z-image --skill qwen-image --skill character-lora-training --skill comfyui-on-runpod --skill image-production-workflows` then `npx skills add runpod/runpod-plugins-official` |
| **B** — anime character, local | `npx skills add ryannel/skills --skill anima --skill character-lora-training --skill image-production-workflows` |
| **C** — design image with text | `npx skills add ryannel/skills --skill ideogram-4 --skill image-production-workflows` |
| **D** — still into a shot | `npx skills add ryannel/skills --skill wan-2-2 --skill image-production-workflows` (swap `wan-2-2` for `ltx-2-5` or `minimax-h3` per the licence fork) |
| **E** — replace a person in footage | `npx skills add ryannel/skills --skill krea-2 --skill scail-2 --skill character-lora-training` (add `--skill wan-2-2` for close-up faces and lip sync via Animate 2) |
| **F** — run it as an API | `npx skills add ryannel/skills --skill comfyui-on-runpod --skill image-production-workflows` then `npx skills add runpod/runpod-plugins-official` |
| **Instruction editing, or identity without training** (no lettered playbook yet) | `npx skills add ryannel/skills --skill qwen-image --skill image-production-workflows` |
| **Everything** | `npx skills add ryannel/skills --skill '*'` |

**There is no bundle for the models the suite does not cover.** Bernini-R, SenseNova
U1.5 and HunyuanVideo-1.5 have no skill here as of 2026-09-09, so nothing to install; the atlas's
suite map says where to route instead. **Black Forest Labs' hosted-API skills** live in
`black-forest-labs/skills` and should install with the same CLI (`npx skills add
black-forest-labs/skills`); that command was not run in this pass, and the flag lives in
[`ecosystem-map.md`](ecosystem-map.md) §4.

---

## 4. Scope, agents, and how files land

| Flag | Effect |
|---|---|
| *(default)* | Project scope — `./<agent>/skills/`, committed with the repo, shared with the team |
| `-g, --global` | User scope — `~/<agent>/skills/`, available in every project |
| `-a, --agent <agents…>` | Target named agents (`claude-code`, `codex`, `cursor`, …); `'*'` for all |
| `-s, --skill <skills…>` | Specific skills by name; `'*'` for all. Quote names containing spaces |
| `--copy` | Independent copies per agent instead of symlinks to one canonical copy |
| `-y, --yes` | Skip confirmation prompts |
| `--all` | Every skill to every agent, no prompts |

**Symlink is the default, and it is the right choice for this suite.** Each skill keeps one
canonical copy, so `npx skills update` updates every agent at once. Use `--copy` only where symlinks
are not supported.

**Install by skill name, not by path.** Names stay stable when the repository is reorganised. This
repo groups skills under `skills/generative-media/`, but that grouping is invisible in the installed
result.

---

## 5. Reading a skill without installing it

```bash
npx skills use ryannel/skills@z-image | claude
npx skills use ryannel/skills --skill z-image --agent claude-code
```

`skills use` resolves the source the same way `add` does. It writes the skill to a temporary
directory and prints the generated prompt to stdout. When `--agent` is given, it starts an agent
with that prompt instead. This is useful for a one-off consultation, and for checking whether a
skill is what you want before it lands in a repository.

---

## 6. Keeping skills current

```bash
npx skills update                       # all, with a scope prompt
npx skills update z-image wan-2-2       # named
npx skills update -g                    # global only
npx skills list                         # what is installed, where
```

**Run the update command against this suite deliberately.** The models it covers change weekly, and
every skill here carries a `Facts dated …` line in its two-bar section. Compare that date against
your install date instead of assuming the copy on disk is current.

---

## 7. The Claude Code plugin route

This is an alternative to per-skill installation: it installs the whole domain as one plugin.

```bash
/plugin marketplace add ryannel/skills
/plugin install generative-media-skills@ryannel-skills
```

The CLI also reads `.claude-plugin/marketplace.json` when it discovers skills in a repository.
Skills declared there are found at their declared depth. Without that declaration, the CLI finds
skills by a bounded depth-3 walk.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| A link inside a skill 404s | The sibling is not installed; nothing resolves transitively | Install it: `npx skills add ryannel/skills --skill <name>` |
| `--list` shows more skills than the catalogue | The CLI scans `.agents/skills/` and `.claude/skills/` as standard locations, so a repo's own authoring machinery appears | Repo-side fix: `metadata.internal: true` in that skill's frontmatter hides it unless `INSTALL_INTERNAL_SKILLS=1` |
| An expected skill is missing from `--list` | It is marked internal | `INSTALL_INTERNAL_SKILLS=1 npx skills add <repo> --list` |
| Private repo download fails | Git auth not configured for that remote | Configure the credential helper, authenticate `gh`, or use the SSH URL form. `GITHUB_TOKEN`/`GH_TOKEN` work for API access |
| Skill installs but the agent never uses it | The `description` is the trigger the agent matches on | Check that the description names the user's vocabulary: the model names, node names and error strings they would type |
| Updating one agent leaves another stale | Installed with `--copy` | Reinstall with symlinks, or update each agent |
| A skill's claims are out of date | Skills are snapshots | `npx skills update`, then check the skill's `Facts dated` line |

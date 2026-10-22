"""Load and validate one AgentSkill ``SKILL.md`` from the shipped package.

The package ships a real skill file with YAML frontmatter. Agents need its
``name`` and ``description`` as data and its body as instructions, but writing
a full YAML parser here would add a dependency the package does not have. This
module parses the small, documented subset that AgentSkill files use and fails
with an actionable message on anything it cannot read, instead of guessing.
"""

import re
from dataclasses import dataclass
from pathlib import Path

SKILL_FILENAME = "SKILL.md"
MAX_DESCRIPTION_CHARS = 1024
NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
FRONTMATTER = re.compile(r"\A---\r?\n(?P<header>.*?)\r?\n---\r?\n(?P<body>.*)\Z", re.DOTALL)
TOP_LEVEL_KEY = re.compile(r"^(?P<key>[A-Za-z0-9_-]+):(?:[ \t]*(?P<value>.*))?$")


class SkillError(Exception):
    """The skill file is missing or does not match the AgentSkill contract."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path

    def instructions(self, package_context: str) -> str:
        """Startup instructions for the agent: description, body, package facts."""
        return (
            f"# Skill: {self.name}\n\n{self.description}\n\n"
            f"{self.body.strip()}\n\n"
            f"# Package context\n\n{package_context.strip()}\n"
        )


def _parse_scalar(raw: str, key: str, path: Path) -> str:
    value = raw.strip()
    if value[:1] in {"'", '"'}:
        if len(value) < 2 or value[-1] != value[0]:
            raise SkillError(f"{path}: unterminated quoted value for {key!r}")
        return value[1:-1]
    if value in {"|", "|-", ">", ">-"}:
        raise SkillError(
            f"{path}: {key!r} uses a block scalar; write it on one line "
            "(the package skill parser supports single-line header values)"
        )
    return value


def _parse_header(header: str, path: Path) -> dict:
    fields = {}
    for line_number, line in enumerate(header.splitlines(), start=2):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0] in " \t":
            # Indented lines belong to a nested block we do not need, such as
            # `metadata:`. They are skipped rather than misread as top level.
            continue
        match = TOP_LEVEL_KEY.match(line)
        if not match:
            raise SkillError(f"{path}:{line_number}: cannot parse frontmatter line {line!r}")
        key = match.group("key")
        fields[key] = _parse_scalar(match.group("value") or "", key, path)
    return fields


def load_skill(skill_path) -> Skill:
    """Load ``SKILL.md`` from a path to the file or to its directory."""
    path = Path(skill_path)
    if path.is_dir():
        path = path / SKILL_FILENAME
    if not path.is_file():
        raise SkillError(
            f"missing skill file {path} — expected {SKILL_FILENAME} in the skill directory"
        )
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER.match(text)
    if not match:
        raise SkillError(
            f"{path}: no YAML frontmatter — the file must open with '---' "
            "and close the header with another '---' line"
        )
    header = _parse_header(match.group("header"), path)
    body = match.group("body").strip()
    problems = []
    name = header.get("name", "")
    if not name:
        problems.append("frontmatter field 'name' is required")
    elif not NAME_PATTERN.match(name) or len(name) > 64:
        problems.append(
            f"name {name!r} must be lowercase letters, digits and single hyphens (max 64)"
        )
    elif path.parent.name != name:
        problems.append(
            f"name {name!r} must match the skill directory {path.parent.name!r}"
        )
    description = header.get("description", "")
    if not description:
        problems.append("frontmatter field 'description' is required")
    elif len(description) > MAX_DESCRIPTION_CHARS:
        problems.append(
            f"description is {len(description)} characters; the limit is "
            f"{MAX_DESCRIPTION_CHARS}"
        )
    if not body:
        problems.append("skill body is empty — there are no instructions to load")
    if problems:
        raise SkillError(f"{path}: " + "; ".join(problems))
    return Skill(name=name, description=description, body=body, path=path)

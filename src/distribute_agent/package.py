"""Resolve and validate the shipped package without importing its internals.

The package is a directory of data plus one standard-library script. This module
reads the manifest and endpoint snapshot as JSON, checks that the files a run
needs exist, and turns template metadata into instructions the agent sees. An
incomplete or malformed package fails here, before any agent code is generated.
"""

import json
from dataclasses import dataclass
from pathlib import Path

MANIFEST_FILENAME = "manifest.json"
ENDPOINTS_FILENAME = "endpoints.json"
RUNNER_RELATIVE_PATH = Path("bin") / "rp_oneshot.py"
WORKFLOWS_DIRNAME = "workflows"
SKILLS_DIRNAME = "skills"


class PackageError(Exception):
    """The package is missing a file or its JSON does not match the contract."""


@dataclass(frozen=True)
class TemplateContext:
    name: str
    family: str
    checkpoint: str
    trigger_word: str
    size_guidance: str
    accepts_image: bool

    def as_instructions(self) -> str:
        trigger = self.trigger_word or "(none declared)"
        image = "yes" if self.accepts_image else "no (text-only template)"
        return (
            f"## Template: {self.name}\n"
            f"- model_family: {self.family or '(not declared)'}\n"
            f"- checkpoint: {self.checkpoint or '(not declared)'}\n"
            f"- trigger_word: {trigger}\n"
            f"- image input: {image}\n"
            f"- size guidance: {self.size_guidance or '(not declared)'}\n"
        )


def load_json_mapping(path: Path, label: str) -> dict:
    if not path.is_file():
        raise PackageError(f"missing {label} {path} — the package is incomplete")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PackageError(f"{label} {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PackageError(f"{label} {path} must be a JSON object, got {type(data).__name__}")
    return data


@dataclass(frozen=True)
class Package:
    root: Path
    manifest: dict
    endpoints: dict

    @classmethod
    def load(cls, root) -> "Package":
        root = Path(root).resolve()
        if not root.is_dir():
            raise PackageError(f"package root {root} is not a directory")
        runner = root / RUNNER_RELATIVE_PATH
        if not runner.is_file():
            raise PackageError(f"missing runner {runner} — the package is incomplete")
        manifest = load_json_mapping(root / MANIFEST_FILENAME, "manifest")
        endpoints = load_json_mapping(root / ENDPOINTS_FILENAME, "endpoint snapshot")
        templates = manifest.get("templates")
        if not isinstance(templates, dict) or not templates:
            raise PackageError(
                f"manifest {root / MANIFEST_FILENAME} has no 'templates' object — "
                "the package is incomplete"
            )
        for name, meta in templates.items():
            if not isinstance(meta, dict):
                raise PackageError(f"manifest template {name!r} must be an object")
            workflow = root / WORKFLOWS_DIRNAME / f"{name}.json"
            if not workflow.is_file():
                raise PackageError(f"missing workflow {workflow} for template {name!r}")
            endpoint = endpoints.get(name)
            if not isinstance(endpoint, dict) or not endpoint.get("endpoint_id"):
                raise PackageError(
                    f"no endpoint recorded for template {name!r} in "
                    f"{root / ENDPOINTS_FILENAME} — ask the maintainer for a refreshed "
                    "package"
                )
        if not (root / SKILLS_DIRNAME).is_dir():
            raise PackageError(
                f"missing skills directory {root / SKILLS_DIRNAME} — the package is incomplete"
            )
        return cls(root=root, manifest=manifest, endpoints=endpoints)

    @property
    def runner_path(self) -> Path:
        return self.root / RUNNER_RELATIVE_PATH

    def template_names(self) -> tuple:
        return tuple(sorted(self.manifest["templates"]))

    def skill_path(self, name: str = "runpod-oneshot") -> Path:
        path = self.root / SKILLS_DIRNAME / name
        if not path.is_dir():
            raise PackageError(
                f"missing skill directory {path}; shipped skills: "
                f"{', '.join(sorted(entry.name for entry in (self.root / SKILLS_DIRNAME).iterdir()))}"
            )
        return path

    def template(self, name: str) -> TemplateContext:
        templates = self.manifest["templates"]
        if name not in templates:
            raise PackageError(
                f"unknown template {name!r}; shipped: {', '.join(self.template_names())}"
            )
        meta = templates[name]
        model_info = meta.get("model_info") or {}
        size = meta.get("size") or {}
        image = meta.get("image") or {}
        return TemplateContext(
            name=name,
            family=str(meta.get("model_family") or ""),
            checkpoint=str(model_info.get("checkpoint") or ""),
            trigger_word=str(model_info.get("trigger_word") or ""),
            size_guidance=str(size.get("guidance") or ""),
            accepts_image=bool(image.get("node")),
        )

    def endpoint_id(self, name: str) -> str:
        entry = self.endpoints.get(name)
        if not isinstance(entry, dict) or not entry.get("endpoint_id"):
            raise PackageError(
                f"no endpoint recorded for template {name!r} — ask the maintainer "
                "for a refreshed package"
            )
        return str(entry["endpoint_id"])

    def template_context(self) -> str:
        return "\n".join(self.template(name).as_instructions() for name in self.template_names())

    def instructions(self, skill) -> str:
        return skill.instructions(self.template_context())

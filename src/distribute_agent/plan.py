"""Validate a structured oneshot request before anything privileged happens.

A plan is data the agent produces and the host may later submit. It contains no
paths, commands or free-form runner flags: only the fields the shipped runner
supports, each range-checked against the same limits the runner enforces. Any
unsupported field (for example an image for a text-only template) is rejected
here, before a bridge or a vendor call exists.
"""

import json
from dataclasses import dataclass
from typing import Optional

MAX_SEED = 2**64 - 1
MAX_DIMENSION = 16384
DIMENSION_STEP = 8
PLAN_FIELDS = ("template", "prompt", "negative", "seed", "width", "height")


class PlanError(Exception):
    """The plan is not a request the shipped runner can execute."""


@dataclass(frozen=True)
class OneshotPlan:
    template: str
    prompt: Optional[str] = None
    negative: Optional[str] = None
    seed: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    def to_argv(self, output_dir) -> list:
        argv = ["--template", self.template, "--output", str(output_dir)]
        for flag, value in (
            ("--prompt", self.prompt),
            ("--negative", self.negative),
            ("--seed", self.seed),
            ("--width", self.width),
            ("--height", self.height),
        ):
            if value is not None:
                argv.extend([flag, str(value)])
        return argv


def parse_plan(manifest: dict, data) -> OneshotPlan:
    """Validate agent-produced plan data against the shipped manifest."""
    if not isinstance(data, dict):
        raise PlanError(f"plan must be a JSON object, got {type(data).__name__}")
    unknown = sorted(set(data) - set(PLAN_FIELDS))
    if unknown:
        hint = ""
        if "image" in unknown:
            hint = (
                " — this package's template is text-only; image input is not part of a "
                "plan and the runner rejects --image for templates without an image node"
            )
        raise PlanError(f"unsupported plan field(s): {', '.join(unknown)}{hint}")
    templates = (manifest or {}).get("templates")
    if not isinstance(templates, dict) or not templates:
        raise PlanError("manifest has no templates; the package is incomplete")
    template = data.get("template")
    if not isinstance(template, str) or not template:
        raise PlanError("plan field 'template' must be a non-empty string")
    if template not in templates:
        raise PlanError(
            f"unknown template {template!r}; shipped: {', '.join(sorted(templates))}"
        )
    prompt = _optional_text(data, "prompt")
    negative = _optional_text(data, "negative")
    seed = data.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PlanError("plan field 'seed' must be an integer or omitted")
        if not 0 <= seed <= MAX_SEED:
            raise PlanError(f"plan field 'seed' must be 0..{MAX_SEED}")
    width = _dimension(data, "width")
    height = _dimension(data, "height")
    return OneshotPlan(
        template=template,
        prompt=prompt,
        negative=negative,
        seed=seed,
        width=width,
        height=height,
    )


def plan_schema(manifest: dict, template: str = None) -> dict:
    """JSON Schema for the flat object :func:`parse_plan` accepts.

    The field names, ranges and step come from the same constants ``parse_plan``
    enforces, and the template enum comes from the active manifest. Constraints
    JSON Schema cannot express (NUL bytes in text, booleans and numeric strings
    are not integers) are stated in :func:`plan_output_instructions`.

    ``template`` narrows the enum to one declared name: a caller that already
    chose the template from a stable category must not let the agent pick
    another.
    """
    templates = (manifest or {}).get("templates")
    if not isinstance(templates, dict) or not templates:
        raise PlanError("manifest has no templates; the package is incomplete")
    if template is not None and template not in templates:
        raise PlanError(f"pinned template {template!r} is not declared by the manifest")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["template"],
        "properties": {
            "template": {
                "type": "string",
                "enum": [template] if template is not None else sorted(templates),
                "description": "Template name declared by the active manifest.",
            },
            "prompt": {
                "type": ["string", "null"],
                "description": (
                    "Positive prompt without NUL bytes; null or omitted keeps the "
                    "template's value."
                ),
            },
            "negative": {
                "type": ["string", "null"],
                "description": (
                    "Negative prompt without NUL bytes; null or omitted keeps the "
                    "template's value."
                ),
            },
            "seed": {
                "type": ["integer", "null"],
                "minimum": 0,
                "maximum": MAX_SEED,
                "description": "JSON integer, not a boolean, numeric string or float.",
            },
            "width": _dimension_schema("width"),
            "height": _dimension_schema("height"),
        },
    }


def plan_output_instructions(manifest: dict, template: str = None) -> str:
    """Agent instructions pairing the flat-plan schema with its prose rules.

    A pinned ``template`` replaces the enum and states the choice, so the agent
    spends no step weighing templates the caller has already decided.
    """
    schema = json.dumps(plan_schema(manifest, template), indent=2)
    if template is None:
        template_rule = (
            "- `template` is required and must be one of the manifest's templates.\n"
        )
    else:
        template_rule = (
            f"- `template` is required and must be `{template}`: the request's "
            "category already chose it, so do not evaluate other templates.\n"
        )
    return (
        "\n# Final answer: one flat plan\n\n"
        "This run produces one plan. End it by calling `final_answer(plan)` with the "
        "flat JSON object below: no `plan` wrapper, no prose, no markdown, no "
        "JSON-encoded string, and no fields beyond these:\n\n"
        f"{schema}\n\n"
        "Field rules:\n"
        f"{template_rule}"
        "- `prompt` and `negative` are strings with no NUL byte; omit them or pass "
        "null to keep the template's value.\n"
        "- `seed`, `width` and `height` are JSON integers: booleans, numeric strings "
        "and floats are rejected.\n"
        f"- `seed` is 0..{MAX_SEED}; `width` and `height` are 1..{MAX_DIMENSION} "
        f"and multiples of {DIMENSION_STEP}.\n"
        "- Use `negative`, never `negative_prompt`; do not add commands, argv, "
        "endpoint ids, exit codes, image inputs or other metadata.\n"
        "`run_oneshot_preview(...)` returns the dry-run envelope plus argv and exit "
        "code; that dict is evidence, not a plan. Build the flat object from the "
        "values you verified. The host validates your final answer with the same "
        "parser the runner uses and rejects a mismatch with the reason, so correct "
        "it and call `final_answer` again.\n"
    )


def _dimension_schema(label):
    return {
        "type": ["integer", "null"],
        "minimum": 1,
        "maximum": MAX_DIMENSION,
        "multipleOf": DIMENSION_STEP,
        "description": (
            f"Output {label} in pixels: a JSON integer (not a boolean, numeric string "
            f"or float), a multiple of {DIMENSION_STEP}."
        ),
    }


def _optional_text(data, name):
    value = data.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise PlanError(f"plan field {name!r} must be a string or omitted")
    if "\x00" in value:
        raise PlanError(f"plan field {name!r} must not contain NUL bytes")
    return value


def _dimension(data, name):
    value = data.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"plan field {name!r} must be an integer or omitted")
    if value <= 0 or value > MAX_DIMENSION:
        raise PlanError(f"plan field {name!r} must be between 1 and {MAX_DIMENSION}")
    if value % DIMENSION_STEP:
        raise PlanError(f"plan field {name!r} must be a multiple of {DIMENSION_STEP}")
    return value

"""Opt-in semantic evaluation for a real model (5 briefs x 3 repetitions).

Two different questions exist here and this module keeps them apart:

* the tests in ``tests/test_distribute_agent.py`` use a scripted model to prove the
  skill, package metadata, sandbox and runner are wired together;
* this harness asks a *real* model to follow the skill. It costs money and
  needs a provider, so it only runs when the caller passes ``--enable-live-model``
  and a model factory. Without that it returns ``status: "not-run"``.

Results record model, executor and skill versions plus per-brief semantic
checks. Explicit settings must survive verbatim; a brief's prompt must carry its
subject, action and setting term groups; the Unicode brief's prompt must match
exactly; the invalid briefs must not produce a submittable plan and must state a
relevant actionable reason.

A generation brief must also show a successful ``run_oneshot_preview`` whose
dry-run request body matches the returned plan. That evidence comes from the
executor's host-side validation of the prelude's invocation record, so it proves
the capability ran and succeeded, not that sandbox-authored files are tamperproof.
Generation briefs are built with the strict flat-plan contract and the native
``final_answer_checks``; rejection briefs are built as prose-refusal runs.

Exclusions accept no/without/...-free phrases in the positive prompt or exact
comma-separated terms in the negative prompt. The term-group and exclusion checks
are labelled keyword heuristics, not complete semantic verification of an image
prompt.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .plan import PlanError, parse_plan

DEFAULT_REPETITIONS = 3
DEFAULT_MAX_STEPS = 6
DEFAULT_DEADLINE_SECONDS = 180.0


class LiveEvalError(RuntimeError):
    """The opt-in live harness was enabled without the configuration it needs."""


@dataclass(frozen=True)
class Brief:
    id: str
    task: str
    expected_settings: dict = field(default_factory=dict)
    expect_rejection: bool = False
    required_prompt_terms: tuple = ()
    required_prompt_term_groups: tuple = ()
    expected_prompt: str = None
    rejection_term_groups: tuple = ()
    required_exclusion_terms: tuple = ()

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "expected_settings": dict(self.expected_settings),
            "expect_rejection": self.expect_rejection,
            "required_prompt_terms": list(self.required_prompt_terms),
            "required_prompt_term_groups": [
                list(group) for group in self.required_prompt_term_groups
            ],
            "expected_prompt": self.expected_prompt,
            "rejection_term_groups": [list(group) for group in self.rejection_term_groups],
            "required_exclusion_terms": list(self.required_exclusion_terms),
        }


BRIEFS = (
    Brief(
        id="simple",
        task=(
            "Use the runpod-oneshot skill to plan one image: a photoreal portrait of "
            "a lighthouse keeper winding a brass clock on a stormy cliff. Call "
            "run_oneshot_preview to check the plan, then return a JSON object with the "
            "fields template, prompt, seed, width and height."
        ),
        required_prompt_terms=("lighthouse", "clock", "cliff"),
        # Subject, winding action and setting: `lighthouse clock cliff` is a list of
        # nouns, not the brief.
        required_prompt_term_groups=(
            ("lighthouse keeper", "keeper"),
            ("winding", "winds", "wound"),
            ("cliff", "headland", "coast"),
        ),
    ),
    Brief(
        id="explicit-settings",
        task=(
            "Plan one image with seed 42 at 832x1216: a rain-soaked cyberpunk alley at "
            "night, neon reflections, no text. Keep seed and size exactly as given; "
            "verify with run_oneshot_preview and return the plan as JSON."
        ),
        expected_settings={"seed": 42, "width": 832, "height": 1216},
        required_prompt_terms=("cyberpunk", "alley"),
        required_exclusion_terms=("text", "lettering", "writing", "letters", "words"),
    ),
    Brief(
        id="unicode-quotes",
        task=(
            "Plan one image whose prompt is exactly the quoted text between the markers: "
            '<<<「雨の街」 — a café terrace, “warm light”, 35mm film >>>. '
            "Preserve the quotes and non-Latin characters verbatim, verify the prompt "
            "with run_oneshot_preview, and return the plan as JSON."
        ),
        expected_prompt="「雨の街」 — a café terrace, “warm light”, 35mm film",
    ),
    Brief(
        id="unknown-template",
        task=(
            "Plan one image using template 'not-a-shipped-template'. If the template "
            "does not exist, report the actionable error instead of inventing an "
            "endpoint or claiming a job was submitted."
        ),
        expect_rejection=True,
        rejection_term_groups=(
            ("not-a-shipped-template",),
            ("unknown", "not exist", "does not exist", "no such", "not shipped"),
        ),
    ),
    Brief(
        id="unsupported-image",
        task=(
            "Plan one image for the shipped unholy-sdxl template and attach the local "
            "file input.png as an image input. If this template is text-only, report the "
            "actionable error instead of dropping the image silently or claiming a job "
            "was submitted."
        ),
        expect_rejection=True,
        rejection_term_groups=(
            ("image",),
            ("text-only", "text only", "no image", "does not accept", "not accept",
             "reject", "unsupported"),
        ),
    ),
)


def run_live_eval(
    *,
    enable_live_model: bool,
    model_factory=None,
    package_root=None,
    workspace_root=None,
    output_dir=None,
    repetitions: int = DEFAULT_REPETITIONS,
    max_steps: int = DEFAULT_MAX_STEPS,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    max_requests=None,
    model_metadata=None,
) -> dict:
    """Run the 5x3 semantic matrix with a caller-supplied real model."""
    report = {
        "status": "not-run",
        "reason": (
            "live semantic evaluation is opt-in: pass --enable-live-model with a "
            "caller-supplied model configuration and budget. No provider was called "
            "and no semantic result is claimed"
        ),
        "briefs": [brief.as_dict() for brief in BRIEFS],
        "repetitions": repetitions,
        "runs": [],
        "model": dict(model_metadata or {}),
        "bounds": {
            "repetitions": repetitions,
            "max_steps": max_steps,
            "deadline_seconds": deadline_seconds,
            "max_requests": max_requests,
        },
    }
    if not enable_live_model:
        return report
    if model_factory is None:
        raise LiveEvalError(
            "enable_live_model=True needs a model_factory that returns a smolagents "
            "Model configured by the caller (model id, provider, key environment "
            "variable). No default provider is chosen here"
        )
    if package_root is None or workspace_root is None:
        raise LiveEvalError("enable_live_model=True needs package_root and workspace_root")
    if repetitions < 1:
        raise LiveEvalError("repetitions must be at least 1")
    if max_requests is not None and max_requests < 1:
        raise LiveEvalError("max_requests must be at least 1 when set")

    from .agent import build_agent, redact, run_agent
    from .model_budget import ModelBudget, ModelBudgetExceeded
    from .package import Package
    from .skill import load_skill

    package = Package.load(package_root)
    skill = load_skill(package.skill_path())
    workspace_root = Path(workspace_root)
    report["environment"] = _environment_metadata(skill)
    budget = ModelBudget(max_requests)
    runs = report["runs"]
    for brief in BRIEFS:
        for repetition in range(1, repetitions + 1):
            if max_requests is not None and budget.requests >= max_requests:
                report["status"] = "budget-exhausted"
                report["requests_used"] = budget.requests
                return _write_report(report, output_dir)
            run_started = time.monotonic()
            requests_before = budget.requests
            entry = {
                "brief": brief.id,
                "repetition": repetition,
                "status": "fail",
                "checks": [],
                "answer": None,
                "error": None,
            }
            try:
                model = model_factory()
                # A refusal brief is graded on prose; every brief that asks for a plan
                # gets the strict contract and the runner's own parser.
                agent = build_agent(
                    model=model,
                    skill=skill,
                    package=package,
                    workspace_dir=workspace_root / f"{brief.id}-{repetition}",
                    max_steps=max_steps,
                    deadline_seconds=deadline_seconds,
                    model_budget=budget,
                    plan_manifest=None if brief.expect_rejection else package.manifest,
                )
                result = run_agent(agent, brief.task, deadline_seconds=deadline_seconds)
                entry["answer"] = redact(json.dumps(result.answer, default=str))[:4000]
                entry["checks"] = _check_answer(brief, result.answer, package)
                entry["checks"].append(
                    _preview_evidence(
                        brief, result.answer, package, agent.python_executor.preview_runs
                    )
                )
                entry["status"] = "pass" if all(check["ok"] for check in entry["checks"]) else "fail"
            except ModelBudgetExceeded as exc:
                entry["error"] = redact(str(exc))[:2000]
                entry["checks"] = [{"name": "run completed", "ok": False,
                                    "detail": "the request budget stopped this run"}]
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
                entry["error"] = redact(str(exc))[:2000]
                entry["checks"] = [{"name": "run completed", "ok": False,
                                    "detail": "the agent run raised; see error"}]
            entry["model_requests"] = budget.requests - requests_before
            entry["duration_seconds"] = round(time.monotonic() - run_started, 2)
            runs.append(entry)
    report["requests_used"] = budget.requests
    report["status"] = "completed" if all(r["status"] == "pass" for r in runs) else "failed"
    return _write_report(report, output_dir)


def _check_answer(brief: Brief, answer, package) -> list:
    """Semantic checks for one answer: the plan must be valid *and* on-brief."""
    if brief.expect_rejection:
        return _check_rejection(brief, answer, package)
    checks = []
    try:
        plan = parse_plan(package.manifest, _coerce_plan(answer))
    except PlanError as exc:
        return [{"name": "plan valid", "ok": False, "detail": str(exc)[:300]}]
    checks.append({"name": "plan valid", "ok": True})
    for name, expected in brief.expected_settings.items():
        actual = getattr(plan, name)
        checks.append({
            "name": f"{name} preserved",
            "ok": actual == expected,
            "detail": f"expected {expected!r}, got {actual!r}",
        })
    prompt = plan.prompt or ""
    checks.append({"name": "prompt non-empty", "ok": bool(prompt.strip())})
    if brief.expected_prompt is not None:
        checks.append({
            "name": "prompt exact",
            "ok": prompt == brief.expected_prompt,
            "detail": f"expected {brief.expected_prompt!r}, got {prompt!r}",
        })
    if brief.required_prompt_terms:
        missing = [
            term for term in brief.required_prompt_terms
            if term.casefold() not in prompt.casefold()
        ]
        checks.append({
            "name": "prompt keeps the brief",
            "ok": not missing,
            "detail": f"heuristic term check, not semantic verification; "
                      f"missing {missing} from {prompt[:200]!r}",
        })
    if brief.required_prompt_term_groups:
        lowered = prompt.casefold()
        missing_groups = [
            list(group) for group in brief.required_prompt_term_groups
            if not any(term.casefold() in lowered for term in group)
        ]
        checks.append({
            "name": "prompt keeps subject action and setting",
            "ok": not missing_groups,
            "detail": f"heuristic keyword groups, not full semantic verification; "
                      f"no term from {missing_groups} in {prompt[:200]!r}",
        })
    if brief.required_exclusion_terms:
        positive_exclusion = any(
            re.search(
                rf"\b(?:no|without)\s+{re.escape(term)}\b|\b{re.escape(term)}[- ]free\b",
                prompt,
                re.IGNORECASE,
            )
            for term in brief.required_exclusion_terms
        )
        negative_terms = {
            term.strip().casefold() for term in (plan.negative or "").split(",")
        }
        negative_exclusion = bool(negative_terms.intersection(
            term.casefold() for term in brief.required_exclusion_terms
        ))
        checks.append({
            "name": "prompt keeps exclusions",
            "ok": positive_exclusion or negative_exclusion,
            "detail": "heuristic keyword check, not full semantic verification; "
                      f"positive exclusion={positive_exclusion}, "
                      f"negative exclusion={negative_exclusion}; "
                      f"expected any of {list(brief.required_exclusion_terms)!r}",
        })
    return checks


def _preview_evidence(brief: Brief, answer, package, preview_runs) -> dict:
    """A generation brief needs a successful preview whose request matches the plan.

    Source text is not evidence: a comment, an unexecuted branch or a preview of a
    different plan changes nothing here. Only the executor's host-validated records
    of exit-0 dry runs count, and a rejection brief needs none at all.
    """
    if brief.expect_rejection:
        return {
            "name": "preview not required",
            "ok": True,
            "detail": "this brief asks for an actionable refusal; no preview is expected",
        }
    try:
        plan = parse_plan(package.manifest, _coerce_plan(answer))
    except PlanError as exc:
        return {
            "name": "preview executed and matched",
            "ok": False,
            "detail": f"the answer is not a valid plan: {str(exc)[:200]}",
        }
    runs = list(preview_runs or [])
    matched = [run for run in runs if _preview_matches_plan(package.manifest, plan, run)]
    return {
        "name": "preview executed and matched",
        "ok": bool(matched),
        "detail": (
            f"{len(matched)} of {len(runs)} validated preview(s) match the returned "
            f"plan for {plan.template}; the record is sandbox-authored evidence, not "
            "a tamperproof attestation"
        ),
    }


def _preview_matches_plan(manifest, plan, run) -> bool:
    """Compare the settings the runner actually put in its request body."""
    settings = _request_settings(manifest, run.get("template"), run.get("envelope") or {})
    if settings is None or settings.get("template") != plan.template:
        return False
    for name in ("prompt", "negative", "seed", "width", "height"):
        expected = getattr(plan, name)
        if expected is not None and settings.get(name) != expected:
            return False
    return True


def _request_settings(manifest, template, envelope):
    """Read the override values back out of a runner dry-run request body.

    The runner writes overrides into the template's manifest-declared nodes, so the
    host can compare the request the preview actually built with the plan the agent
    returned instead of trusting the agent's summary of it.
    """
    templates = (manifest or {}).get("templates")
    if not isinstance(template, str) or not isinstance(templates, dict):
        return None
    meta = templates.get(template)
    if not isinstance(meta, dict):
        return None
    body = envelope.get("input") if isinstance(envelope, dict) else None
    workflow = body.get("workflow") if isinstance(body, dict) else None
    if not isinstance(workflow, dict):
        return None

    def read(spec, field_key, default_field):
        spec = spec if isinstance(spec, dict) else {}
        node = workflow.get(spec.get("node"))
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            return None
        return inputs.get(spec.get(field_key, default_field))

    return {
        "template": template,
        "prompt": read(meta.get("positive"), "field", "text"),
        "negative": read(meta.get("negative"), "field", "text"),
        "seed": read(meta.get("seed"), "field", "seed"),
        "width": read(meta.get("size"), "width_field", "width"),
        "height": read(meta.get("size"), "height_field", "height"),
    }


def _check_rejection(brief: Brief, answer, package) -> list:
    """A rejected brief needs no submittable plan and a relevant, actionable answer."""
    text = _answer_text(answer)
    if not text.strip():
        return [{"name": "actionable rejection", "ok": False, "detail": "the answer is empty"}]
    try:
        parse_plan(package.manifest, _coerce_plan(answer))
    except PlanError:
        rejected = True
    else:
        rejected = False
    checks = [{
        "name": "no submittable plan",
        "ok": rejected,
        "detail": "a plan that the package would accept was returned for invalid input",
    }]
    lowered = text.casefold()
    missing_groups = [
        group for group in brief.rejection_term_groups
        if not any(term.casefold() in lowered for term in group)
    ]
    checks.append({
        "name": "actionable rejection",
        "ok": not missing_groups,
        "detail": (
            f"missing any of {missing_groups} in {text[:300]!r}"
            if missing_groups else f"{text[:300]!r}"
        ),
    })
    return checks


def _answer_text(answer) -> str:
    if isinstance(answer, str):
        return answer
    return json.dumps(answer, default=str)


def _coerce_plan(answer):
    if isinstance(answer, dict):
        return answer
    if isinstance(answer, str):
        text = answer.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except ValueError:
            return None
    return None


def _environment_metadata(skill) -> dict:
    try:
        import smolagents

        version = smolagents.__version__
    except ModuleNotFoundError:  # pragma: no cover - the harness needs it anyway
        version = "unavailable"
    digest = hashlib.sha256(skill.path.read_bytes()).hexdigest()
    return {
        "smolagents": version,
        "executor": "SandboxedPythonExecutor (bubblewrap)",
        "skill": {"name": skill.name, "sha256": digest},
    }


def _write_report(report: dict, output_dir) -> dict:
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "live-eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

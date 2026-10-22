"""Build and run the CodeAgent that drives the shipped package.

The skill file becomes the agent's instructions at startup, template metadata
comes from the package manifest (never hard-coded), and generated code runs only
through :class:`~distribute_agent.executor.SandboxedPythonExecutor`. This module
also owns the run budget and the redacted failure diagnostics.
"""

import json
import time
from importlib import resources
from pathlib import Path

import yaml
from smolagents import CodeAgent, Tool
from smolagents.monitoring import LogLevel

from .executor import (
    CAPABILITY_NAME,
    SandboxDeadlineExceeded,
    SandboxedPythonExecutor,
)
from .model_budget import BudgetedModel, ModelBudget
from .package import Package
from .plan import PlanError, parse_plan, plan_output_instructions
from .sandbox import SandboxPolicy, runtime_report

AUTHORIZED_IMPORTS = ["json", "subprocess", "pathlib", "time", "re", "base64"]

# SmolAgents' stock CodeAgent prompt promises that variables and imports persist
# between code blocks. The executor here starts a fresh interpreter per action,
# so the promise is replaced instead of silently broken.
PERSISTENT_STATE_CLAIM = (
    "10. The state persists between code executions: so if in one step you've "
    "created variables or imported modules, these will all persist."
)
STATELESS_STATE_RULE = (
    "10. Every code block runs in a fresh Python process: variables and imports "
    "from earlier blocks are gone. Save anything you need to reuse as a file under "
    "/workspace and read it back in the next block."
)


def code_agent_prompt_templates() -> dict:
    """The stock CodeAgent prompt with its persistent-state claim corrected."""
    templates = yaml.safe_load(
        resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
    )
    system_prompt = templates["system_prompt"]
    if PERSISTENT_STATE_CLAIM not in system_prompt:
        raise RuntimeError(
            "smolagents' CodeAgent system prompt no longer contains the persistent "
            "state claim this prototype replaces; review "
            "agent.STATELESS_STATE_RULE against the installed prompt"
        )
    templates = dict(templates)
    templates["system_prompt"] = system_prompt.replace(
        PERSISTENT_STATE_CLAIM, STATELESS_STATE_RULE
    )
    return templates


class AgentRunError(RuntimeError):
    """The run failed or exceeded its budget; ``diagnostics`` says what happened."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class AgentRunResult:
    def __init__(self, answer, state, duration_seconds, diagnostics):
        self.answer = answer
        self.state = state
        self.duration_seconds = duration_seconds
        self.diagnostics = diagnostics


class PreviewOneshotTool(Tool):
    """Prompt-facing declaration of the sandbox capability.

    ``forward`` is never called: the same function name and signature are
    defined by the sandbox prelude and executed inside bubblewrap. The tool
    object exists so CodeAgent's system prompt documents the capability.
    """

    name = CAPABILITY_NAME
    description = (
        "Run the packaged bin/rp_oneshot.py with --dry-run and return a dict with "
        "the parsed envelope (input.workflow), the runner argv, exit_code and "
        "stderr. No network, no API key, no submission. Use it to verify template "
        "overrides before the host submits anything."
    )
    inputs = {
        "template": {"type": "string", "description": "Template name from manifest.json."},
        "prompt": {"type": "string", "description": "Positive prompt text.", "nullable": True},
        "negative": {
            "type": "string",
            "description": "Negative prompt override; omit to keep the template's.",
            "nullable": True,
        },
        "seed": {"type": "integer", "description": "Fixed seed, or omit for the template policy.", "nullable": True},
        "width": {"type": "integer", "description": "Output width in pixels.", "nullable": True},
        "height": {"type": "integer", "description": "Output height in pixels.", "nullable": True},
    }
    output_type = "object"

    def forward(
        self, template, prompt=None, negative=None, seed=None, width=None, height=None
    ):  # pragma: no cover - executed only in the sandbox
        raise RuntimeError(
            "run_oneshot_preview runs inside the OS sandbox; this host-side stub "
            "must never execute"
        )


def build_agent(
    *,
    model,
    skill,
    package: Package,
    workspace_dir,
    max_steps: int = 8,
    deadline_seconds: float = 120.0,
    per_call_timeout: float = 30.0,
    max_model_requests: int = None,
    model_budget: ModelBudget = None,
    plan_manifest: dict = None,
) -> CodeAgent:
    """Build the agent; ``plan_manifest`` opts into the strict flat-plan contract.

    ``None`` (the default) is a generic wiring or refusal run: no schema is
    injected and the final answer is not constrained. Pass ``package.manifest``
    for any run whose answer will be submitted: the agent gets the flat-plan
    schema and the real :func:`~distribute_agent.plan.parse_plan` runs as a native
    final-answer check and again at the host boundary in :func:`run_agent`.
    """
    workspace_dir = Path(workspace_dir)
    sandbox_workspace = workspace_dir / "workspace"
    sandbox_workspace.mkdir(parents=True, exist_ok=True)
    policy = SandboxPolicy(
        package_root=package.root,
        workspace=sandbox_workspace,
        timeout_seconds=per_call_timeout,
    )
    executor = SandboxedPythonExecutor(
        policy,
        deadline_seconds=deadline_seconds,
        per_call_timeout=per_call_timeout,
    )
    instructions = package.instructions(skill) + (
        "\n# Execution boundary\n\n"
        "Generated Python runs inside a local OS sandbox: the package is mounted "
        "read-only at /package, only one writable workspace exists, the environment "
        "is cleared and there is no network. Each code block is a fresh process, so "
        "carry state between blocks through files under /workspace. The only command "
        "capability is run_oneshot_preview(...), which executes the real "
        "bin/rp_oneshot.py with "
        "--dry-run. Actual submission happens on the host outside the sandbox; do "
        "not claim a job was submitted from inside generated code.\n"
    )
    budget = model_budget or ModelBudget(max_model_requests)
    plan_checks = []
    if plan_manifest is not None:
        instructions += plan_output_instructions(plan_manifest)
        plan_checks.append(_plan_final_answer_check(plan_manifest))
    agent = CodeAgent(
        tools=[PreviewOneshotTool()],
        model=BudgetedModel(model, budget, deadline=lambda: executor.remaining_seconds),
        prompt_templates=code_agent_prompt_templates(),
        executor=executor,
        max_steps=max_steps,
        instructions=instructions,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        step_callbacks=[_deadline_callback(executor)],
        final_answer_checks=plan_checks,
        verbosity_level=LogLevel.ERROR,
    )
    agent.plan_manifest = plan_manifest
    return agent


def _plan_final_answer_check(manifest):
    """The native ``final_answer_checks`` hook: the real parser, no coercion."""

    def parse_plan_check(final_answer, memory=None, agent=None):
        parse_plan(manifest, final_answer)
        return True

    return parse_plan_check


def _deadline_callback(executor):
    """Hard stop for the run budget.

    SmolAgents treats executor errors as observations and keeps looping. The
    step callback runs in the step's ``finally`` block, so raising here aborts
    the run even when the executor error was swallowed.
    """

    def check(_step, agent=None):
        if executor.remaining_seconds <= 0:
            executor.deadline_exceeded = True
            raise SandboxDeadlineExceeded(
                f"wall-clock budget of {executor.deadline_seconds:.1f}s exhausted "
                f"after step {executor.call_count}; raise the budget or shorten the task"
            )

    return check


def run_agent(agent: CodeAgent, task: str, *, deadline_seconds: float = None) -> AgentRunResult:
    executor = agent.python_executor
    if deadline_seconds is not None:
        executor.reset_deadline(deadline_seconds)
    budget = executor.deadline_seconds
    started = time.monotonic()
    try:
        result = agent.run(
            task,
            max_steps=agent.max_steps,
            return_full_result=True,
        )
    except Exception as exc:
        diagnostics = _diagnostics(agent, started, error=str(exc), deadline=budget)
        raise AgentRunError(
            f"agent run failed after {diagnostics['duration_seconds']:.1f}s: "
            f"{diagnostics['error']}",
            diagnostics,
        ) from exc
    duration = time.monotonic() - started
    diagnostics = _diagnostics(agent, started, error=None, deadline=budget)
    diagnostics["state"] = result.state
    if executor.deadline_exceeded or duration > budget:
        diagnostics["error"] = "wall-clock budget exceeded"
        raise AgentRunError(
            f"agent run exceeded its {budget:.1f}s wall-clock budget "
            f"(took {duration:.1f}s)",
            diagnostics,
        )
    plan_manifest = getattr(agent, "plan_manifest", None)
    if plan_manifest is not None:
        # ``_handle_max_steps_reached`` bypasses final_answer_checks, so a strict
        # plan run validates the returned answer here as well, without coercion.
        # SmolAgents wraps a string answer in ``AgentText`` (a str subclass); unwrap
        # only that wrapper so the parser reports the real kind, and still reject it.
        answer = str(result.output) if isinstance(result.output, str) else result.output
        try:
            parse_plan(plan_manifest, answer)
        except PlanError as exc:
            diagnostics["error"] = redact(f"final answer is not a valid plan: {exc}")
            diagnostics["final_answer"] = redact(_answer_text(result.output))[:600]
            raise AgentRunError(
                f"agent returned an invalid plan after {duration:.1f}s "
                f"(state {result.state}): {diagnostics['error']}",
                diagnostics,
            ) from exc
    return AgentRunResult(
        answer=result.output,
        state=result.state,
        duration_seconds=duration,
        diagnostics=diagnostics,
    )


def _answer_text(answer) -> str:
    if isinstance(answer, str):
        return answer
    try:
        return json.dumps(answer, default=str)
    except (TypeError, ValueError):
        return repr(answer)


def _model_calls(agent) -> int:
    return len(getattr(agent.model, "calls", []) or [])


def _diagnostics(agent: CodeAgent, started: float, *, error, deadline: float) -> dict:
    last_code = ""
    for step in reversed(getattr(agent.memory, "steps", [])):
        code = getattr(step, "code_action", None)
        if code:
            last_code = code
            break
    executor = agent.python_executor
    return {
        "error": redact(error or ""),
        "duration_seconds": time.monotonic() - started,
        "model_requests": getattr(agent.model, "requests", 0),
        "wall_clock_budget_seconds": deadline,
        "executor_calls": getattr(executor, "call_count", 0),
        "validated_previews": len(getattr(executor, "preview_runs", [])),
        "model_calls": _model_calls(agent),
        "agent_steps": sum(
            1 for step in getattr(agent.memory, "steps", [])
            if getattr(step, "code_action", None)
        ),
        "deadline_exceeded": bool(getattr(executor, "deadline_exceeded", False)),
        "last_code": redact(last_code)[:600],
        "sandbox": runtime_report(executor.policy),
    }


def redact(text: str) -> str:
    """Remove credential-shaped strings before diagnostics are recorded or shown."""
    import re

    if not text:
        return ""
    patterns = (
        (re.compile(r"RUNPOD_API_KEY\s*=\s*\S+"), "RUNPOD_API_KEY=[redacted]"),
        (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [redacted]"),
        (re.compile(r"\bsk-[A-Za-z0-9_-]{6,}"), "[redacted]"),
        (re.compile(r"\bhf_[A-Za-z0-9]{8,}"), "[redacted]"),
        (re.compile(r"\brpa_[A-Za-z0-9]{6,}"), "[redacted]"),
    )
    for pattern, replacement in patterns:
        text = pattern.sub(replacement, text)
    return text

"""SmolAgents executor that runs generated Python through the OS sandbox.

The executor is the bridge between ``CodeAgent`` and :mod:`distribute_agent.sandbox`.
It deliberately implements the documented ``PythonExecutor`` interface rather
than subclassing ``LocalPythonExecutor``: generated code never touches this
process. The sandbox prelude exposes exactly one capability, a dry-run of the
real packaged ``rp_oneshot.py``, plus ``final_answer``.

Every action starts a fresh interpreter, so variables do not survive between
code blocks. ``agent.code_agent_prompt_templates`` replaces SmolAgents' stock
promise of persistent state with this contract, and state crosses blocks only as
files in the workspace. Exchange files are read and written without following
links and under a size cap, because generated code can plant both.

The prelude also records every ``run_oneshot_preview`` call and its exit status in
a workspace file. The host reads that file after each action and keeps only
records it can validate: exit 0, a dry-run argv and a parsed request body. The
record is sandbox-authored, so it is evidence that the capability ran, not a
tamperproof attestation.
"""

import json
import shutil
import time
from pathlib import Path

from smolagents.local_python_executor import CodeOutput, PythonExecutor

from .sandbox import (
    MAX_RESULT_BYTES,
    SandboxPolicy,
    SandboxResultRejected,
    read_result_bytes,
    run_sandboxed,
)

CAPABILITY_NAME = "run_oneshot_preview"
INTRINSIC_TOOLS = frozenset({"final_answer"})
PREVIEW_EVIDENCE_NAME = "preview-evidence.jsonl"
PREVIEW_EVIDENCE_MAX_BYTES = 256 * 1024
MAX_RETAINED_PREVIEWS = 16

_PRELUDE = '''"""Sandbox prelude: the only capabilities generated code receives."""
import json as _json
import subprocess as _subprocess
import sys as _sys

_PREVIEW_OUTPUT = "/workspace/out"
_PREVIEW_RESULT = "/workspace/result.json"
_PREVIEW_EVIDENCE = "/workspace/preview-evidence.jsonl"
_PREVIEW_RUNNER = "/package/bin/rp_oneshot.py"
_PREVIEW_TIMEOUT = 120
_PREVIEW_CAPTURE_CHARS = 120000


def _record_preview(argv, exit_code, envelope):
    """Append one invocation record; the host decides whether it counts."""
    try:
        record = _json.dumps(
            {"argv": argv, "exit_code": exit_code, "envelope": envelope},
            default=str,
        )
    except (TypeError, ValueError):
        record = _json.dumps({"argv": argv, "exit_code": exit_code, "envelope": None})
    try:
        with open(_PREVIEW_EVIDENCE, "a", encoding="utf-8") as handle:
            handle.write(record + "\\n")
    except OSError:
        pass


def run_oneshot_preview(template, prompt=None, negative=None, seed=None,
                        width=None, height=None):
    """Run the packaged rp_oneshot.py with --dry-run and return its envelope.

    This is the only command capability inside the sandbox. It cannot submit:
    the runner has no API key and the sandbox has no network.
    """
    argv = [
        _sys.executable,
        _PREVIEW_RUNNER,
        "--template", str(template),
        "--output", _PREVIEW_OUTPUT,
        "--dry-run",
    ]
    for flag, value in (
        ("--prompt", prompt),
        ("--negative", negative),
        ("--seed", seed),
        ("--width", width),
        ("--height", height),
    ):
        if value is not None:
            argv.extend([flag, str(value)])
    completed = _subprocess.run(
        argv, capture_output=True, text=True, timeout=_PREVIEW_TIMEOUT
    )
    envelope = None
    if completed.returncode == 0:
        try:
            envelope = _json.loads(completed.stdout)
        except ValueError:
            envelope = None
    _record_preview(argv[2:], completed.returncode, envelope)
    return {
        "argv": argv[2:],
        "exit_code": completed.returncode,
        "envelope": envelope,
        "stdout": completed.stdout[:_PREVIEW_CAPTURE_CHARS],
        "stderr": completed.stderr[:_PREVIEW_CAPTURE_CHARS],
    }


def final_answer(value):
    """Return the final answer to the host process and stop the program."""
    try:
        payload = _json.dumps({"value": value})
    except TypeError:
        payload = _json.dumps({"value": repr(value)})
    with open(_PREVIEW_RESULT, "w", encoding="utf-8") as handle:
        handle.write(payload)
    raise SystemExit(0)
'''


class SandboxConfigurationError(RuntimeError):
    """The agent declared tools the sandbox does not implement; refuse to run."""


class SandboxDeadlineExceeded(RuntimeError):
    """The run's wall-clock budget ran out while generated code was executing."""


class SandboxedPythonExecutor(PythonExecutor):
    """Execute every code action in a bubblewrap sandbox.

    ``deadline_seconds`` is a real wall-clock budget for the whole run: the
    per-call timeout shrinks to whatever is left, and exhausted budget raises
    instead of silently truncating the run.
    """

    def __init__(
        self,
        policy: SandboxPolicy,
        *,
        deadline_seconds: float = 120.0,
        per_call_timeout: float = None,
        max_output_chars: int = None,
        max_result_bytes: int = MAX_RESULT_BYTES,
    ):
        self.policy = policy
        self.deadline_seconds = float(deadline_seconds)
        self.per_call_timeout = float(
            policy.timeout_seconds if per_call_timeout is None else per_call_timeout
        )
        self.max_output_chars = (
            policy.max_output_chars if max_output_chars is None else max_output_chars
        )
        self.max_result_bytes = int(max_result_bytes)
        self.tool_names = set()
        self.variables = {}
        self.call_count = 0
        self.deadline_exceeded = False
        # Validated preview invocations, newest last: evidence for the live harness,
        # never a claim that generated code cannot forge a workspace file.
        self.preview_runs = []
        self._deadline = time.monotonic() + self.deadline_seconds

    def reset_deadline(self, seconds: float = None) -> None:
        self.deadline_seconds = float(seconds or self.deadline_seconds)
        self._deadline = time.monotonic() + self.deadline_seconds
        self.deadline_exceeded = False

    @property
    def remaining_seconds(self) -> float:
        return self._deadline - time.monotonic()

    def send_tools(self, tools) -> None:
        names = set(tools)
        allowed = {CAPABILITY_NAME} | INTRINSIC_TOOLS
        if CAPABILITY_NAME not in names or not names <= allowed:
            raise SandboxConfigurationError(
                f"the sandbox only implements {sorted(allowed)}; agent declared "
                f"{sorted(names)}. Add an explicit capability to executor.py and "
                "the sandbox prelude before exposing it."
            )
        self.tool_names = names

    def send_variables(self, variables) -> None:
        # Variables stay host-side on purpose: generated code must receive data
        # through the task text, never through host objects.
        self.variables = dict(variables)

    def __call__(self, code_action: str) -> CodeOutput:
        self.call_count += 1
        remaining = self.remaining_seconds
        if remaining <= 0:
            self.deadline_exceeded = True
            raise SandboxDeadlineExceeded(
                f"wall-clock budget of {self.deadline_seconds:.1f}s exhausted before "
                f"step {self.call_count}; raise the budget or shorten the task"
            )
        budget_bound = remaining <= self.per_call_timeout
        timeout = min(self.per_call_timeout, remaining)
        result_path = Path(self.policy.result_path)
        evidence_path = Path(self.policy.workspace) / PREVIEW_EVIDENCE_NAME
        _discard_entry(result_path)
        _discard_entry(evidence_path)
        program = _PRELUDE + "\n" + code_action + "\n"
        result = run_sandboxed(self.policy, program, timeout=timeout)
        self._record_preview_runs(evidence_path)
        if result.timed_out:
            if budget_bound:
                self.deadline_exceeded = True
                raise SandboxDeadlineExceeded(
                    f"wall-clock budget of {self.deadline_seconds:.1f}s exhausted during "
                    f"step {self.call_count}; generated code was stopped after "
                    f"{timeout:.1f}s"
                )
            return CodeOutput(
                output=(
                    f"Execution stopped after {timeout:.1f}s (per-step sandbox limit). "
                    "Reduce the work or split it across steps."
                ),
                logs=result.combined_tail,
                is_final_answer=False,
            )
        if result.output_limit_exceeded:
            return CodeOutput(
                output=(
                    f"Execution stopped: generated code wrote more than "
                    f"{self.policy.max_capture_bytes} bytes to one output stream. "
                    "Print a bounded summary instead of the whole payload."
                ),
                logs=_truncate(result.stdout, self.max_output_chars),
                is_final_answer=False,
            )
        if result.exit_code != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "(no output)"
            return CodeOutput(
                output=(
                    f"Sandboxed execution failed (exit {result.exit_code}):\n"
                    f"{_truncate(detail, self.max_output_chars)}"
                ),
                logs=_truncate(result.stdout, self.max_output_chars),
                is_final_answer=False,
            )
        try:
            raw = read_result_bytes(result_path, self.max_result_bytes)
        except FileNotFoundError:
            return CodeOutput(
                output=_truncate(result.stdout.strip(), self.max_output_chars) or "(no output)",
                logs=_truncate(result.stdout, self.max_output_chars),
                is_final_answer=False,
            )
        except SandboxResultRejected as exc:
            _discard_entry(result_path)
            return CodeOutput(
                output=f"Sandbox result rejected: {exc}",
                logs=_truncate(result.stdout, self.max_output_chars),
                is_final_answer=False,
            )
        _discard_entry(result_path)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            return CodeOutput(
                output=f"Sandbox wrote an unreadable result.json: {exc}",
                logs=_truncate(result.stdout, self.max_output_chars),
                is_final_answer=False,
            )
        if not isinstance(payload, dict) or "value" not in payload:
            return CodeOutput(
                output=(
                    "Sandbox result.json must be a JSON object with a 'value' key; "
                    f"got {type(payload).__name__}"
                ),
                logs=_truncate(result.stdout, self.max_output_chars),
                is_final_answer=False,
            )
        return CodeOutput(
            output=payload["value"],
            logs=_truncate(result.stdout, self.max_output_chars),
            is_final_answer=True,
        )

    def _record_preview_runs(self, evidence_path: Path) -> None:
        """Keep only preview records this host can validate for this action."""
        runs = _validated_preview_runs(evidence_path)
        _discard_entry(evidence_path)
        self.preview_runs.extend(runs)
        del self.preview_runs[:-MAX_RETAINED_PREVIEWS]


def _validated_preview_runs(path: Path) -> list:
    """Read the evidence file defensively and return well-formed successes.

    The file is written by the sandbox prelude, but generated code shares the
    workspace and could append its own line, so nothing here is treated as
    trusted input: the read is link-safe and byte-capped, and a record counts only
    when the runner exited 0 with ``--dry-run`` and printed a request body.
    """
    try:
        raw = read_result_bytes(path, PREVIEW_EVIDENCE_MAX_BYTES)
    except (FileNotFoundError, SandboxResultRejected):
        return []
    runs = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        run = _validated_preview(record)
        if run is not None:
            runs.append(run)
    return runs


def _validated_preview(record):
    """Accept one record only if it describes a successful packaged dry run."""
    if not isinstance(record, dict) or record.get("exit_code") != 0:
        return None
    argv = record.get("argv")
    if not isinstance(argv, list) or "--dry-run" not in argv:
        return None
    if not all(isinstance(argument, str) for argument in argv):
        return None
    if "--template" not in argv or argv.index("--template") + 1 >= len(argv):
        return None
    envelope = record.get("envelope")
    if not isinstance(envelope, dict):
        return None
    body = envelope.get("input")
    if not isinstance(body, dict) or not isinstance(body.get("workflow"), dict):
        return None
    return {
        "template": argv[argv.index("--template") + 1],
        "envelope": envelope,
    }


def _discard_entry(path: Path) -> None:
    """Remove one fixed workspace entry; never follow a link it may have become.

    Generated code can leave a directory or a read-only entry at a host-read path.
    That is the action's own mess: the next action still has to run, so the removal
    is best-effort and the reader rejects whatever is left.
    """
    try:
        path.unlink(missing_ok=True)
        return
    except IsADirectoryError:
        pass
    except OSError:
        return
    # A fixed exchange path inside the workspace; rmtree refuses a symlink itself.
    shutil.rmtree(path, ignore_errors=True)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} characters]"

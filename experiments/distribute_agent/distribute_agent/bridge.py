"""Host-side submission bridge: the only place a vendor credential is used.

Generated Python never sees ``RUNPOD_API_KEY`` and never reaches the network.
When a run is approved, the host validates the agent's plan and executes the
real packaged runner once. The bridge never retries: a timeout after the POST
is ambiguous, so it reports that ambiguity instead of submitting a second
billable job.
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .plan import parse_plan

DEFAULT_KEY_ENV = "RUNPOD_API_KEY"
# rp_oneshot.py prints these two markers; they are the runner's structured
# contract for "a job exists" versus "this failure says nothing either way".
SUBMISSION_ACCEPTED_PATTERN = re.compile(r"^\[info\] job (\S+) accepted$", re.MULTILINE)
SUBMISSION_UNKNOWN_PHRASE = "submission outcome is UNKNOWN"


class BridgeError(RuntimeError):
    """Submission failed or its outcome is unknown; ``may_have_submitted`` says which.

    ``job_id`` is set when the runner reported an accepted job, so a caller can
    look the job up instead of submitting a second billable one.
    """

    def __init__(self, message, *, may_have_submitted=False, job_id=None):
        super().__init__(message)
        self.may_have_submitted = may_have_submitted
        self.job_id = job_id


@dataclass(frozen=True)
class BridgeResult:
    returncode: int
    job_id: object
    output_dir: object
    stdout: str
    stderr: str


def submit_plan(
    package,
    plan_data,
    *,
    output_dir,
    api_key_env: str = DEFAULT_KEY_ENV,
    env=None,
    runner_prefix=None,
    wait_timeout_s: int = 1800,
    timeout: float = 3600.0,
) -> BridgeResult:
    """Validate ``plan_data`` and run the packaged runner exactly once."""
    plan = parse_plan(package.manifest, plan_data)
    output = Path(output_dir).resolve()
    package_root = Path(package.root).resolve()
    if output == package_root or package_root in output.parents:
        raise BridgeError(
            f"output directory {output} is inside the read-only package {package_root}; "
            "write job artifacts somewhere else"
        )
    environment = dict(os.environ if env is None else env)
    api_key = environment.get(api_key_env)
    if not api_key:
        raise BridgeError(
            f"{api_key_env} is not set — export your RunPod API key before submitting. "
            "The key is read from the environment and never written to artifacts"
        )
    prefix = tuple(runner_prefix) if runner_prefix else (
        sys.executable,
        str(package.runner_path),
    )
    argv = list(prefix) + plan.to_argv(output) + [
        "--wait",
        "--timeout-s",
        str(int(wait_timeout_s)),
    ]
    child_env = {
        "PATH": "/usr/bin:/bin",
        api_key_env: api_key,
        "HOME": str(output),
        "NO_COLOR": "1",
    }
    output.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=child_env,
            cwd=str(output),
        )
    except subprocess.TimeoutExpired as exc:
        raise BridgeError(
            f"runner did not finish within {timeout:.0f}s — the job may have been "
            "accepted and may still complete on RunPod; check the endpoint's jobs "
            "before resubmitting",
            may_have_submitted=True,
        ) from exc
    except OSError as exc:
        raise BridgeError(f"could not start the packaged runner: {exc}") from exc
    stdout = _scrub(completed.stdout, api_key)
    stderr = _scrub(completed.stderr, api_key)
    if completed.returncode != 0:
        accepted = SUBMISSION_ACCEPTED_PATTERN.search(stderr)
        job_id = accepted.group(1) if accepted else None
        may_have_submitted = job_id is not None or SUBMISSION_UNKNOWN_PHRASE in stderr
        if job_id is not None:
            consequence = (
                f" — job {job_id} was submitted; do not resubmit it. Read the existing "
                "job on the endpoint before running anything else"
            )
        elif may_have_submitted:
            consequence = (
                " — the submission may have been accepted; check the endpoint's jobs "
                "before rerunning"
            )
        else:
            consequence = ""
        raise BridgeError(
            f"runner exited {completed.returncode}: {_tail(stderr) or _tail(stdout)}"
            + consequence,
            may_have_submitted=may_have_submitted,
            job_id=job_id,
        )
    job_dir = _last_path_line(stdout)
    job_id = job_dir.name if job_dir else None
    return BridgeResult(
        returncode=completed.returncode,
        job_id=job_id,
        output_dir=job_dir,
        stdout=stdout,
        stderr=stderr,
    )


def _scrub(text: str, secret: str) -> str:
    return (text or "").replace(secret, "[redacted]") if secret else (text or "")


def _tail(text: str, limit: int = 1200) -> str:
    text = (text or "").strip()
    return text[-limit:]


def _last_path_line(stdout: str):
    for line in reversed((stdout or "").splitlines()):
        candidate = line.strip()
        if candidate and Path(candidate).is_dir():
            return Path(candidate).resolve()
    return None

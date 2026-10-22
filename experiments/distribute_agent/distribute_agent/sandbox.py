"""OS-enforced execution boundary for generated Python.

``LocalPythonExecutor`` is documented by SmolAgents as *not* a security sandbox:
its AST checks are a best-effort guard. Generated code here is untrusted, so it
runs in a separate process inside bubblewrap with:

* a fresh mount namespace and no host filesystem except the interpreter runtime,
  the package (read-only) and one writable workspace directory,
* an empty environment followed by the few variables the runtime needs, so no
  API key or credential of the host process is visible,
* no network namespace at all,
* all capabilities dropped, and the process killed with its parent.

Anything the sandbox cannot guarantee (bubblewrap missing, user namespaces
disabled, a runtime layout it cannot mount) raises :class:`SandboxUnavailable`
instead of falling back to in-process execution.
"""

import os
import selectors
import secrets
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BWRAP = Path("/usr/bin/bwrap")
DEFAULT_PYTHON = Path("/usr/bin/python3")
MAX_OUTPUT_CHARS = 8000
MAX_CAPTURE_BYTES = 256 * 1024
MAX_RESULT_BYTES = 64 * 1024
READ_CHUNK_BYTES = 64 * 1024


class SandboxUnavailable(RuntimeError):
    """The OS boundary cannot be established; generated code must not run."""


class SandboxResultRejected(RuntimeError):
    """The workspace holds a result file the host must not read."""


@dataclass(frozen=True)
class SandboxPolicy:
    package_root: Path
    workspace: Path
    python: Path = field(default_factory=lambda: Path(os.path.realpath(DEFAULT_PYTHON)))
    bwrap: Path = DEFAULT_BWRAP
    timeout_seconds: float = 60.0
    max_output_chars: int = MAX_OUTPUT_CHARS
    max_capture_bytes: int = MAX_CAPTURE_BYTES

    def __post_init__(self):
        root = Path(self.package_root).resolve()
        workspace = Path(self.workspace).resolve()
        if not Path(self.python).is_file():
            raise SandboxUnavailable(
                f"sandbox interpreter {self.python} does not exist; pass a system "
                "CPython (for example /usr/bin/python3)"
            )
        if workspace == root or root in workspace.parents:
            raise SandboxUnavailable(
                f"sandbox workspace {workspace} must be outside the package {root}"
            )
        if not Path(self.bwrap).is_file():
            raise SandboxUnavailable(
                f"bubblewrap not found at {self.bwrap}: generated code needs an OS "
                "sandbox. Install bubblewrap (for example 'apt install bubblewrap') "
                f"or pass bwrap=Path(...). Host python: {sys.executable}"
            )
        if self.max_capture_bytes <= 0 or self.max_output_chars <= 0:
            raise SandboxUnavailable(
                "sandbox output limits must be positive; pass max_capture_bytes and "
                "max_output_chars greater than zero"
            )

    @property
    def program_path(self) -> Path:
        return Path(self.workspace) / "program.py"

    @property
    def result_path(self) -> Path:
        return Path(self.workspace) / "result.json"


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool
    output_limit_exceeded: bool = False

    @property
    def combined_tail(self) -> str:
        return (self.stderr or "").strip() or (self.stdout or "").strip()


@dataclass(frozen=True)
class SandboxCommand:
    argv: tuple

    def describe(self) -> str:
        return " ".join(self.argv)


_PROBE_CACHE = set()


def build_argv(policy: SandboxPolicy, program_path: Path) -> SandboxCommand:
    """The complete bubblewrap invocation for one program.

    Only the interpreter, its shared libraries and loaders, the package and the
    workspace are visible. ``/usr/lib`` is mounted read-only because CPython's
    standard library and the dynamic loader live there; no command directories
    or host home directories are mounted.
    """
    package = Path(policy.package_root).resolve()
    workspace = Path(policy.workspace).resolve()
    python = Path(policy.python).resolve()
    argv = [
        str(policy.bwrap),
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop", "ALL",
        "--clearenv",
        "--setenv", "PATH", "/usr/bin",
        "--setenv", "HOME", "/workspace",
        "--setenv", "TMPDIR", "/tmp",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--ro-bind", "/usr/lib", "/usr/lib",
        "--ro-bind", "/usr/lib", "/lib64",
        "--ro-bind", str(python), "/usr/bin/python3",
        "--ro-bind", str(package), "/package",
        "--bind", str(workspace), "/workspace",
        "--chdir", "/workspace",
        "--", "/usr/bin/python3", f"/workspace/{Path(program_path).name}",
    ]
    return SandboxCommand(argv=tuple(argv))


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} characters]"


def write_program(path: Path, source: str) -> None:
    """Replace ``path`` with a regular file created by this process.

    Generated code shares the workspace and can plant a symlink at the program
    path. Writing through that link would hand that code the host user's write
    permissions outside the sandbox, so the file is created fresh with
    ``O_EXCL | O_NOFOLLOW`` and moved into place: ``os.replace`` swaps the
    directory entry and never follows the link target.
    """
    path = Path(path)
    directory = path.parent
    for _attempt in range(16):
        candidate = directory / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(
                candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
        except FileExistsError:
            continue
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(source)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(candidate, path)
        except BaseException:
            candidate.unlink(missing_ok=True)
            raise
        return
    raise SandboxUnavailable(
        f"could not create a fresh program file in {directory}; remove stale "
        f".{path.name}.* files and retry"
    )


def read_result_bytes(path: Path, max_bytes: int = MAX_RESULT_BYTES) -> bytes:
    """Read a sandbox result file without following links, blocking or over-reading.

    The result file is written inside the sandbox, so it is untrusted input: a
    symlink there would turn the host's read into a read of any host file, an
    oversized one would be parsed in host memory, and a FIFO there would block
    the host in ``open`` until a writer appeared. ``O_NOFOLLOW`` refuses links,
    ``O_NONBLOCK`` makes a FIFO open return immediately, and the descriptor is
    checked with ``fstat`` before a single byte is read, so only a regular file
    is ever consumed.
    """
    path = Path(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise SandboxResultRejected(
            f"{path.name} is not a regular file opened without following links "
            f"({exc}); generated code must write a plain file in the workspace"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise SandboxResultRejected(f"{path.name} is not a regular file")
        if info.st_size > max_bytes:
            raise SandboxResultRejected(
                f"{path.name} is {info.st_size} bytes; the cap is {max_bytes}"
            )
        chunks = []
        total = 0
        while total <= max_bytes:
            chunk = os.read(descriptor, READ_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if len(data) > max_bytes:
        raise SandboxResultRejected(
            f"{path.name} grew past the {max_bytes}-byte cap while reading"
        )
    return data


def probe(policy: SandboxPolicy) -> None:
    """Run one trivial program through the real sandbox command; cache success."""
    key = (
        str(Path(policy.bwrap).resolve()),
        str(Path(policy.python).resolve()),
        str(Path(policy.package_root).resolve()),
        str(Path(policy.workspace).resolve()),
    )
    if key in _PROBE_CACHE:
        return
    workspace = Path(policy.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    program = workspace / "probe.py"
    write_program(program, 'print("sandbox-ok")\n')
    command = build_argv(policy, program)
    try:
        completed = subprocess.run(
            list(command.argv),
            capture_output=True,
            text=True,
            timeout=min(30.0, policy.timeout_seconds),
            env={},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SandboxUnavailable(
            f"bubblewrap probe failed ({exc}); generated code needs a working OS "
            "sandbox. On Debian/Ubuntu install bubblewrap and enable unprivileged "
            "user namespaces (kernel.unprivileged_userns_clone=1). "
            f"Probe command: {command.describe()}"
        ) from exc
    finally:
        program.unlink(missing_ok=True)
    if completed.returncode != 0 or "sandbox-ok" not in completed.stdout:
        detail = (completed.stderr or completed.stdout or "").strip()[:500]
        hint = ""
        if "Operation not permitted" in detail or "permission" in detail.lower():
            hint = (
                " Unprivileged user namespaces appear to be blocked; enable them "
                "(kernel.unprivileged_userns_clone=1) or run on a host that allows them."
            )
        raise SandboxUnavailable(
            f"bubblewrap probe exited {completed.returncode}: {detail}.{hint} "
            f"Probe command: {command.describe()}"
        )
    _PROBE_CACHE.add(key)


def run_sandboxed(policy: SandboxPolicy, source: str, timeout: float = None) -> SandboxResult:
    """Run ``source`` as a Python program inside the sealed sandbox.

    Output is read through bounded buffers while the program runs: a program
    that writes without end is killed at ``max_capture_bytes`` instead of
    filling host memory until the timeout.
    """
    probe(policy)
    workspace = Path(policy.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    program = policy.program_path
    write_program(program, source)
    limit = policy.timeout_seconds if timeout is None else max(0.05, float(timeout))
    command = build_argv(policy, program)
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            list(command.argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={},
            cwd=str(workspace),
        )
    except OSError as exc:
        raise SandboxUnavailable(f"could not start the sandbox: {exc}") from exc
    captured, exceeded, timed_out = _drain(process, limit, policy.max_capture_bytes)
    return SandboxResult(
        exit_code=process.returncode,
        stdout=_truncate(captured["stdout"], policy.max_output_chars),
        stderr=_truncate(captured["stderr"], policy.max_output_chars),
        duration_seconds=time.monotonic() - started,
        timed_out=timed_out,
        output_limit_exceeded=exceeded,
    )


def _drain(process, timeout: float, capture_cap: int):
    """Collect both pipes under a byte cap and a deadline, then reap the process."""
    streams = {
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    buffers = {name: bytearray() for name in streams}
    descriptor_names = {stream.fileno(): name for name, stream in streams.items()}
    selector = selectors.DefaultSelector()
    for stream in streams.values():
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    exceeded = ran_out_of_time = False
    open_streams = len(streams)
    captured = {}
    timed_out = False
    try:
        while open_streams:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                ran_out_of_time = True
                break
            for key, _events in selector.select(timeout=remaining):
                name = descriptor_names.get(key.fd)
                if name is None:
                    continue
                chunk = os.read(key.fd, READ_CHUNK_BYTES)
                if not chunk:
                    selector.unregister(key.fileobj)
                    open_streams -= 1
                    continue
                buffer = buffers[name]
                buffer.extend(chunk)
                if len(buffer) > capture_cap:
                    exceeded = True
                    break
            if exceeded:
                break
        captured = {name: _decode(bytes(buffer)) for name, buffer in buffers.items()}
    finally:
        selector.close()
        still_running = process.poll() is None
        if still_running:
            try:
                process.kill()
            except OSError:
                pass
        process.wait()
        timed_out = ran_out_of_time and still_running
        for stream in streams.values():
            stream.close()
    return captured, exceeded, timed_out


def _decode(data) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    return str(data)


def runtime_report(policy: SandboxPolicy) -> dict:
    """Facts about the boundary, recorded in run artifacts and diagnostics."""
    return {
        "bwrap": str(Path(policy.bwrap).resolve()),
        "bwrap_version": _version(policy.bwrap),
        "python": str(Path(policy.python).resolve()),
        "package_root": str(Path(policy.package_root).resolve()),
        "workspace": str(Path(policy.workspace).resolve()),
        "network": "unshared",
        "host_home_mounted": False,
        "package_mount": "read-only at /package",
        "environment": "cleared; PATH/HOME/TMPDIR only",
    }


def _version(binary: Path) -> str:
    try:
        completed = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True, timeout=10
        )
        return (completed.stdout or completed.stderr).strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"

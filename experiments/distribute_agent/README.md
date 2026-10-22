# distribute_agent (prototype)

A SmolAgents `CodeAgent` that loads the shipped `runpod-oneshot` skill and drives
`distribute/bin/rp_oneshot.py`. The distributed package stays standalone and
standard-library only; this prototype is the only side that imports smolagents.
Dependency direction: `experiments/distribute_agent` → `distribute/` (read-only).

## Status

| Piece | State |
|---|---|
| E1 tracer bullet: real CodeAgent → generated Python → real runner `--dry-run` → parsed envelope | passing offline |
| Skill parsing, package validation, sandbox boundary, plan validation, runner request/job handling | passing offline |
| Live semantic model evaluation (5 briefs × 3 reps) | implemented, **not run** — needs an authorized provider and budget |
| Real vendor smoke test (one billed job) | implemented, **not run** — needs authorization, budget and `RUNPOD_API_KEY` |

No result in this repository claims a live model or vendor pass.

## Layout

| Path | Role |
|---|---|
| `distribute_agent/skill.py` | Parse and validate one `SKILL.md` (frontmatter subset, name/directory match, description limit, non-empty body) |
| `distribute_agent/package.py` | Resolve the package from any cwd; validate manifest, endpoint snapshot, workflows and runner; expose template metadata as instructions |
| `distribute_agent/sandbox.py` | Bubblewrap policy and the single sandboxed execution path; fails closed |
| `distribute_agent/executor.py` | SmolAgents `PythonExecutor` that sends every code action through the sandbox; prelude exposes `run_oneshot_preview` and `final_answer` |
| `distribute_agent/agent.py` | `build_agent`, `run_agent`, stateless-execution prompt contract, run budget, redacted failure diagnostics |
| `distribute_agent/model_budget.py` | Shared provider-request counter and per-call wall-clock gate around any SmolAgents model |
| `distribute_agent/scripted_model.py` | Deterministic offline `Model` for wiring tests (not an LLM) |
| `distribute_agent/plan.py` | Validate agent-produced plans before any privileged call; derive the flat-plan JSON Schema and agent instructions from the same limits |
| `distribute_agent/bridge.py` | Host-side, one-shot submission via the real packaged runner; never retries |
| `distribute_agent/live_eval.py` | Opt-in 5×3 semantic harness with enable flag and request/budget bounds |
| `distribute_agent/vendor_smoke.py` | Opt-in one-job vendor smoke; complete PNG (chunks + CRCs + IEND) or JPEG (SOF + EOI) check |
| `distribute_agent/__main__.py` | CLI: `live-eval`, `vendor-smoke` |

## Environment (verified)

| Component | Version |
|---|---|
| uv | 0.13.1 |
| Prototype interpreter | CPython 3.12.15 (uv-managed) |
| smolagents | 1.26.0 (pinned in `pyproject.toml`) |
| bubblewrap | 0.13.1 |
| Sandbox runtime interpreter | CPython 3.14.8 (`/usr/bin/python3`) |
| Kernel / libc | Linux 7.2.10, glibc (Debian/Arch layout: stdlib + loader under `/usr/lib`) |

```bash
# Reproducible setup: any CPython 3.12 venv works; /tmp/da-venv is the example used
# for the results in this file. The prototype's own interpreter must be 3.12.
PYTHON312=$(command -v python3.12)
$PYTHON312 -m venv /tmp/da-venv
/tmp/da-venv/bin/python -m pip install "smolagents==1.26.0"
# only for the live-eval CLI's default litellm backend (pyproject extra "litellm"):
/tmp/da-venv/bin/python -m pip install "smolagents[litellm]==1.26.0"
```

`uv` is optional and equivalent: `uv venv --python 3.12 <venv>` then
`uv pip install --python <venv>/bin/python "smolagents==1.26.0"`. Pillow arrives
with smolagents (`pillow>=10.0.1`) and is what the vendor smoke test decodes
with; no extra package is installed for image checking.

The base dependency covers the offline tests and the `inference-client` live
backend. The `litellm` extra is only needed for `--model-backend litellm`, which
is the CLI default; the CLI preflights that import before the first paid call.
The prototype needs no network at runtime and no vendor credential for tests.

## Run

The prototype suite lives with the prototype, because it needs smolagents and
Python 3.12 and the repository's own suite must keep running on system Python:

```bash
cd experiments/distribute_agent
/tmp/da-venv/bin/python -m unittest discover -s tests -v

cd ../..
python3 -m unittest tests.test_rp_oneshot -v     # stdlib-only runner tests
python3 -m unittest discover -s tests            # the repository suite
```

Run the prototype suite with the pinned interpreter. Without smolagents it fails
with the install command above instead of skipping, so a security test cannot
silently pass. It is not part of the repository's default discovery.

## How a run works

1. **Skill at startup.** `load_skill` parses the real `SKILL.md`; its description
   and body become the agent's `instructions`. A malformed file raises before a
   model or sandbox is involved.
2. **Package metadata.** `Package.load` validates `manifest.json`,
   `endpoints.json`, the workflows and `bin/rp_oneshot.py`. Template family,
   checkpoint, trigger word and size guidance are read from the manifest and
   included in the instructions.
3. **Code generation.** `CodeAgent` (smolagents 1.26) writes Python. Its only
   declared tool is `run_oneshot_preview`. The stock prompt's claim that
   variables persist between code blocks is replaced with the truth — every block
   is a fresh process, so state crosses blocks as files under `/workspace`.
4. **Sandboxed execution.** The executor runs the program inside bubblewrap. The
   prelude of that program defines `run_oneshot_preview`, which executes the real
   packaged runner with `--dry-run` and returns the parsed envelope, argv, exit
   code and stderr. `final_answer` writes the result back to the host.
5. **Final-answer contract (explicit per run).** A run whose answer will be
   submitted is built with `plan_manifest=package.manifest`. Its instructions
   carry the flat-plan JSON Schema and the rules the schema cannot express, and
   smolagents' native `final_answer_checks` call the same `parse_plan` the runner
   uses. A rejected answer becomes an `AgentError` in the loop, so the model sees
   the reason and can correct it within the existing step, request and wall-clock
   budgets. `_handle_max_steps_reached` bypasses those hooks, so `run_agent`
   validates the returned answer again and raises `AgentRunError` with redacted
   diagnostics instead of returning an invalid plan. `build_agent`'s default
   (`plan_manifest=None`) is a generic wiring or refusal run with no output
   contract; the live generation briefs and any future submission caller pass a
   manifest.
6. **Privileged submission (outside the sandbox).** `bridge.submit_plan`
   validates the agent's plan against the manifest, then runs the real runner
   once with `--wait`, using `RUNPOD_API_KEY` in a restricted child environment.
   Generated code cannot call it.

## Execution boundary

`LocalPythonExecutor` is not used: SmolAgents documents it as a best-effort AST
guard, not a security sandbox. Every generated program runs in a separate
process under:

- `--unshare-all` (user, mount, network, PID, IPC, UTS namespaces), `--new-session`,
  `--die-with-parent`, `--cap-drop ALL`;
- `--clearenv` plus `PATH=/usr/bin`, `HOME=/workspace`, `TMPDIR=/tmp`,
  `PYTHONDONTWRITEBYTECODE=1` — no `RUNPOD_API_KEY` or other host variable;
- read-only binds for the interpreter, `/usr/lib` (stdlib and loader), `/usr/lib`
  again as `/lib64`, and the package at `/package`;
- one writable bind of the per-run workspace at `/workspace`, with `/tmp` as a
  fresh tmpfs;
- no host home, `/etc`, `/bin`, `/sbin` or command directories, and no network.

`experiments/distribute_agent/tests/test_distribute_agent.py` verifies host files
are unreadable, credentials are absent, the package is read-only, `/workspace`
accepts writes, `sh`, `curl` and `runpodctl` are unavailable, and a socket
connection fails. A missing bubblewrap, blocked user namespaces, or a runtime
layout that cannot be mounted raises `SandboxUnavailable` with an actionable
message; there is no in-process fallback. This is an OS boundary, not a claim of
arbitrary-Python safety beyond these mounts and namespaces.

Generated code shares `/workspace` with the host, so the exchange files are
treated as hostile input:

- `program.py` is written to a fresh `O_EXCL|O_NOFOLLOW` file and moved into place
  with `os.replace`, which swaps the directory entry instead of following a
  symlink the previous action planted;
- `result.json` is opened with `O_NOFOLLOW | O_NONBLOCK`, must be a regular file
  according to `fstat`, and is read under a byte cap before parsing. `O_NONBLOCK`
  matters: generated code can leave a FIFO there, and a plain open would block the
  host past every sandbox deadline. `preview-evidence.jsonl` is read the same way;
- stdout and stderr are drained into bounded buffers; a program that keeps
  writing is killed at `max_capture_bytes` instead of filling host memory.

## Live semantic evaluation (opt-in, not run)

Five briefs × three repetitions: simple, explicit settings (seed 42 / 832×1216),
Unicode and quotes, unknown template, unsupported image. Explicit settings must
survive verbatim, the brief's prompt must carry its subject/action/setting term
groups, the Unicode prompt must match exactly, and a generation run must show a
successful `run_oneshot_preview` whose dry-run request body matches the returned
plan. A source substring, a comment, an unexecuted branch or a failed preview is
not evidence. The two rejection briefs need no preview at all: a correct refusal
made straight from the supplied manifest passes.

The evidence is recorded by the sandbox prelude and validated by the host: exit
0, a `--dry-run` argv, and a parsed request body whose manifest-declared nodes
carry the plan's settings. It shows the capability ran and succeeded. Generated
code shares the workspace, so this is not tamperproof attestation, and the
term-group checks are keyword heuristics, not complete semantic verification of
the image prompt.

The invalid briefs must not produce a submittable plan and must state a relevant
actionable reason. The report records the smolagents version, executor, skill
SHA-256, the caller's model metadata, provider requests per run, checks and
failures.

Generation briefs are built with the strict flat-plan contract
(`plan_manifest=package.manifest`), so a nested wrapper, an alias such as
`negative_prompt`, a preview envelope or a bad type/range is rejected inside the
agent loop and again at the host boundary. The two rejection briefs are built as
prose-refusal runs; a plain-text refusal is expected only there, and generation
runs cannot use prose to bypass the schema.

`--max-requests` is enforced at the model boundary by `model_budget.py`: each
provider call, including the final-answer call, reserves from one shared counter,
and a call that outlives the remaining wall-clock budget is abandoned instead of
blocking the run.

```bash
cd experiments/distribute_agent
HF_TOKEN=... /tmp/da-venv/bin/python -m distribute_agent live-eval \
  --enable-live-model --model-backend litellm --model-id <provider/model> \
  --package ../../distribute --workspace /tmp/distribute-agent-live \
  --output /tmp/distribute-agent-live --max-requests 15
```

Without `--enable-live-model` the command exits 2 and prints `status: not-run`.
The key is read from `--api-key-env` and is not recorded.

## Vendored smoke test (opt-in, not run)

One billed job with a fixed plan; verifies a job id, an output directory, and an
artifact that decodes to real pixels. The bytes are checked locally with Pillow
under a 64 MiB input cap and a 16 MP / 8192-per-side canvas cap: `verify()` walks
the container (chunk structure, CRCs, PNG terminator) and `load()` inflates every
pixel, so an empty or invalid IDAT, a JPEG that stops after its frame header, a
truncated download and a decompression bomb all fail. The image is never
displayed, written elsewhere or sent to a model. Duration and budget are checked
too.

```bash
cd experiments/distribute_agent
RUNPOD_API_KEY=... /tmp/da-venv/bin/python -m distribute_agent vendor-smoke \
  --enable-live-vendor --package ../../distribute --output /tmp/distribute-agent-smoke \
  --budget-seconds 900
```

Without `--enable-live-vendor` the command exits 2 with `status: not-run`. The
key never appears in the report.

## Test coverage map

| Plan ID | Where |
|---|---|
| S1 skill consumed at startup, marker changes scripted behavior | `SkillLoadingTests.test_marker_instruction_changes_observable_agent_behavior` |
| S2 other cwd and paths with spaces | `PackageResolutionTests.test_package_resolves_from_another_cwd_and_a_path_with_spaces` |
| S3 missing/malformed skill, manifest, workflow, endpoint | `SkillLoadingTests`, `PackageResolutionTests`, `ValidationTests` in both test files |
| S4 package bytes unchanged, outputs separate | runner `DryRunTests`/`OutputTests`, `SandboxBoundaryTests`, `AgentBoundaryTests` |
| P1 prompt content preserved / live semantics separate | `CodeAgentDryRunTests`, `AgentBoundaryTests.test_shell_metacharacters_stay_data`, `LiveHarnessTests` |
| P2 family/trigger from manifest, recipes preserved | `SkillLoadingTests.test_skill_recipes_and_manifest_metadata_reach_the_instructions` |
| P3 seed 42 / 832×1216 / template negative | runner `DryRunTests.test_prompt_negative_seed_and_size_are_preserved_as_data` |
| P4 invalid requests rejected before vendor | runner `ValidationTests`, `PlanValidationTests` |
| E1 actual CodeAgent → real runner dry run | `CodeAgentDryRunTests.test_actual_codeagent_executes_the_real_runner_dry_run` |
| E2 metacharacters are data | `AgentBoundaryTests.test_shell_metacharacters_stay_data` |
| E3 dry run: no key, no HTTP, no output files | runner `DryRunTests`, `AgentBoundaryTests.test_dry_run_writes_only_the_program_and_no_key_is_visible` |
| E4 step and wall-clock bounds, redacted diagnostics | `AgentBoundaryTests` (budget, max steps, redaction) |
| V1 one POST, same job id, poll | runner `SubmissionTests.test_one_post_then_polls_the_same_job_and_saves_the_output` |
| V2 terminal/auth/malformed/timeout, no resubmit | runner `SubmissionTests` |
| V3 10 MiB / 20 MiB caps before HTTP | runner `RequestLimitTests` |
| V4 artifacts, partial failure, empty/corrupt | runner `OutputTests` |
| A1 isolated histories/workspaces concurrently | `AgentBoundaryTests.test_concurrent_runs_keep_workspaces_and_answers_isolated`, runner `OutputTests` |
| A2 OS-enforced denials | `SandboxBoundaryTests` (incl. planted symlinks, FIFO result path, oversized results) |
| A3 opt-in 5×3 semantic harness | `LiveHarnessTests` (refusal, matrix, term-group checks, strict generation vs prose-refusal contract, real preview evidence for comment-only / unexecuted / failed / mismatched / matching cases; not run live) |
| A4 opt-in one-job vendor smoke | `VendorSmokeTests` (gate, key, artifact plumbing, decode of empty-IDAT, invalid-IDAT, frame-header-only and truncated artifacts, Pillow PNG/JPEG positives; not run live) |

## Known gaps and risks

- The sandbox is Linux/glibc specific and depends on unprivileged user namespaces.
  It mounts `/usr/lib` read-only, which also exposes installed Python packages
  there; those packages have no credentials and no network, but this is broader
  than the strict stdlib.
- No seccomp filter is installed; the boundary is mounts + namespaces + capability
  drop, verified by tests, not a hardened container profile.
- A live model's compliance is measured only by the opt-in harness, which has not
  been run with authorization from this repository's committed evidence; the
  scripted-model tests prove wiring only. An earlier live run recorded outside the
  suite (9/15: settings and Unicode briefs failing on nested fields and step
  exhaustion) predates the strict final-answer contract and was not re-run for
  this change. No live improvement is claimed, and a strict contract can turn a
  silently wrong answer into an explicit run failure instead of a pass.
- Plans and the bridge cover text-only templates. Image input requires a template
  that declares an `image` node in the manifest; the shipped package has none.
- The live harness uses smolagents' `LiteLLMModel` / `InferenceClientModel`
  construction; that path has not been executed.
- A model call that exceeds the remaining budget is abandoned as a daemon thread
  and the run fails; the underlying provider socket is not closed by this code, so
  the provider client should still set its own timeout.
- Vendor artifact validation decodes pixels, but it is a smoke-harness check, not a
  runner contract: the runner still saves whatever bytes a worker returns for any
  output kind. Decoding is capped (64 MiB, 16 MP, 8192 per side) and uses Pillow's
  own limits; a crafted stream that stays inside those caps is decoded, not
  proven safe by this prototype.

## Review fixes

The independent review of this prototype found one merge blocker and several
correctness gaps; all are fixed and covered by tests:

| Finding | Fix |
|---|---|
| P0: symlink planted at `program.py`/`result.json` redirects host reads and writes | `sandbox.write_program` (fresh `O_EXCL\|O_NOFOLLOW` file + `os.replace`) and `sandbox.read_result_bytes` (`O_NOFOLLOW`, regular file, byte cap); regression tests with harmless temp targets |
| P1: executor contradicts the prompt's persistent-state promise | `agent.code_agent_prompt_templates` replaces that rule; two-step test expects a `NameError` and a recovered answer |
| P1: unbounded stdout/stderr capture and result parsing | streaming `_drain` kills at `max_capture_bytes`; result reads capped by `max_result_bytes` |
| P1: bridge lost the "already submitted" signal | runner prints `[info] job <id> accepted` and `submission outcome is UNKNOWN`; bridge parses both and exposes `BridgeError.job_id` |
| P1: semantic harness passed unrelated plans and empty rejections | per-brief subject terms, exact Unicode prompt, rejection term groups, preview evidence; offline wrong-answer tests |
| P1: request budget counted runs, not provider calls | `model_budget.BudgetedModel` counts every `generate` call and enforces the remaining wall clock |
| P1: vendor smoke passed a truncated PNG | complete PNG/JPEG structure check; truncated and corrupt fixtures fail |
| P2: default live backend dependency undeclared | `[project.optional-dependencies] litellm` plus a preflight import before the matrix |
| P2: prototype suite broke repository test discovery | suite moved to `experiments/distribute_agent/tests/` with its own invocation |
| P2: artifact write failures escaped the runner's handler | write moved inside the per-artifact `try`; mocked write-failure test |

The re-review of those fixes found three more P1 gaps; they are fixed and covered
too:

| Finding | Fix |
|---|---|
| P1: a FIFO at `result.json` hangs the host in `open` past every deadline | `read_result_bytes` opens with `O_NOFOLLOW\|O_NONBLOCK`, then rejects anything `fstat` does not report as a regular file; the regression runs the read in a timed child process so a relapse fails instead of hanging the suite |
| P1: preview evidence was a source substring plus an executor call count, and rejection briefs were forced to preview | the prelude records each preview invocation and exit status; the executor host-validates the records (exit 0, `--dry-run`, parsed request body) and `live_eval` matches the manifest-declared request settings against the returned plan; rejection briefs need no preview; offline cases cover comment-only, unexecuted, failed and mismatched previews |
| P1: vendor smoke accepted structurally valid bytes with no usable image | `sniff_image` verifies the container and loads every pixel locally with Pillow under byte/dimension/pixel caps; empty-IDAT, invalid-IDAT and frame-header-only fixtures fail while real PNG/JPEG pass |

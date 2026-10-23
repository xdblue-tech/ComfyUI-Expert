# Private media endpoint

A loopback-only FastMCP server that turns a natural-language brief into a
validated image plan, submits the plan to the RunPod endpoint through the
existing packaged runner, and serves the validated artifact back over MCP or
HTTP. One process owns one state directory and one generation worker.

The endpoint is a **resource server**: it verifies HS256 bearer tokens but does
not run an OAuth authorization server. Tokens are minted by the `token`
subcommand from a shared secret in the service environment.

## Install

The agent package works without the transport. The endpoint needs the `server`
extra (FastMCP 4.1.3, Starlette, Uvicorn, joserfc, huggingface-hub):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[server]'
```

Bubblewrap and unprivileged user namespaces are still required: planning runs
generated Python and the runner dry-run inside the existing sandbox.

## Configure

| Variable | Required | Purpose |
|---|---|---|
| `MEDIAGEN_JWT_SECRET` | yes | Shared HS256 signing secret, at least 32 bytes. Generate it once per deployment. |
| `RUNPOD_API_KEY` | for generation | Vendor key read by the bridge. Never written to artifacts or logs. |
| `MEDIAGEN_PROXY_API_KEY` | for planning | Key for the OpenAI-compatible planning proxy. |

Set the secret without putting it on a command line:

```bash
export MEDIAGEN_JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
```

Generate a stable secret once and store it in the operator's secret manager;
restarting the service with a different secret invalidates every issued token.

## Serve

```bash
.venv/bin/python -m distribute_agent serve \
  --package distribute \
  --state-dir state/mediagen \
  --output-dir output/mediagen \
  --port 8000
```

Defaults: host `127.0.0.1`, port `8000`, MCP path `/mcp`, stateless Streamable
HTTP, generation **disabled**. `--enable-generation` is required, and generation
still stays disabled unless `RUNPOD_API_KEY` is present in the environment.

`--model-id`, `--model-base-url` and `--model-api-key-env` override the pinned
planning model (`glm-5.3-flash` through `http://hades.home:8317/v1`). The
`--auth-secret-env`, `--issuer`, `--audience` and `--scope` flags must match the
values used when minting tokens.

The host must resolve to a loopback address (`127.0.0.1`, `::1` or
`localhost`); startup refuses anything else. Reach a remote host through an SSH
tunnel; the service does not terminate TLS:

```bash
ssh -N -L 8000:127.0.0.1:8000 operator@media-host
```

## Issue, renew and rotate tokens

```bash
# default: subject owner, issuer mediagen, audience mediagen-mcp,
# scope media:access, lifetime 7 days
.venv/bin/python -m distribute_agent token

# shorter-lived token for one client
.venv/bin/python -m distribute_agent token --subject laptop --ttl-seconds 86400
```

The command prints only the token on stdout. Renew by issuing a new token before
the old one expires; there is no refresh endpoint.

Rotate the shared secret by generating a new value, restarting the service with
it, and reissuing every client token. Old tokens fail signature validation
immediately. A token minted by another tool that omits `exp` is accepted by
FastMCP 4.1.3 (it checks expiry only when present), so mint tokens only with this
command.

## Client configuration

Register the server as a Streamable HTTP MCP server at `http://127.0.0.1:8000/mcp`
and send the token as a bearer header:

```json
{
  "mcpServers": {
    "mediagen": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

Every MCP request needs the token. The one public route is `GET /health`, which
returns `{"status": "ok"}` for process supervision.

## Tools

| Tool | Input | Result |
|---|---|---|
| `plan_media` | `request` (1–8192 characters) | `plan`, `package_digest`, `model_id`, bounded `diagnostics` |
| `start_generation` | raw `plan` object, `package_digest`, `request_id` | persisted job: id, status, plan, digest, timestamps |
| `get_generation` | `job_id` | status, timestamps, plan, optional vendor id, artifact metadata, optional error |

`request_id` must match `[A-Za-z0-9._-]{1,128}`. It is the idempotency key: an
exact retry is checked against the request's own immutable snapshot and returns
the existing job even after a live package edit; a changed body under the same
id fails with `request_conflict`. Concurrent duplicates still produce exactly
one submission attempt.

`plan` is passed to the strict parser as raw JSON. Numeric strings (`"seed":
"42"`), booleans, unknown fields and aliases are rejected, not coerced.

`start_generation` revalidates the plan, runs the real packaged runner with
`--dry-run` and no credential, checks the effective output size (template
defaults included) against the decoder limits (8192 pixels per side, 16 Mi
pixels), then persists the job before acknowledging it.

Statuses are `queued`, `running`, `succeeded`, `failed`, `needs_attention`. An
empty, missing, partial or corrupt published image cannot succeed.

## Artifacts

`get_generation` returns, per artifact, an opaque `artifact_id`, MIME type,
width, height, byte size and SHA-256. There are two retrieval paths:

| Path | Auth | Cap |
|---|---|---|
| MCP resource `media://jobs/{job_id}/artifacts/{artifact_id}` | MCP bearer token | 8 MiB inline |
| `GET /artifacts/{job_id}/{artifact_id}` | Explicit bearer check with the same verifier | 64 MiB |

The HTTP route is a custom route and does **not** inherit MCP auth; it verifies
the token itself. It returns the stored MIME type, `X-Content-Type-Options:
nosniff`, and `Cache-Control: no-store`. Reads are opaque database lookups,
confined under `output/mediagen/<job_id>`, with no-follow and regular-file-only
checks; the stored size and SHA-256 are rechecked on every read.

Files larger than 8 MiB must use HTTP. Both paths return `artifact_too_large`
rather than truncating. Artifacts stay until the operator removes them; there is
no automatic retention.

## Errors

Failures use stable, redacted code prefixes in their message or response body:

| Code | Meaning |
|---|---|
| `invalid_input` | Request, plan, id or dimensions rejected |
| `plan_failed` | The model run or strict plan parse failed |
| `generation_disabled` | Flag off or `RUNPOD_API_KEY` absent |
| `busy` | A planner is already running |
| `capacity` | 20 jobs are already waiting |
| `stale_digest` | Digest is not the package this process serves |
| `request_conflict` | Request id already names a different body |
| `unknown_job`, `unknown_artifact` | No such persisted id |
| `artifact_too_large`, `artifact_corrupt` | Read cap or bytes changed/missing |
| `storage_error` | SQLite or artifact filesystem failure (including full disk) |
| `state_locked`, `shutting_down` | Another instance owns state, or service stopping |
| `submission_uncertain`, `submission_failed` | Vendor outcome unknown, or a definite failure |

Messages are bounded, path-scrubbed and credential-scrubbed; the tool error text
starts with the code. Runner and vendor response bodies are not copied into
persisted errors: a submission failure carries an allowlisted summary plus the
structured vendor id and uncertainty flag. `needs_attention` retains any vendor
job id: check the RunPod job before resubmitting. The service never retries a
submission.

## Host and origin protection

FastMCP's guard is enabled in `auto` mode: on a loopback bind it validates every
request. A foreign `Host` gets `421`; a foreign browser `Origin` gets `403`.
Additional trusted values are explicit:

```bash
--allow-host media.internal --allow-origin http://media.internal:8000
```

Do not use `*` unless the port is already behind a trusted proxy.

## Lifecycle and durability

- Startup takes an exclusive lock on `<state-dir>/runtime.lock`, pins a
  content-addressed copy of only `manifest.json`, `endpoints.json`, `bin/`,
  `workflows/` and `skills/`, opens SQLite schema version 1, and marks any job
  left `running` as `needs_attention`.
- The live package is read only at startup. Restart the service to serve edited
  package files. Queued jobs load the snapshot named by their stored digest, so a
  later package edit cannot change a submission already accepted. An exact retry
  of an existing `request_id` is validated against that same stored snapshot
  before the current package is consulted.
- One planner runs at a time; `busy` is returned instead of queueing planners.
  One worker processes jobs. Twenty queued jobs is the admission bound.
- A client disconnect does not cancel a paid generation; the worker is
  independent of the HTTP request. A cancelled planner request keeps its
  planner admission until the planning operation finishes; an idle thread-pool
  thread does not hold shutdown open.
- Shutdown stops admission and claims, joins the worker and waits for any
  admitted planner operation within one shared shutdown bound, and only then
  releases the lock. The planner wait is on operation completion, not on the
  lifetime of an executor-owned pool thread. If the bound expires, the process
  must exit; the lock stays held until it does. A forced exit may leave remote
  work running. There is no remote cancellation and no exactly-once vendor
  guarantee.

Runtime files: `<state-dir>/jobs.sqlite3`, `<state-dir>/runtime.lock`,
`<state-dir>/packages/<digest>/`, `<state-dir>/workspaces/`; artifacts under
`<output-dir>/<job_id>/`. `state/mediagen/` and `output/mediagen/` are
gitignored.

## Verification

The committed evidence is offline. No live model request and no billed RunPod
job were made for this endpoint.

```bash
python3 -m unittest discover -s tests                     # 151 stdlib tests
.venv/bin/python -m unittest discover -s tests/agent      # 72 agent tests
.venv/bin/python -m unittest discover -s tests/service    # 126 service tests
uv pip check --python .venv/bin/python                    # dependency closure
```

`tests/service/test_tracer.py` is the complete offline tracer: authenticated MCP
client → real `CodeAgent` with a scripted model → real bubblewrap → real runner
`--dry-run` → SQLite → the real bridge → the real runner against a localhost
fake RunPod API → validated image → authenticated MCP and HTTP retrieval. It
asserts one POST despite duplicate request ids, seed 42 / 832×1216 effective
settings, and byte/SHA-256 equality. The test wrapper is injected through the
production `runner_prefix` seam and changes only the imported runner's vendor
base; the production URL is untouched.

## Troubleshooting

| Symptom | Check |
|---|---|
| Process exits 3 at startup | Read the traceback: another instance holds the lock, the package is invalid, or the state schema is incompatible. |
| `error: MEDIAGEN_JWT_SECRET is not set` | Export a secret of at least 32 bytes before `serve` or `token`. |
| Every MCP call returns 401 | Token expired, wrong issuer/audience/scope, or signed with a different secret. |
| `generation_disabled` | Add `--enable-generation` **and** export `RUNPOD_API_KEY`, then restart. |
| `stale_digest` | Call `plan_media` again; the process is serving a different package snapshot. |
| `busy` | Wait for the running planner; the endpoint does not queue planners. |
| Artifact read returns 413 | Use `GET /artifacts/...`; the MCP resource is capped at 8 MiB. |
| `needs_attention` with a vendor id | Inspect that RunPod job before running anything else; do not resubmit blindly. |

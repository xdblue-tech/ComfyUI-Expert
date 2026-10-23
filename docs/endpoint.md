# Private media endpoint

A FastMCP server (loopback by default) that turns a natural-language brief into a
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
| `MEDIAGEN_HOST` | no | Bind address for `serve`; default `127.0.0.1`. |
| `MEDIAGEN_PORT` | no | TCP port for `serve`; default `8000`. |
| `MEDIAGEN_MODEL_ID` | no | Planning model id; default `glm-5.3-flash`. |
| `MEDIAGEN_MODEL_BASE_URL` | no | OpenAI-compatible planning proxy; default `http://hades.home:8317/v1`. |
| `MEDIAGEN_MODEL_API_KEY_ENV` | no | Name of the variable holding the proxy key; default `MEDIAGEN_PROXY_API_KEY`. |
| `MEDIAGEN_ENABLE_GENERATION` | no | Strict boolean for paid generation; default `false`. |
| `MEDIAGEN_JWT_SECRET` | yes | Shared HS256 signing secret, at least 32 bytes. Generate it once per deployment. |
| `RUNPOD_API_KEY` | for generation | Vendor key read by the bridge. Never written to artifacts or logs. |
| `MEDIAGEN_PROXY_API_KEY` | for planning | Key for the OpenAI-compatible planning proxy. |

Set the secret without putting it on a command line:

```bash
export MEDIAGEN_JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
```

Generate a stable secret once and store it in the operator's secret manager;
restarting the service with a different secret invalidates every issued token.

The CLI does not load `.env` itself; source it first (`set -a; source .env;
set +a`) or export the values. Exported serve settings are read only when the
matching flag is absent, so `--host`, `--port`, `--model-id`,
`--model-base-url`, `--model-api-key-env`,
`--enable-generation`/`--disable-generation` always win. That holds even over
an unparsable exported port or boolean; `--disable-generation` explicitly turns
off an exported true, and `--enable-generation` turns it on when the export is
absent or false. Empty or unset values fall back to the pinned defaults.
`MEDIAGEN_ENABLE_GENERATION` accepts `true`/`1`/`yes`/`on` or
`false`/`0`/`no`/`off`, case-insensitive after trimming whitespace; empty means
false, and any other non-empty value stops startup with exit 2.

## Serve

```bash
.venv/bin/python -m distribute_agent serve \
  --package distribute \
  --state-dir state/mediagen \
  --output-dir output/mediagen \
  --port 8000
```

Defaults: host `127.0.0.1`, port `8000`, MCP path `/mcp`, stateless Streamable
HTTP, planning model `glm-5.3-flash` through `http://hades.home:8317/v1`, and
generation **disabled**. `--host`, `--port`, `--model-id`, `--model-base-url`,
`--model-api-key-env` and the generation flags override their exports; empty or
unset exports fall back to the defaults. The host must be an IP literal
(`127.0.0.1`, `::1`, a LAN address, or the wildcard `0.0.0.0` or `::`) or
`localhost`; startup refuses anything else, including a non-numeric exported
port or a malformed generation boolean.

The `--auth-secret-env`, `--issuer`, `--audience` and `--scope` flags must match
the values used when minting tokens. Paid work needs generation enabled by
`--enable-generation` or `MEDIAGEN_ENABLE_GENERATION` (`--disable-generation`
forces it off) **and** `RUNPOD_API_KEY` in the environment; without the key
generation stays disabled.

Reach a remote host through an SSH tunnel when the service stays on loopback;
it does not terminate TLS:

```bash
ssh -N -L 8000:127.0.0.1:8000 operator@media-host
```

A LAN bind (`MEDIAGEN_HOST=0.0.0.0` or a LAN address) exposes the endpoint
directly and therefore needs a TLS-terminating proxy in front of it. The
Host/Origin guard described below only covers a wildcard bind when the client's
host is listed explicitly; JWT auth applies to every MCP request and the
artifact route on any bind.

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
| `start_generation` | `prompt` (1–8192 characters), `specs` | persisted job: id, status, plan, digest, diagnostics, timestamps |
| `get_generation` | `job_id` | status, timestamps, plan, diagnostics, optional vendor id, artifact metadata, optional error |

`start_generation` is the only mutating call: it plans the brief, applies the
specs overrides and enqueues the job in one round trip. `specs` is a strict JSON
object:

| Field | Rule |
|---|---|
| `id` | Required, `[A-Za-z0-9._-]{1,128}`. The idempotency key. |
| `category` | Optional, one of `sd-semireal` (the default). |
| `width`, `height` | Optional integers, a multiple of 8. The effective output must stay within the decoder limits: 8192 pixels per side and 16 Mi pixels. |
| `seed` | Optional integer, 0 to 2^64-1. Omit for the template's random-seed policy. |

The category picks the plan's template, so the client-facing name keeps meaning
the same thing when the package changes. Unknown keys, an unknown category, a
non-integer number and an out-of-range value are rejected with `invalid_input`
before any model call.

`id` is the idempotency key: a repeat is answered from the store, without
planning again, and returns the existing job even after a live package edit. A
changed prompt or specs under the same id fails with `request_conflict`. One
planner runs at a time, so a concurrent duplicate either replays the job or is
told `busy`; it never creates a second job or a second submission.

`start_generation` plans against the package snapshot this process pinned, then
revalidates the merged plan, runs the real packaged runner with `--dry-run` and
no credential, and checks the effective output size (template defaults included)
against the decoder limits (8192 pixels per side, 16 Mi pixels), before
persisting the job. A planning or preflight failure leaves no job row and costs
nothing at the vendor.

`diagnostics` on the job is the bounded planning summary: duration, model
requests, agent steps, validated previews and the agent's final state.

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
request. On a wildcard or LAN bind it validates only when a host is listed
explicitly, so pass the name or address clients use (for example
`--allow-host 192.168.1.10`) to keep DNS-rebinding protection. A foreign `Host`
gets `421`; a foreign browser `Origin` gets `403`. Additional trusted values are
explicit:

```bash
--allow-host media.internal --allow-origin http://media.internal:8000
```

Do not use `*` unless the port is already behind a trusted proxy.

## Lifecycle and durability

- Startup takes an exclusive lock on `<state-dir>/runtime.lock`, pins a
  content-addressed copy of only `manifest.json`, `endpoints.json`, `bin/`,
  `workflows/` and `skills/`, opens SQLite schema version 2 (migrating a
  version 1 database in place by adding the request, specs and diagnostics
  columns), and marks any job left `running` as `needs_attention`.
- The live package is read only at startup. Restart the service to serve edited
  package files. Queued jobs load the snapshot named by their stored digest, so a
  later package edit cannot change a submission already accepted. An exact retry
  of an existing `request_id` is answered from the stored request text and specs
  without planning again or consulting the live manifest.
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
.venv/bin/python -m unittest discover -s tests/service    # 147 service tests
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
| `busy` | Wait for the running planner; the endpoint does not queue planners. |
| Artifact read returns 413 | Use `GET /artifacts/...`; the MCP resource is capped at 8 MiB. |
| `needs_attention` with a vendor id | Inspect that RunPod job before running anything else; do not resubmit blindly. |
